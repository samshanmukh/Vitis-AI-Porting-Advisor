# agent_repo_scanner.py
# Phase 1 — Preliminary investigation on model architecture.
#
# Statically scans a PyTorch model *repository* (source code, not a compiled graph):
#   • finds the nn.Module classes that make up the architecture
#   • lists every layer / function the model uses, with file:line provenance
#   • maps each construct to the Vitis AI DPU op vocabulary so the RAG (agent_two)
#     can judge compatibility
#
# Deterministic (pure AST) — no network, no LLM. The LLM explanation and the local
# capture happen in separate agents.
import ast
import os

# torch / nn / functional construct  ->  (DPU op vocabulary, category, human note)
# DPU op names line up with the vocabulary seeded in setup_vitis_rag.py so the
# existing compatibility RAG can be reused unchanged.
CONSTRUCT_MAP = {
    # convolutions
    "Conv1d": ("Conv", "layer", "1D convolution"),
    "Conv2d": ("Conv", "layer", "2D convolution"),
    "Conv3d": ("Conv", "layer", "3D convolution"),
    "ConvTranspose2d": ("ConvTranspose", "layer", "transposed / deconv"),
    "conv2d": ("Conv", "functional", "F.conv2d"),
    # normalization
    "BatchNorm1d": ("BatchNormalization", "layer", "batch norm"),
    "BatchNorm2d": ("BatchNormalization", "layer", "batch norm"),
    "GroupNorm": ("GroupNormalization", "layer", "group norm — often CPU"),
    "LayerNorm": ("LayerNormalization", "layer", "layer norm — often CPU"),
    "InstanceNorm2d": ("InstanceNormalization", "layer", "instance norm — CPU"),
    # linear / matmul
    "Linear": ("Gemm", "layer", "fully connected"),
    "matmul": ("MatMul", "functional", "torch.matmul"),
    "bmm": ("MatMul", "functional", "batched matmul"),
    "einsum": ("Einsum", "functional", "einsum — not DPU-native"),
    # pooling
    "MaxPool2d": ("MaxPool", "layer", "max pool"),
    "AvgPool2d": ("AveragePool", "layer", "avg pool"),
    "AdaptiveAvgPool2d": ("GlobalAveragePool", "layer", "adaptive avg pool"),
    "AdaptiveAvgPool1d": ("GlobalAveragePool", "layer", "adaptive avg pool"),
    "max_pool2d": ("MaxPool", "functional", "F.max_pool2d"),
    "avg_pool2d": ("AveragePool", "functional", "F.avg_pool2d"),
    # activations
    "ReLU": ("Relu", "layer", "ReLU"),
    "ReLU6": ("Relu", "layer", "ReLU6"),
    "relu": ("Relu", "functional", "F.relu"),
    "LeakyReLU": ("LeakyRelu", "layer", "leaky ReLU"),
    "leaky_relu": ("LeakyRelu", "functional", "F.leaky_relu"),
    "Sigmoid": ("Sigmoid", "layer", "sigmoid — DPU exit risk"),
    "sigmoid": ("Sigmoid", "functional", "sigmoid — DPU exit risk"),
    "SiLU": ("SiLU", "layer", "SiLU/Swish = Sigmoid×x — needs replacement"),
    "silu": ("SiLU", "functional", "SiLU/Swish — needs replacement"),
    "Softmax": ("Softmax", "layer", "softmax — not DPU-native"),
    "softmax": ("Softmax", "functional", "softmax — not DPU-native"),
    "Tanh": ("Tanh", "layer", "tanh"),
    "GELU": ("Gelu", "layer", "GELU — CPU fallback"),
    "gelu": ("Gelu", "functional", "GELU — CPU fallback"),
    "Hardswish": ("HardSwish", "layer", "hard-swish"),
    "Hardsigmoid": ("HardSigmoid", "layer", "hard-sigmoid (DPU-friendly)"),
    # shape / resize / concat
    "Upsample": ("Resize", "layer", "upsample/resize"),
    "interpolate": ("Resize", "functional", "F.interpolate"),
    "cat": ("Concat", "functional", "torch.cat"),
    "stack": ("Concat", "functional", "torch.stack"),
    "view": ("Reshape", "tensor", "tensor reshape"),
    "reshape": ("Reshape", "tensor", "tensor reshape"),
    "flatten": ("Reshape", "tensor", "flatten"),
    "permute": ("Transpose", "tensor", "permute"),
    "transpose": ("Transpose", "tensor", "transpose"),
    "split": ("Split", "functional", "split"),
    "chunk": ("Split", "functional", "chunk"),
    "pad": ("Pad", "functional", "F.pad"),
    # element-wise / reductions
    "add": ("Add", "functional", "elementwise add"),
    "mul": ("Mul", "functional", "elementwise mul"),
    "div": ("Div", "functional", "elementwise div — CPU-bound"),
    "mean": ("ReduceMean", "functional", "reduce mean"),
    "sum": ("ReduceSum", "functional", "reduce sum"),
    "sqrt": ("Sqrt", "functional", "sqrt — CPU"),
    "exp": ("Exp", "functional", "exp — CPU"),
    "Dropout": ("Dropout", "layer", "dropout (no-op at inference)"),
}

# kwargs worth capturing per construct — these drive the architecture-specific "why"
KWARGS_OF_INTEREST = {
    "Conv2d", "Conv1d", "Conv3d", "ConvTranspose2d", "conv2d",
    "MaxPool2d", "AvgPool2d", "Upsample", "interpolate",
}
_KW_KEYS = {"dilation", "groups", "kernel_size", "stride", "padding", "mode", "scale_factor", "size"}


def _dotted(node) -> str | None:
    """Resolve an ast call target to a dotted name, e.g. nn.Conv2d / F.interpolate / x.view."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return None


def _kwargs(call: ast.Call) -> dict:
    out = {}
    for kw in call.keywords:
        if kw.arg in _KW_KEYS:
            try:
                out[kw.arg] = ast.unparse(kw.value)
            except Exception:
                out[kw.arg] = "?"
    return out


def _is_module_base(bases) -> bool:
    for b in bases:
        name = _dotted(b) or ""
        if name.split(".")[-1] == "Module":
            return True
    return False


def scan_repo(input_data: dict) -> dict:
    """
    Input:  {"repo_path": "model_input/Transfer-Model_original"}
    Output: architecture map + construct list (+ ONNX-op adapter for agent_two).
    """
    repo_path = input_data.get("repo_path")
    if not repo_path or not os.path.isdir(repo_path):
        return {"error": f"repo_path not a directory: {repo_path}"}

    py_files = []
    for root, _dirs, files in os.walk(repo_path):
        if any(skip in root for skip in (".git", "__pycache__", ".venv", "venv", "node_modules")):
            continue
        for f in files:
            if f.endswith(".py"):
                py_files.append(os.path.join(root, f))

    modules = []          # nn.Module subclasses (the architecture)
    occurrences = []      # every mapped construct use, with provenance
    parse_errors = []

    for path in sorted(py_files):
        rel = os.path.relpath(path, repo_path)
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                tree = ast.parse(fh.read(), filename=path)
        except Exception as e:
            parse_errors.append({"file": rel, "error": str(e)})
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and _is_module_base(node.bases):
                modules.append({
                    "name": node.name,
                    "file": rel,
                    "line": node.lineno,
                    "bases": [(_dotted(b) or "?") for b in node.bases],
                })
            elif isinstance(node, ast.Call):
                dotted = _dotted(node.func)
                if not dotted:
                    continue
                simple = dotted.split(".")[-1]
                mapped = CONSTRUCT_MAP.get(simple)
                if not mapped:
                    continue
                dpu_op, category, note = mapped
                occurrences.append({
                    "construct": simple,
                    "dotted": dotted,
                    "dpu_op": dpu_op,
                    "category": category,
                    "note": note,
                    "file": rel,
                    "line": getattr(node, "lineno", 0),
                    "kwargs": _kwargs(node) if simple in KWARGS_OF_INTEREST else {},
                })

    # group occurrences by construct
    by_construct: dict[str, dict] = {}
    for occ in occurrences:
        key = occ["construct"]
        b = by_construct.setdefault(key, {
            "construct": key, "dpu_op": occ["dpu_op"], "category": occ["category"],
            "note": occ["note"], "count": 0, "locations": [],
        })
        b["count"] += 1
        b["locations"].append({"file": occ["file"], "line": occ["line"], "kwargs": occ["kwargs"]})

    constructs = sorted(by_construct.values(), key=lambda d: (-d["count"], d["construct"]))

    # adapter: feed agent_two (search_vitis_compatibility) one "operation" per occurrence
    operations = [
        {"name": f"{occ['construct']} @ {occ['file']}:{occ['line']}", "type": occ["dpu_op"]}
        for occ in occurrences
    ]

    return {
        "repo_path": repo_path,
        "modules": modules,
        "constructs": constructs,
        "operations": operations,
        "dpu_op_list": sorted({occ["dpu_op"] for occ in occurrences}),
        "summary": {
            "files_scanned": len(py_files),
            "modules": len(modules),
            "unique_constructs": len(constructs),
            "total_construct_uses": len(occurrences),
            "parse_errors": len(parse_errors),
        },
        "parse_errors": parse_errors,
        "status": "success",
    }


if __name__ == "__main__":
    import json
    res = scan_repo({"repo_path": "model_input/Transfer-Model_original"})
    print(json.dumps(res["summary"], indent=2))
    print("\nModules:")
    for m in res["modules"]:
        print(f"  {m['name']:24s} {m['file']}:{m['line']}")
    print("\nConstructs -> DPU op:")
    for c in res["constructs"]:
        print(f"  {c['construct']:18s} -> {c['dpu_op']:18s} ×{c['count']}  ({c['note']})")
