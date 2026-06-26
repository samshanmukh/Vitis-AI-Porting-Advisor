# agent_vendor_compat.py
# The moat: score the SAME model (ONNX-op vocabulary) against three accelerator
# back-ends and recommend the best target. No single vendor's tooling does this.
#
# Verdicts per op:  native (runs on the accelerator) · limited (supported w/ caveats)
#                   · unsupported (falls back to host / needs a plugin)
#
# Knowledge is curated (not exhaustive) — TensorRT/OpenVINO accept a very broad
# ONNX opset; the AMD DPU is a fixed-function INT8 edge engine, so far more falls back.

VENDORS = {
    "AMD Vitis DPU": {
        "accel": "FPGA DPU", "power": "very low (edge)", "default": "unsupported",
        "native": {
            "Conv", "ConvTranspose", "BatchNormalization", "Relu", "LeakyRelu",
            "MaxPool", "AveragePool", "GlobalAveragePool", "Concat", "Add", "Mul",
            "Gemm", "HardSigmoid", "HardSwish", "Pad", "Dropout",
        },
        "limited": {
            "Resize": "nearest/bilinear at static size only",
            "Reshape": "static shapes only",
            "Transpose": "static at boundaries only",
            "Split": "forces a subgraph boundary",
            "Div": "element-wise div is CPU-bound",
            "ReduceMean": "ok as global avg-pool",
        },
        "unsupported": {
            "Sigmoid": "mid-graph sigmoid forces CPU exit", "SiLU": "Sigmoid×x not native",
            "Softmax": "not DPU-native → CPU", "Gelu": "→ CPU", "Tanh": "→ CPU",
            "Einsum": "not native", "MatMul": "general matmul → CPU",
            "LayerNormalization": "→ CPU", "GroupNormalization": "→ CPU",
            "InstanceNormalization": "→ CPU", "Sqrt": "→ CPU", "Exp": "→ CPU",
            "Log": "→ CPU", "ReduceSum": "breaks subgraph", "NonMaxSuppression": "post-proc CPU",
            "TopK": "→ CPU",
        },
    },
    "NVIDIA TensorRT": {
        "accel": "GPU", "power": "high (datacenter/edge GPU)", "default": "native",
        "native": set(),  # default native — runs almost the entire ONNX opset
        "limited": {
            "Einsum": "via matmul rewrite / plugin",
            "NonMaxSuppression": "EfficientNMS plugin",
            "TopK": "supported, perf varies",
            "InstanceNormalization": "supported; perf-sensitive",
        },
        "unsupported": {},  # custom ops need plugins, but our vocabulary maps
    },
    "Intel OpenVINO": {
        "accel": "CPU / iGPU / NPU", "power": "low–medium", "default": "native",
        "native": set(),  # default native — broad ONNX support on CPU
        "limited": {
            "NonMaxSuppression": "supported op; NPU may fall back",
            "TopK": "supported; NPU caveats",
            "Einsum": "supported on CPU/GPU; NPU may fall back",
        },
        "unsupported": {},
    },
}


def _verdict(cfg: dict, op: str):
    if op in cfg["unsupported"]:
        return "unsupported", cfg["unsupported"][op]
    if op in cfg["limited"]:
        return "limited", cfg["limited"][op]
    if op in cfg["native"]:
        return "native", ""
    return cfg["default"], ""


def compare_vendors(input_data: dict) -> dict:
    """
    Input:  {"constructs": <agent_repo_scanner 'constructs'>}  (each: construct, dpu_op, count)
    Output: per-vendor coverage + per-op matrix + a recommendation.
    """
    constructs = input_data.get("constructs", [])
    if not constructs:
        return {"error": "no constructs to compare"}

    total = sum(c["count"] for c in constructs)
    results = {}
    for name, cfg in VENDORS.items():
        ops, native, limited, unsupported = [], 0, 0, 0
        for c in constructs:
            verdict, note = _verdict(cfg, c["dpu_op"])
            if verdict == "native":
                native += c["count"]
            elif verdict == "limited":
                limited += c["count"]
            else:
                unsupported += c["count"]
            ops.append({"construct": c["construct"], "op": c["dpu_op"],
                        "count": c["count"], "verdict": verdict, "note": note})
        pct = lambda n: round(100 * n / total, 1) if total else 0.0
        results[name] = {
            "accel": cfg["accel"], "power": cfg["power"],
            "native_pct": pct(native), "runs_pct": pct(native + limited),
            "unsupported_pct": pct(unsupported), "port_effort_pct": pct(limited + unsupported),
            "ops": sorted(ops, key=lambda o: (-o["count"], o["op"])),
        }

    # recommendation: least porting effort = highest native %, with the edge/power note
    best = max(results.items(), key=lambda kv: kv[1]["native_pct"])
    amd = results.get("AMD Vitis DPU", {})
    recommendation = {
        "least_effort": best[0],
        "least_effort_native_pct": best[1]["native_pct"],
        "lowest_power": "AMD Vitis DPU",
        "summary": (
            f"Runs with the least change on **{best[0]}** ({best[1]['native_pct']}% native). "
            f"**AMD Vitis DPU** is the lowest-power edge option but needs the most refactoring "
            f"({amd.get('port_effort_pct', 0)}% of ops need changes or move to CPU)."
        ),
    }

    return {"status": "success", "total_construct_uses": total,
            "vendors": results, "recommendation": recommendation}


if __name__ == "__main__":
    import json
    from agents.agent_repo_scanner import scan_repo
    scan = scan_repo({"repo_path": "model_input/Transfer-Model_original"})
    out = compare_vendors({"constructs": scan["constructs"]})
    for name, v in out["vendors"].items():
        print(f"{name:18s} native {v['native_pct']:5}%  runs {v['runs_pct']:5}%  "
              f"effort {v['port_effort_pct']:5}%  ({v['power']})")
    print("\n", out["recommendation"]["summary"])
