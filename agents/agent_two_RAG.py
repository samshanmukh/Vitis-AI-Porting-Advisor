# agent_two_RAG.py
import json
from collections import Counter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

try:
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = Chroma(embedding_function=embeddings, persist_directory="./vitis_docs_db")
except Exception as e:
    vector_db = None
    print(f"Warning: Vitis docs DB not found. Run setup_vitis_rag.py first. ({e})")

# Known ops that are technically "supported" but cause DPU/CPU transitions or have caveats
_PARTIAL_SUPPORT = {
    "Sigmoid": "warning",   # supported but causes DPU exit when used mid-graph
    "Softmax": "critical",  # not DPU-native; forces CPU subgraph
    "Resize":  "warning",   # only nearest/bilinear with fixed sizes
    "Reshape": "warning",   # dynamic shapes break DPU compilation
    "Transpose": "warning",
    "Slice":   "warning",
    "Gather":  "critical",
    "Split":   "warning",
    "Div":     "warning",   # element-wise div is CPU-bound on most DPUs
    "Sub":     "warning",
    "Sqrt":    "critical",
    "Exp":     "critical",
    "Log":     "critical",
    "NonMaxSuppression": "critical",
    "TopK":    "critical",
}

# Ops that map cleanly to the DPU — force "ok" over a noisy RAG lookup.
_DPU_NATIVE = {
    "Conv", "BatchNormalization", "Relu", "LeakyRelu", "MaxPool", "AveragePool",
    "GlobalAveragePool", "Concat", "Add", "Mul", "Gemm", "HardSigmoid", "HardSwish",
}

async def search_vitis_compatibility(input_data: dict) -> dict:
    """
    Agent 2: Cross-reference ops against Vitis AI docs

    Input: {"operations": [...], "target_arch": "B4096"}
    Output: {"compatibility_report": [...], "unsupported_ops": [...], "op_summary": {...}}
    """

    operations   = input_data.get("operations", [])
    target_arch  = input_data.get("target_arch", "B4096")

    if not operations:
        return {"error": "operations list required"}

    if not vector_db:
        return {"error": "Vitis documentation not loaded"}

    # Count op frequencies for the summary
    type_counts = Counter(op.get("type") for op in operations)

    # Only hit the vector DB once per unique op type (not 100× for "Conv")
    unique_types = list(type_counts.keys())
    type_info: dict[str, dict] = {}

    for op_type in unique_types:
        # Query is architecture-specific
        query = f"Is {op_type} supported in Vitis AI on {target_arch} DPU?"
        try:
            # Fetch top-3 and prefer docs that match the target arch or "ALL"
            results = vector_db.similarity_search(query, k=3)
            best = None
            for r in results:
                arch = r.metadata.get("arch", "ALL")
                if arch == target_arch or arch == "ALL":
                    best = r
                    break
            doc = best or (results[0] if results else None)
            doc_content = doc.page_content if doc else "No documentation found"

            # Use metadata supported flag if available, else parse text
            if doc and "supported" in doc.metadata:
                is_supported = doc.metadata["supported"]
            else:
                is_supported = "supported" in doc_content.lower() and "not supported" not in doc_content.lower()

            # Override with known partial-support cases
            severity = _PARTIAL_SUPPORT.get(op_type, "ok" if is_supported else "critical")
            if not is_supported:
                severity = "critical"

            # Known DPU-native ops always map cleanly (wins over a noisy RAG lookup)
            if op_type in _DPU_NATIVE:
                is_supported, severity = True, "ok"

            type_info[op_type] = {
                "supported": is_supported,
                "severity": severity,
                "vitis_info": doc_content[:300],
                "count": type_counts[op_type],
                "optimization_hint": _optimization_hint(op_type, is_supported),
            }
        except Exception as e:
            type_info[op_type] = {
                "supported": True, "severity": "ok",
                "vitis_info": f"lookup error: {e}", "count": type_counts[op_type],
                "optimization_hint": "",
            }

    # Build per-op report from the deduplicated type_info
    compatibility_report = []
    unsupported_ops = []

    for op in operations:
        op_type = op.get("type")
        info = type_info.get(op_type, {})
        entry = {
            "op_name": op.get("name"),
            "op_type": op_type,
            "supported": info.get("supported", True),
            "severity": info.get("severity", "ok"),
            "vitis_info": info.get("vitis_info", ""),
            "optimization_hint": info.get("optimization_hint", ""),
        }
        compatibility_report.append(entry)
        if not info.get("supported", True):
            unsupported_ops.append({"name": op.get("name"), "type": op_type})

    # Aggregate summary
    op_summary = {
        op_type: {
            "count": d["count"],
            "supported": d["supported"],
            "severity": d["severity"],
            "optimization_hint": d["optimization_hint"],
        }
        for op_type, d in type_info.items()
    }

    critical_types = [t for t, d in type_info.items() if d["severity"] == "critical"]
    warning_types  = [t for t, d in type_info.items() if d["severity"] == "warning"]

    return {
        "compatibility_report": compatibility_report,
        "unsupported_ops": unsupported_ops,
        "op_summary": op_summary,
        "target_arch": target_arch,
        "total_checked": len(operations),
        "unique_op_types": len(unique_types),
        "unsupported_count": len(unsupported_ops),
        "critical_types": critical_types,
        "warning_types": warning_types,
        "status": "success",
    }


def _optimization_hint(op_type: str, supported: bool) -> str:
    hints = {
        "Conv":     "Fuse with BN+ReLU for single DPU kernel; use depthwise where possible.",
        "Sigmoid":  "Replace with HardSigmoid or move to CPU post-processing to avoid DPU exit.",
        "Softmax":  "Not DPU-native — move to CPU post-processing after last Conv output.",
        "Resize":   "Fix output size statically at export; use nearest-neighbor with integer scales.",
        "Reshape":  "Ensure all shapes are static at export time; dynamic Reshape breaks DPU compilation.",
        "Split":    "Refactor using explicit Conv branches; Split forces DPU subgraph boundary.",
        "Slice":    "Replace with stride-2 Conv; Slice is not DPU-mappable.",
        "Concat":   "Supported with static shapes; verify fan-in ≤ DPU arch limit.",
        "MaxPool":  "Verify kernel ≤ 8×8 for target DPU arch (B4096 supports up to 8×8).",
        "Mul":      "Element-wise Mul is DPU-supported in residual; SiLU (Sigmoid×x) needs replacement.",
        "Div":      "Element-wise Div is CPU-bound on most DPUs — consider fusing or eliminating.",
        "Gemm":     "Fully connected layers map well to DPU; ensure static batch size.",
        "NonMaxSuppression": "Post-processing only — run on ARM CPU after DPU inference.",
        "TopK":     "Run on CPU; not supported on DPU.",
        "Transpose":"Static Transpose at graph boundaries is fine; dynamic causes DPU exit.",
    }
    return hints.get(op_type, "No specific hint." if supported else "Replace or move to CPU post-processing.")


if __name__ == "__main__":
    import asyncio
    test_ops = [
        {"name": "Conv_0", "type": "Conv"},
        {"name": "Sig_0",  "type": "Sigmoid"},
        {"name": "Pool_0", "type": "AdaptiveAvgPool"},
    ]
    result = asyncio.run(search_vitis_compatibility({"operations": test_ops}))
    print(json.dumps(result, indent=2))
