# wire.py
import asyncio
import json
import os
import sys

# Ensure the repository root is on sys.path when running this script directly.
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from agents.agent_one_modelExtractor import extract_model
from agents.agent_three_proposal_a import generate_conservative_proposal
from agents.agent_two_RAG import search_vitis_compatibility
from agents.agent_three_proposal_b import generate_aggressive_proposal

async def main_pipeline(model_path: str) -> dict:
    """
    Full pipeline: Extract → RAG → Proposals
    """
    
    print(f"Starting analysis of {model_path}...")
    
    # Stage 1: Extract operations
    print("\n[Stage 1/4] Extracting model operations...")
    extraction = await extract_model({"model_path": model_path})
    
    if "error" in extraction:
        return {"error": extraction["error"]}
    
    print(f"  ✓ Found {extraction['total_ops']} operations")
    
    # Stage 2: Check compatibility with Vitis
    print("\n[Stage 2/4] Checking Vitis AI compatibility...")
    compatibility = await search_vitis_compatibility({
        "operations": extraction["operations"]
    })
    
    if "error" in compatibility:
        return {"error": compatibility["error"]}

    print(f"  ✓ Checked {compatibility['total_checked']} ops ({compatibility.get('unique_op_types','?')} unique types)")
    print(f"  ⚠ Found {compatibility['unsupported_count']} unsupported ops")
    if compatibility.get("critical_types"):
        print(f"  🔴 Critical: {', '.join(compatibility['critical_types'])}")
    if compatibility.get("warning_types"):
        print(f"  🟡 Warning:  {', '.join(compatibility['warning_types'])}")

    shared = {
        "model_name": extraction["model_name"],
        "operations": extraction["operations"],
        "unsupported_ops": compatibility["unsupported_ops"],
        "op_summary": compatibility.get("op_summary", {}),
        "critical_types": compatibility.get("critical_types", []),
        "warning_types": compatibility.get("warning_types", []),
    }

    # Stage 3: Generate conservative proposal
    print("\n[Stage 3/4] Generating conservative hardware audit...")
    proposal_a = await generate_conservative_proposal(shared)

    print(f"  ✓ Proposal A: {proposal_a.get('vitis_compatibility', 'N/A')} compatibility")

    # Stage 4: Generate aggressive proposal
    print("\n[Stage 4/4] Generating aggressive deployment plan...")
    proposal_b = await generate_aggressive_proposal(shared)
    
    print(f"  ✓ Proposal B: {proposal_b.get('vitis_compatibility', 'N/A')} compatibility")
    
    # Combine results
    final_result = {
        "model_name": extraction["model_name"],
        "extraction": extraction,
        "compatibility": compatibility,
        "proposal_a": proposal_a,
        "proposal_b": proposal_b,
        "timestamp": __import__("datetime").datetime.now().isoformat()
    }
    
    return final_result

# For local testing
if __name__ == "__main__":
    import os
    model_arg = sys.argv[1] if len(sys.argv) > 1 else "yolov8m.onnx"
    # Prefer the already-exported .onnx over the .pt to avoid reloading weights
    if model_arg.endswith(".pt"):
        onnx_candidate = os.path.splitext(model_arg)[0] + ".onnx"
        if os.path.exists(onnx_candidate):
            model_arg = onnx_candidate
    result = asyncio.run(main_pipeline(model_arg))
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(json.dumps(result, indent=2))
    
    # Save to file
    with open("analysis_result.json", "w") as f:
        json.dump(result, f, indent=2)
    print("\nResults saved to analysis_result.json")