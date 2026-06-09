# agent_three_proposal_b.py
import json
import os
from collections import Counter
from openai import OpenAI

client = OpenAI(
    base_url="https://api.butterbase.ai/v1",
    api_key=os.environ.get("BUTTERBASE_API_KEY", "bb_sk_1712fc41088ebdca35ee087d985d32e8a07c0c69"),
)

SYSTEM_PROMPT = """\
You are a senior Vitis AI deployment engineer and quantization specialist.
You produce full hardware deployment plans — DPU subgraph maps, INT8 calibration
pipelines, latency budgets, and power estimates — for Xilinx embedded AI accelerators.

Rules:
- Think like a compiler: analyse op graph topology, not just individual ops.
- Be quantitative: give real FPS estimates, real power numbers, real memory footprints.
- Call out specific DPU architecture constraints (B1024 vs B4096 vs DPUCVDX8G).
- Identify DPU subgraph boundaries — which layer is the last DPU op and why.
- Return ONLY raw JSON. No markdown fences, no commentary outside the JSON.
"""

async def generate_aggressive_proposal(input_data: dict) -> dict:
    """
    Hardware deployment plan — aggressive INT8 quantization strategy.

    Input:  { model_name, operations, unsupported_ops, op_summary, critical_types, warning_types }
    Output: full deployment plan with subgraph map, quantization pipeline, perf estimates
    """

    model_name     = input_data.get("model_name", "unknown")
    operations     = input_data.get("operations", [])
    unsupported    = input_data.get("unsupported_ops", [])
    critical_types = input_data.get("critical_types", [])
    warning_types  = input_data.get("warning_types", [])
    target_arch    = input_data.get("target_arch", "B4096")

    type_counts  = Counter(op.get("type") for op in operations)
    total_ops    = len(operations)
    unique_types = len(type_counts)

    op_breakdown = "\n".join(
        f"  {t:30s} ×{c:4d}  ({c/total_ops*100:.1f}%)"
        for t, c in type_counts.most_common()
    )

    critical_str    = ", ".join(critical_types) if critical_types else "none"
    warning_str     = ", ".join(warning_types)  if warning_types  else "none"
    unsupported_str = "\n".join(
        f"  - {op['name']} ({op['type']})" for op in unsupported
    ) if unsupported else "  none detected by RAG scan"

    prompt = f"""
HARDWARE DEPLOYMENT PLAN REQUEST — AGGRESSIVE INT8 MODE
=========================================================
Target platform  : Vitis AI 3.x / Xilinx DPU — {target_arch}
Fallback targets : next lower DPUCZDX8G tier or DPUCVDX8G (Versal)
Model            : {model_name}
Total ONNX ops   : {total_ops}  ({unique_types} unique types)

OP FREQUENCY BREAKDOWN
----------------------
{op_breakdown}

STATIC SCAN RESULTS
-------------------
Critical (block DPU or force CPU): {critical_str}
Warning  (DPU↔CPU transition risk): {warning_str}
Unsupported ops:
{unsupported_str}

TASK
----
Produce an AGGRESSIVE full-deployment hardware plan:
1. DPU subgraph partition: identify the exact DPU subgraph boundary (first and last DPU op).
2. CPU post-processing ops: list every op that must run on ARM CPU and why.
3. INT8 quantization pipeline: calibration method, dataset size, expected SQNR.
4. Architecture changes: everything needed to maximise DPU coverage — no compromise.
5. Performance envelope: FPS at 1080p, latency (ms), power draw (W), memory footprint (MB).
6. Deployment pipeline: exact sequence of vai_q_pytorch → vai_c_torch → xmodel steps.

Return ONLY this JSON (no markdown):
{{
  "target_dpu_arch": "DPUCZDX8G B4096 | DPUCVDX8G | ...",
  "dpu_subgraph": {{
    "first_dpu_op": "...",
    "last_dpu_op":  "...",
    "dpu_op_count": 0,
    "cpu_op_count": 0,
    "dpu_coverage": "...%"
  }},
  "cpu_residual": [
    {{"op_type": "...", "reason": "...", "location": "pre-DPU | post-DPU | mid-graph"}}
  ],
  "architecture_changes": [
    {{
      "op_name":        "...",
      "current":        "...",
      "replacement":    "...",
      "affected_layers":["..."],
      "dpu_gain":       "...",
      "risk":           "low | medium | high",
      "effort":         "..."
    }}
  ],
  "quantization_plan": {{
    "method":           "PTQ | QAT",
    "calibration_images": 0,
    "expected_sqnr_db": "...",
    "sensitive_layers": ["..."],
    "mixed_precision":  true,
    "calibration_cmd":  "vai_q_pytorch ..."
  }},
  "performance_estimates": {{
    "fps_1080p":        "...",
    "latency_ms":       "...",
    "power_draw_w":     "...",
    "memory_mb":        "...",
    "vs_cpu_baseline":  "...x speedup"
  }},
  "deployment_pipeline": ["step 1", "step 2", "..."],
  "strategy":               "...",
  "estimated_accuracy_drop":"...",
  "vitis_compatibility":    "...%",
  "inference_speedup":      "...",
  "dev_effort":             "..."
}}
"""

    message = client.chat.completions.create(
        model="anthropic/claude-sonnet-4.6",
        max_tokens=4096,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
    )

    raw = message.choices[0].message.content.strip()
    # Extract outermost JSON object, tolerating any preamble/fences
    start = raw.find("{")
    end   = raw.rfind("}")
    text  = raw[start:end+1] if start != -1 and end > start else raw

    try:
        proposal = json.loads(text)
        proposal["status"] = "success"
        return proposal
    except json.JSONDecodeError:
        return {"error": "Failed to parse response", "raw_response": raw}


if __name__ == "__main__":
    import asyncio
    result = asyncio.run(generate_aggressive_proposal({
        "model_name": "yolov8m.onnx",
        "operations": [{"type": "Conv"}, {"type": "Sigmoid"}, {"type": "Mul"}],
        "unsupported_ops": [],
        "op_summary": {},
        "critical_types": ["Softmax"],
        "warning_types": ["Sigmoid", "Resize"],
    }))
    print(json.dumps(result, indent=2))
