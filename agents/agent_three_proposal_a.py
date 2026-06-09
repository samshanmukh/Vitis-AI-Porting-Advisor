# agent_three_proposal_a.py
import json
import os
from collections import Counter
from openai import OpenAI

client = OpenAI(
    base_url="https://api.butterbase.ai/v1",
    api_key=os.environ.get("BUTTERBASE_API_KEY", "bb_sk_1712fc41088ebdca35ee087d985d32e8a07c0c69"),
)

SYSTEM_PROMPT = """\
You are a Vitis AI hardware compatibility scanner and DPU deployment engineer.
Your role is to perform a deep static analysis of an ONNX model's operator graph
and produce a structured hardware audit report — like a compiler diagnostics pass,
not a general suggestion document.

Rules:
- Analyse ALL op types, not just unsupported ones. Every op has a DPU utilisation story.
- Flag ops that cause CPU↔DPU transitions even if technically "supported".
- Quantify impact: how many ops of each type, what fraction of the graph.
- Be specific about which Xilinx DPU architectures are affected (DPUCZDX8G B4096, DPUCVDX8G, etc.).
- Return ONLY raw JSON. No markdown fences, no commentary outside the JSON.
"""

async def generate_conservative_proposal(input_data: dict) -> dict:
    """
    Hardware audit — conservative porting strategy.

    Input:  { model_name, operations, unsupported_ops, op_summary, critical_types, warning_types }
    Output: structured hardware audit report
    """

    model_name     = input_data.get("model_name", "unknown")
    operations     = input_data.get("operations", [])
    unsupported    = input_data.get("unsupported_ops", [])
    op_summary     = input_data.get("op_summary", {})
    critical_types = input_data.get("critical_types", [])
    warning_types  = input_data.get("warning_types", [])
    target_arch    = input_data.get("target_arch", "B4096")

    type_counts = Counter(op.get("type") for op in operations)
    total_ops   = len(operations)

    op_breakdown = "\n".join(
        f"  {t:30s} ×{c:4d}  ({c/total_ops*100:.1f}%)"
        for t, c in type_counts.most_common()
    )

    critical_str = ", ".join(critical_types) if critical_types else "none"
    warning_str  = ", ".join(warning_types)  if warning_types  else "none"
    unsupported_str = "\n".join(
        f"  - {op['name']} ({op['type']})" for op in unsupported
    ) if unsupported else "  none detected"

    hint_lines = "\n".join(
        f"  {t}: {d.get('optimization_hint','')}"
        for t, d in op_summary.items()
        if d.get("optimization_hint") and d.get("severity") in ("warning", "critical")
    )

    prompt = f"""
HARDWARE SCAN REQUEST — CONSERVATIVE PORTING MODE
===================================================
Target platform : Vitis AI 3.x / Xilinx DPU — {target_arch}
Model           : {model_name}
Total ops       : {total_ops}

OP FREQUENCY BREAKDOWN
----------------------
{op_breakdown}

SCAN FINDINGS
-------------
Critical ops (block DPU compilation or force full CPU fallback): {critical_str}
Warning  ops (supported but cause DPU↔CPU transitions or have constraints): {warning_str}
Fully unsupported ops:
{unsupported_str}

OPTIMIZATION HINTS FROM RAG SCAN
---------------------------------
{hint_lines if hint_lines else "  (all ops appear DPU-compatible at surface level)"}

TASK
----
Produce a CONSERVATIVE hardware audit report:
- Target audience: an embedded ML engineer who will hand-edit PyTorch source.
- Conservative = minimal code changes; accept residual CPU subgraph for complex ops.
- Focus on the highest-impact, lowest-risk changes that maximise DPU subgraph coverage.
- For each recommended change, specify WHICH layer names are affected (use the op breakdown).
- Estimate DPU subgraph coverage before and after changes.
- Flag any ops that will remain on CPU even after conservative fixes.

Return ONLY this JSON (no markdown):
{{
  "scan_summary": {{
    "total_ops": {total_ops},
    "dpu_eligible_before": "<count or %>",
    "dpu_eligible_after":  "<count or %> after conservative fixes",
    "cpu_residual_ops":    ["<op_type>", ...],
    "risk_level": "low | medium | high"
  }},
  "critical_findings": [
    {{
      "op_type": "...",
      "instance_count": 0,
      "impact": "...",
      "dpu_effect": "blocks subgraph | forces CPU transition | quantisation risk"
    }}
  ],
  "recommended_changes": [
    {{
      "op_name": "...",
      "current": "...",
      "replacement": "...",
      "affected_layers": ["..."],
      "dpu_gain": "...",
      "risk": "low | medium",
      "effort": "..."
    }}
  ],
  "strategy": "one-sentence overall approach",
  "estimated_accuracy_drop": "...",
  "vitis_compatibility": "...%",
  "dev_effort": "...",
  "dpu_arch_notes": "specific notes for {target_arch} — kernel limits, channel width, stride constraints"
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
    result = asyncio.run(generate_conservative_proposal({
        "model_name": "yolov8m.onnx",
        "operations": [{"type": "Conv"}, {"type": "Sigmoid"}, {"type": "Mul"}],
        "unsupported_ops": [],
        "op_summary": {},
        "critical_types": ["Softmax"],
        "warning_types": ["Sigmoid", "Resize"],
    }))
    print(json.dumps(result, indent=2))
