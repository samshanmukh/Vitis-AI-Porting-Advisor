# agent_arch_explainer.py
# Phase 1 — the explaining "agent". Takes the local agent's captured findings and
# explains, in grounded terms, WHAT in the architecture is incompatible and WHY,
# citing the DPU rule and the exact source location. Uses the user-selected LLM
# provider (Anthropic / Grok / OpenAI) via agents.llm.
import json
from agents.llm import get_llm

SYSTEM_PROMPT = """\
You are a Vitis AI DPU deployment engineer reviewing a model's SOURCE architecture
(not a compiled graph). You are given: the model's nn.Module classes, and a list of
constructs the code uses that were flagged against the target DPU's documented limits.

Your job is to explain, precisely and without hand-waving, WHAT is not compatible and WHY.

Rules:
- Ground every "why" in the provided DPU rule text and the source evidence (file:line).
  Do NOT invent limits that aren't in the rule text.
- Distinguish "unsupported" (blocks the DPU subgraph / forces CPU) from "supported but
  costly" (causes DPU↔CPU transitions, quantisation risk, or constraint violations).
- Be specific to THIS architecture — reference the actual module/construct names.
- Return ONLY raw JSON. No markdown fences, no prose outside the JSON.
"""


async def explain_architecture(input_data: dict) -> dict:
    """
    Input:  {"capture": <agent_observer output>, "target_arch": "B4096"}
    Output: structured explanation of incompatibilities, grounded in DPU rules + source.
    """
    capture = input_data.get("capture", {})
    target_arch = input_data.get("target_arch", capture.get("target_arch", "B4096"))

    arch = capture.get("architecture", {})
    findings = capture.get("incompatible", [])
    if not findings:
        return {"status": "success", "verdict": "No incompatibilities detected.",
                "incompatibilities": [], "architecture_assessment": "All scanned constructs map to DPU-supported ops."}

    # compact evidence block for the prompt
    lines = []
    for f in findings:
        lines.append(
            f"- {f['construct']} (DPU op: {f['dpu_op']}, severity: {f['severity']}, uses: {f['count']})\n"
            f"    DPU rule: {f.get('dpu_rule','')[:280]}\n"
            f"    Constraint flags: {'; '.join(f.get('constraint_flags', [])) or 'none'}\n"
            f"    Source: {', '.join(f.get('evidence', [])[:6])}"
        )
    findings_block = "\n".join(lines)
    modules = ", ".join(arch.get("modules", [])[:30]) or "unknown"

    prompt = f"""
TARGET DPU: {target_arch}

MODEL ARCHITECTURE (nn.Module classes):
{modules}

FLAGGED CONSTRUCTS (from static source scan + DPU-doc RAG):
{findings_block}

TASK
Explain what is not compatible and why. Return ONLY this JSON:
{{
  "architecture_assessment": "2-3 sentence read on how DPU-friendly this architecture is overall",
  "estimated_dpu_coverage": "<rough % of the graph that stays on the DPU, with one-line reasoning>",
  "incompatibilities": [
    {{
      "construct": "...",
      "dpu_op": "...",
      "verdict": "unsupported | partial | costly",
      "what": "what the construct does here",
      "why": "why it fails / is costly on {target_arch}, citing the DPU rule",
      "where": "file:line references from the evidence",
      "impact": "blocks DPU subgraph | forces CPU transition | quantisation risk",
      "fix": "concrete refactor to regain DPU coverage"
    }}
  ],
  "top_priority": "the single highest-impact thing to fix first"
}}
"""

    client, model = get_llm()
    msg = client.chat.completions.create(
        model=model,
        max_tokens=4096,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    raw = msg.choices[0].message.content.strip()
    start, end = raw.find("{"), raw.rfind("}")
    text = raw[start:end + 1] if start != -1 and end > start else raw
    try:
        out = json.loads(text)
        out["status"] = "success"
        return out
    except json.JSONDecodeError:
        return {"error": "Failed to parse explanation", "raw_response": raw}


if __name__ == "__main__":
    import asyncio, os
    from agents.agent_repo_scanner import scan_repo
    from agents.agent_two_RAG import search_vitis_compatibility
    from agents.agent_observer import observe_and_capture

    # needs an LLM key in env to actually run the explanation
    scan = scan_repo({"repo_path": "model_input/Transfer-Model_original"})
    compat = asyncio.run(search_vitis_compatibility({"operations": scan["operations"], "target_arch": "B4096"}))
    cap = observe_and_capture({"scan": scan, "compatibility": compat, "target_arch": "B4096"})
    if not os.environ.get("LLM_API_KEY"):
        print("Set LLM_PROVIDER/LLM_MODEL/LLM_API_KEY to run the explainer.")
    else:
        print(json.dumps(asyncio.run(explain_architecture({"capture": cap, "target_arch": "B4096"})), indent=2))
