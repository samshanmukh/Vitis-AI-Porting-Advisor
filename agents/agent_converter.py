# agent_converter.py
# Structure converter: given the captured scan, compute the before -> after
# structural transformation that makes the model DPU-portable, and the DPU
# coverage that change buys. Deterministic core (no key); optional LLM narrative.
import datetime
import json
import os
import re
from agents.llm import get_llm

CONVERTED_DIR = "restructured"   # where the restructured output is persisted

# dpu_op -> (replacement, structural effect, where it lands afterwards)
#   target "dpu"  → construct becomes DPU-eligible after the change
#   target "cpu"  → intentionally moved off-DPU (e.g. final Softmax / NMS)
REPLACEMENTS = {
    "Sigmoid":     ("HardSigmoid", "swap activation — DPU-native, keeps the subgraph intact", "dpu"),
    "SiLU":        ("ReLU / HardSwish", "replace Swish (Sigmoid×x) which forces a CPU exit", "dpu"),
    "Softmax":     ("CPU post-process", "move the final Softmax off-DPU, after the last Conv", "cpu"),
    "Exp":         ("CPU post-process", "exp is not DPU-native — fold into the CPU head", "cpu"),
    "Log":         ("CPU post-process", "log is not DPU-native — fold into the CPU head", "cpu"),
    "Gelu":        ("ReLU approximation", "GELU→ReLU keeps the activation on the DPU", "dpu"),
    "Tanh":        ("HardTanh", "approximate tanh with a DPU-native clamp", "dpu"),
    "Resize":      ("nearest/bilinear @ static size", "fix interpolation mode and size at export", "dpu"),
    "Reshape":     ("static reshape", "freeze shapes at export — dynamic reshape breaks compilation", "dpu"),
    "Transpose":   ("static transpose at boundary", "keep transpose static so it doesn't trigger a DPU exit", "dpu"),
    "Split":       ("explicit Conv branches", "avoid the Split subgraph boundary", "dpu"),
    "Div":         ("fuse into BN / scale", "element-wise Div is CPU-bound — fold into a scale", "dpu"),
    "ReduceSum":   ("fuse / move to CPU head", "reductions often break the DPU subgraph", "cpu"),
    "ReduceMean":  ("GlobalAveragePool", "express mean as global avg-pool, which the DPU supports", "dpu"),
    "Einsum":      ("Conv / MatMul rewrite", "einsum is not DPU-native — rewrite as conv/matmul", "dpu"),
}
# constraint-flag driven replacement (dilated/grouped conv etc.)
_DILATION_FIX = ("regular conv (remove dilation) / cascaded conv",
                 "atrous/dilated conv is unsupported on most archs — replace with regular or cascaded conv", "dpu")


def transform_structure(input_data: dict) -> dict:
    """
    Input:  {"capture": <agent_observer output>}
    Output: before/after structural changes + DPU-coverage delta (deterministic).
    """
    capture = input_data.get("capture", {})
    findings = capture.get("findings", [])
    if not findings:
        return {"error": "no findings to transform"}

    total_uses = sum(f["count"] for f in findings)
    dpu_before = sum(f["count"] for f in findings if not f["is_issue"])

    changes, dpu_after_gain, cpu_moved = [], 0, 0
    converted_structure = []          # the FULL post-conversion structure (every construct)
    for f in findings:
        is_issue = f["is_issue"]
        if not is_issue:
            after, effect, target = f["construct"], "already DPU-compatible — no change", "dpu"
        else:
            # pick a replacement: constraint-driven first, else op-table, else generic
            if f.get("constraint_flags") and any("dilat" in c.lower() for c in f["constraint_flags"]):
                after, effect, target = _DILATION_FIX
            elif f["dpu_op"] in REPLACEMENTS:
                after, effect, target = REPLACEMENTS[f["dpu_op"]]
            else:
                after, effect, target = ("CPU subgraph", "no DPU-native equivalent — runs on CPU", "cpu")

            if target == "dpu":
                dpu_after_gain += f["count"]
            else:
                cpu_moved += f["count"]

            changes.append({
                "construct": f["construct"],
                "dpu_op": f["dpu_op"],
                "severity": f["severity"],
                "count": f["count"],
                "before": f["construct"],
                "after": after,
                "effect": effect,
                "lands": target,                 # dpu | cpu
                "why": (f.get("dpu_rule") or "")[:240],
                "evidence": f.get("evidence", [])[:6],
            })

        converted_structure.append({
            "construct": f["construct"],
            "dpu_op": f["dpu_op"],
            "count": f["count"],
            "original": f["construct"],
            "converted": after,
            "changed": is_issue,
            "lands": target,                     # dpu | cpu
            "evidence": f.get("evidence", [])[:6],
        })

    dpu_after = dpu_before + dpu_after_gain
    pct = lambda n: round(100 * n / total_uses, 1) if total_uses else 0.0
    # critical/warning first, biggest first
    changes.sort(key=lambda c: (0 if c["severity"] == "critical" else 1, -c["count"]))
    converted_structure.sort(key=lambda c: (0 if c["changed"] else 1, -c["count"]))

    return {
        "status": "success",
        "coverage": {
            "total_construct_uses": total_uses,
            "dpu_before": dpu_before,
            "dpu_after": dpu_after,
            "cpu_after": total_uses - dpu_after,
            "coverage_before_pct": pct(dpu_before),
            "coverage_after_pct": pct(dpu_after),
            "delta_pct": round(pct(dpu_after) - pct(dpu_before), 1),
            "moved_to_cpu": cpu_moved,
        },
        "changes": changes,
        "converted_structure": converted_structure,
    }


def save_conversion(transform: dict, repo_name: str, target_arch: str, architecture: dict | None = None) -> dict:
    """Persist the restructured output (JSON + Markdown) so it can be re-scanned / shown in the UI."""
    os.makedirs(CONVERTED_DIR, exist_ok=True)
    tag = re.sub(r"[^A-Za-z0-9]+", "_", str(repo_name or "repo")).strip("_")
    base = f"{tag}__{target_arch}"
    artifact = {
        "repo": repo_name,
        "target_arch": target_arch,
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "architecture": architecture or {},
        "coverage": transform.get("coverage", {}),
        "changes": transform.get("changes", []),
        "converted_structure": transform.get("converted_structure", []),
    }
    json_path = os.path.join(CONVERTED_DIR, base + ".json")
    md = _to_markdown(artifact)
    md_path = os.path.join(CONVERTED_DIR, base + ".md")
    try:
        with open(json_path, "w") as f:
            json.dump(artifact, f, indent=2)
        with open(md_path, "w") as f:
            f.write(md)
    except Exception as e:
        return {"error": str(e)}
    return {"json_path": json_path, "md_path": md_path, "markdown": md,
            "json_text": json.dumps(artifact, indent=2), "artifact": artifact}


def _to_markdown(a: dict) -> str:
    cov = a.get("coverage", {})
    lines = [
        f"# Restructured model — DPU port plan ({a.get('target_arch')})",
        f"_Repo: {a.get('repo')} · generated {a.get('generated_at')}_",
        "",
        f"**DPU coverage:** {cov.get('coverage_before_pct')}% → "
        f"**{cov.get('coverage_after_pct')}%** (Δ {cov.get('delta_pct')}%), "
        f"{cov.get('moved_to_cpu')} construct-uses moved to CPU.",
        "",
        "## Changes",
        "| original | → converted | lands | uses |",
        "|---|---|---|---|",
    ]
    for c in a.get("changes", []):
        lines.append(f"| `{c['before']}` | `{c['after']}` | {c['lands'].upper()} | {c['count']} |")
    lines += ["", "## Full converted structure", "| construct | converted to | changed | lands | uses |",
              "|---|---|---|---|---|"]
    for c in a.get("converted_structure", []):
        lines.append(f"| `{c['construct']}` | `{c['converted']}` | "
                     f"{'yes' if c['changed'] else '—'} | {c['lands'].upper()} | {c['count']} |")
    return "\n".join(lines) + "\n"


async def narrate_conversion(input_data: dict) -> dict:
    """Optional LLM pass: describe the converted architecture in grounded prose."""
    capture = input_data.get("capture", {})
    transform = input_data.get("transform", {})
    target_arch = input_data.get("target_arch", capture.get("target_arch", "B4096"))
    modules = ", ".join(capture.get("architecture", {}).get("modules", [])[:25]) or "unknown"

    change_lines = "\n".join(
        f"- {c['before']} → {c['after']}  ({c['lands'].upper()}; ×{c['count']}; {c['effect']})"
        for c in transform.get("changes", [])
    )
    cov = transform.get("coverage", {})

    prompt = f"""
You are a Vitis AI deployment engineer. A model's source architecture was scanned and
a deterministic transform proposed the structural changes below to maximise DPU coverage
on {target_arch}. Summarise the CONVERTED structure — do not invent new changes.

ARCHITECTURE: {modules}
DPU coverage: {cov.get('coverage_before_pct')}% → {cov.get('coverage_after_pct')}% (Δ {cov.get('delta_pct')}%)
PROPOSED CHANGES:
{change_lines}

Return ONLY JSON:
{{
  "converted_summary": "3-4 sentences describing what the model looks like after these changes",
  "structural_changes": ["short imperative bullet per change, grouped sensibly"],
  "remaining_cpu": "what intentionally stays on CPU and why that's fine",
  "risk": "accuracy/effort risk of this conversion in one line"
}}
"""
    client, model = get_llm()
    msg = client.chat.completions.create(
        model=model, max_tokens=2048,
        messages=[{"role": "system", "content": "Return only raw JSON, no fences."},
                  {"role": "user", "content": prompt}],
    )
    raw = msg.choices[0].message.content.strip()
    a, b = raw.find("{"), raw.rfind("}")
    try:
        out = json.loads(raw[a:b + 1] if a != -1 and b > a else raw)
        out["status"] = "success"
        return out
    except json.JSONDecodeError:
        return {"error": "failed to parse", "raw_response": raw}


if __name__ == "__main__":
    import asyncio
    from agents.agent_repo_scanner import scan_repo
    from agents.agent_two_RAG import search_vitis_compatibility
    from agents.agent_observer import observe_and_capture

    scan = scan_repo({"repo_path": "model_input/Transfer-Model_original"})
    compat = asyncio.run(search_vitis_compatibility({"operations": scan["operations"], "target_arch": "B4096"}))
    cap = observe_and_capture({"scan": scan, "compatibility": compat, "target_arch": "B4096"})
    t = transform_structure({"capture": cap})
    print(json.dumps(t["coverage"], indent=2))
    for c in t["changes"]:
        print(f"  {c['before']:14s} → {c['after']:36s} [{c['lands']}] ×{c['count']}")
