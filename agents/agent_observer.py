# agent_observer.py
# Phase 1 — the "local agent". It observes the first scan (repo architecture +
# RAG compatibility) and CAPTURES a structured, evidence-backed record of what is
# incompatible and why. Fully local & deterministic — no API/LLM. The capture is
# persisted so later phases (and the LLM explainer) can reuse it.
import json
import os
import re

CAPTURE_DIR = "scan_memory"

# deterministic constraint checks: when a construct's kwargs (from the AST scan)
# violate a common DPU rule, we can state the "why" without any model.
def _constraint_flags(construct: str, kwargs: dict) -> list[str]:
    flags = []
    dil = kwargs.get("dilation")
    if dil and not re.fullmatch(r"\(?\s*1\s*(,\s*1\s*)?\)?", str(dil)):
        flags.append(f"dilation={dil} → atrous/dilated conv is NOT supported on most DPU archs "
                     f"(B512–B4096); replace with regular conv or an ASPP-free alternative.")
    grp = kwargs.get("groups")
    if grp and grp not in ("1", "None"):
        flags.append(f"groups={grp} → grouped/depthwise conv; only supported from B800 onward "
                     f"(B512 has no depthwise).")
    mode = kwargs.get("mode")
    if mode and not re.search(r"nearest|bilinear", str(mode), re.I):
        flags.append(f"mode={mode} → DPU resize supports nearest/bilinear only; this mode falls to CPU.")
    ks = kwargs.get("kernel_size")
    if construct in ("MaxPool2d",) and ks and re.search(r"\b5\b|\b7\b|\b9\b", str(ks)):
        flags.append(f"kernel_size={ks} → large MaxPool (SPPF-style) unsupported below B1600; "
                     f"cascade 3×3 MaxPools instead.")
    return flags


def observe_and_capture(input_data: dict) -> dict:
    """
    Input:  {"scan": <agent_repo_scanner output>,
             "compatibility": <agent_two search_vitis_compatibility output>,
             "target_arch": "B4096"}
    Output: a captured record { architecture, findings, incompatible, captured_path, ... }
    """
    scan = input_data.get("scan", {})
    compat = input_data.get("compatibility", {})
    target_arch = input_data.get("target_arch", "B4096")

    if scan.get("error"):
        return {"error": f"scan failed: {scan['error']}"}

    # per-DPU-op compatibility info from the RAG (op_summary keyed by op type)
    op_summary = compat.get("op_summary", {})

    findings = []
    for c in scan.get("constructs", []):
        op = c["dpu_op"]
        info = op_summary.get(op, {})
        severity = info.get("severity", "ok")
        supported = info.get("supported", True)

        # collect deterministic constraint violations across this construct's uses
        constraint_flags, flagged_locs = [], []
        for loc in c.get("locations", []):
            fl = _constraint_flags(c["construct"], loc.get("kwargs", {}))
            if fl:
                constraint_flags.extend(fl)
                flagged_locs.append(f"{loc['file']}:{loc['line']}")

        # a construct is a "finding" if the op is non-ok OR a constraint is violated
        is_issue = severity in ("warning", "critical") or not supported or bool(constraint_flags)
        # constraint violations escalate severity
        if constraint_flags and severity == "ok":
            severity = "warning"

        findings.append({
            "construct": c["construct"],
            "dpu_op": op,
            "category": c["category"],
            "count": c["count"],
            "severity": severity,
            "supported": supported,
            "is_issue": is_issue,
            "dpu_rule": info.get("vitis_info", ""),          # the WHY, from DPU docs
            "optimization_hint": info.get("optimization_hint", ""),
            "constraint_flags": sorted(set(constraint_flags)),
            "evidence": [f"{l['file']}:{l['line']}" for l in c.get("locations", [])][:8],
            "flagged_locations": flagged_locs,
        })

    incompatible = sorted(
        [f for f in findings if f["is_issue"]],
        key=lambda f: (0 if f["severity"] == "critical" else 1, -f["count"]),
    )

    architecture = {
        "repo_path": scan.get("repo_path"),
        "modules": [m["name"] for m in scan.get("modules", [])],
        "module_files": scan.get("modules", []),
        "summary": scan.get("summary", {}),
    }

    capture = {
        "target_arch": target_arch,
        "architecture": architecture,
        "findings": findings,
        "incompatible": incompatible,
        "counts": {
            "constructs": len(findings),
            "issues": len(incompatible),
            "critical": sum(1 for f in incompatible if f["severity"] == "critical"),
            "warning": sum(1 for f in incompatible if f["severity"] == "warning"),
        },
        "status": "success",
    }

    # persist the capture (the "local agent" remembers the scan)
    try:
        os.makedirs(CAPTURE_DIR, exist_ok=True)
        repo_tag = re.sub(r"[^A-Za-z0-9]+", "_", str(architecture["repo_path"] or "repo")).strip("_")
        path = os.path.join(CAPTURE_DIR, f"{repo_tag}__{target_arch}.json")
        with open(path, "w") as fh:
            json.dump(capture, fh, indent=2)
        capture["captured_path"] = path
    except Exception as e:
        capture["captured_path"] = None
        capture["capture_error"] = str(e)

    return capture


if __name__ == "__main__":
    import asyncio
    from agents.agent_repo_scanner import scan_repo
    from agents.agent_two_RAG import search_vitis_compatibility

    scan = scan_repo({"repo_path": "model_input/Transfer-Model_original"})
    compat = asyncio.run(search_vitis_compatibility(
        {"operations": scan["operations"], "target_arch": "B4096"}))
    cap = observe_and_capture({"scan": scan, "compatibility": compat, "target_arch": "B4096"})
    print(json.dumps(cap["counts"], indent=2))
    print("captured ->", cap.get("captured_path"))
    print("\nIncompatible / risky constructs:")
    for f in cap["incompatible"]:
        print(f"  [{f['severity']:8s}] {f['construct']:16s} ({f['dpu_op']}) ×{f['count']}")
        for cf in f["constraint_flags"]:
            print(f"             ↳ {cf}")
