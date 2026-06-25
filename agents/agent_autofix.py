# agent_autofix.py
# Analyzer → OUTCOME. Applies the safe, mechanical DPU refactors to a COPY of the
# model repo, then VERIFIES by re-scanning the patched copy and measuring the real
# DPU-coverage delta. Produces a reviewable patch + a verified record.
#
# Conservative by design:
#   • only swaps DPU-native activation modules (nn.Sigmoid/SiLU/GELU)
#   • never edits loss / training / eval / util code (not the deployed model)
#   • everything it can't safely auto-apply is reported as "manual"
import datetime
import difflib
import json
import os
import re
import shutil
import tempfile

VERIFIED_DIR = "verified"

# (regex, replacement, human description) — only clearly DPU-safe activation swaps
AUTO_RULES = [
    (r"\bnn\.SiLU\b",    "nn.Hardswish",   "nn.SiLU → nn.Hardswish (DPU-native swish approximation)"),
    (r"\bnn\.Sigmoid\b", "nn.Hardsigmoid", "nn.Sigmoid → nn.Hardsigmoid (DPU-native)"),
    (r"\bnn\.GELU\b",    "nn.Hardswish",   "nn.GELU → nn.Hardswish (DPU-friendly approximation)"),
]
# never auto-edit non-model code (losses, metrics, training, eval, options, data)
_EXCLUDE = re.compile(r"(loss|metric|train|eval|option|dataset|dataloader|transform|parser|/utils/|/options/)", re.I)


def apply_autofix(repo_path: str) -> dict:
    """Copy the repo, apply AUTO_RULES to model files, return patched path + edits + unified diff."""
    work = tempfile.mkdtemp(prefix="dpu_fix_")
    patched_root = os.path.join(work, "repo")
    shutil.copytree(repo_path, patched_root,
                    ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc", ".DS_Store"))

    edits, diffs, skipped = [], [], []
    for root, _dirs, files in os.walk(patched_root):
        for fn in files:
            if not fn.endswith(".py"):
                continue
            path = os.path.join(root, fn)
            rel = os.path.relpath(path, patched_root)
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                orig = f.read()

            if _EXCLUDE.search("/" + rel):
                if any(re.search(p, orig) for p, _r, _d in AUTO_RULES):
                    skipped.append(rel)   # had a candidate but we won't touch non-model code
                continue

            new, file_changes = orig, []
            for pat, repl, desc in AUTO_RULES:
                cnt = len(re.findall(pat, new))
                if cnt:
                    new = re.sub(pat, repl, new)
                    file_changes.append({"rule": desc, "count": cnt})
            if new != orig:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(new)
                edits.append({"file": rel, "changes": file_changes})
                diffs.append("".join(difflib.unified_diff(
                    orig.splitlines(True), new.splitlines(True),
                    fromfile=f"a/{rel}", tofile=f"b/{rel}")))

    return {
        "patched_path": patched_root,
        "edits": edits,
        "patch": "\n".join(diffs),
        "files_changed": len(edits),
        "total_edits": sum(c["count"] for e in edits for c in e["changes"]),
        "skipped_non_model": sorted(set(skipped)),
    }


async def verify_fix(original_path: str, patched_path: str, target_arch: str) -> dict:
    """Re-scan original vs patched and MEASURE the DPU-coverage delta (not estimated)."""
    from agents.agent_repo_scanner import scan_repo
    from agents.agent_two_RAG import search_vitis_compatibility
    from agents.agent_observer import observe_and_capture
    from agents.agent_converter import transform_structure

    async def _coverage(path):
        scan = scan_repo({"repo_path": path})
        compat = await search_vitis_compatibility({"operations": scan["operations"], "target_arch": target_arch})
        cap = observe_and_capture({"scan": scan, "compatibility": compat, "target_arch": target_arch})
        tr = transform_structure({"capture": cap})
        return cap, tr

    cap_before, tr_before = await _coverage(original_path)
    cap_after, tr_after = await _coverage(patched_path)

    cov_b = tr_before["coverage"]["coverage_before_pct"]   # raw "as authored" coverage
    cov_a = tr_after["coverage"]["coverage_before_pct"]     # raw coverage AFTER the patch (measured)

    # what still needs manual work after the auto-fix
    remaining = [
        {"construct": f["construct"], "dpu_op": f["dpu_op"], "severity": f["severity"], "count": f["count"]}
        for f in cap_after.get("incompatible", [])
    ]
    return {
        "measured_before_pct": cov_b,
        "measured_after_pct": cov_a,
        "delta_pct": round(cov_a - cov_b, 1),
        "issues_before": cap_before["counts"]["issues"],
        "issues_after": cap_after["counts"]["issues"],
        "remaining_manual": remaining,
        "capture_after": cap_after,
    }


def save_verified(autofix: dict, verify: dict, repo_name: str, target_arch: str) -> dict:
    os.makedirs(VERIFIED_DIR, exist_ok=True)
    tag = re.sub(r"[^A-Za-z0-9]+", "_", str(repo_name or "repo")).strip("_")
    base = os.path.join(VERIFIED_DIR, f"{tag}__{target_arch}__verified")
    record = {
        "repo": repo_name, "target_arch": target_arch,
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "status": "verified",
        "auto_fixes": {"files_changed": autofix["files_changed"], "total_edits": autofix["total_edits"],
                       "edits": autofix["edits"], "skipped_non_model": autofix["skipped_non_model"]},
        "verification": {"measured_before_pct": verify["measured_before_pct"],
                         "measured_after_pct": verify["measured_after_pct"],
                         "delta_pct": verify["delta_pct"],
                         "issues_before": verify["issues_before"], "issues_after": verify["issues_after"],
                         "remaining_manual": verify["remaining_manual"]},
    }
    try:
        with open(base + ".json", "w") as f:
            json.dump(record, f, indent=2)
        with open(base + ".patch", "w") as f:
            f.write(autofix["patch"] or "# no auto-applicable changes\n")
    except Exception as e:
        return {"error": str(e)}
    return {"json_path": base + ".json", "patch_path": base + ".patch",
            "json_text": json.dumps(record, indent=2), "record": record}


if __name__ == "__main__":
    import asyncio
    src = "model_input/Transfer-Model_original"
    fix = apply_autofix(src)
    print(f"auto-fixes: {fix['total_edits']} edits across {fix['files_changed']} files")
    for e in fix["edits"]:
        print("  ", e["file"], e["changes"])
    print("  skipped (non-model):", fix["skipped_non_model"])
    v = asyncio.run(verify_fix(src, fix["patched_path"], "B4096"))
    print(f"VERIFIED coverage: {v['measured_before_pct']}% → {v['measured_after_pct']}% (Δ {v['delta_pct']}%)")
    print(f"issues: {v['issues_before']} → {v['issues_after']}; remaining manual: {len(v['remaining_manual'])}")
