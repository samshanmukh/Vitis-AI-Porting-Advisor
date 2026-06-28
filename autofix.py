"""Auto-Fix & Verify page — apply safe DPU refactors, then prove the coverage gain by re-scanning."""
import asyncio
import os
import sys

import streamlit as st

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from agents.repo_fetch import resolve_repo
from agents.repo_input import repo_source_input
from agents.agent_repo_scanner import scan_repo
from agents.agent_two_RAG import search_vitis_compatibility
from agents.agent_observer import observe_and_capture
from agents.agent_converter import transform_structure
from agents.agent_autofix import apply_autofix, verify_fix, save_verified

DPU_ARCHS = ["B512", "B800", "B1024", "B1152", "B1600", "B2304", "B3136", "B4096",
             "DPUCVDX8G", "DPUCAHX8H", "DPUCADF8H"]


def run_async(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


st.markdown("""
<div style="display:flex;align-items:center;gap:0.8rem;margin-bottom:0.25rem;">
  <div style="width:38px;height:38px;background:linear-gradient(135deg,#16a34a,#15803d);
              border-radius:10px;display:flex;align-items:center;justify-content:center;
              box-shadow:0 4px 14px rgba(22,163,74,0.45);"><span style="font-size:1.1rem;">✅</span></div>
  <h1 style="margin:0;line-height:1.1;">Auto-Fix &amp; Verify</h1>
</div>
<p style="margin:0 0 1rem 0;color:rgba(100,116,139,0.9);font-size:0.85rem;padding-left:46px;">
  Apply the safe DPU refactors to a copy &rarr; <strong>re-scan</strong> &rarr; prove the coverage gain.
  Estimates become <strong>measured outcomes</strong>.
</p>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("**Auto-Fix & Verify**")
    src = repo_source_input()
    target_arch = st.selectbox("Target DPU arch", DPU_ARCHS, index=DPU_ARCHS.index("B4096"))
    run = st.button("▶  Auto-fix & verify", type="primary", use_container_width=True)

if not run:
    st.info("Runs the safe activation refactors on a **copy**, then re-scans to measure the real coverage gain. "
            "Your original source is never modified.", icon="✅")
    st.stop()

# ── resolve repo ────────────────────────────────────────────────────────────
with st.spinner("Resolving repo…"):
    resolved = resolve_repo(src)
if resolved.get("error"):
    st.error(f"Could not load repo: {resolved['error']}")
    st.stop()

# ── status pipeline ─────────────────────────────────────────────────────────
st.markdown("""
<div style="display:flex;gap:0.4rem;align-items:center;margin:0.25rem 0 0.75rem;font-size:0.8rem;">
  <span style="padding:0.25rem 0.6rem;border-radius:999px;background:rgba(37,99,235,0.18);color:#93c5fd;">1 · Analyzed</span>
  <span style="color:#64748b;">→</span>
  <span style="padding:0.25rem 0.6rem;border-radius:999px;background:rgba(13,148,136,0.18);color:#5eead4;">2 · Auto-fixed</span>
  <span style="color:#64748b;">→</span>
  <span style="padding:0.25rem 0.6rem;border-radius:999px;background:rgba(22,163,74,0.20);color:#86efac;">3 · Verified</span>
</div>
""", unsafe_allow_html=True)

# ── apply + verify ──────────────────────────────────────────────────────────
with st.spinner("Applying safe refactors to a copy & re-scanning…"):
    fix = apply_autofix(resolved["path"])
    verify = run_async(verify_fix(resolved["path"], fix["patched_path"], target_arch))
    # "estimated if you do everything" for contrast
    scan0 = scan_repo({"repo_path": resolved["path"]})
    compat0 = run_async(search_vitis_compatibility({"operations": scan0["operations"], "target_arch": target_arch}))
    cap0 = observe_and_capture({"scan": scan0, "compatibility": compat0, "target_arch": target_arch})
    est = transform_structure({"capture": cap0})["coverage"]
    saved = save_verified(fix, verify, resolved["name"], target_arch)

# ── measured outcome ────────────────────────────────────────────────────────
st.markdown("### Measured outcome (re-scanned, not estimated)")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Coverage before", f"{verify['measured_before_pct']}%")
m2.metric("Coverage after ✅", f"{verify['measured_after_pct']}%", delta=f"+{verify['delta_pct']}%")
m3.metric("Issues fixed", verify["issues_before"] - verify["issues_after"])
m4.metric("Auto-edits applied", fix["total_edits"])
st.progress(min(verify["measured_after_pct"] / 100, 1.0),
            text=f"Verified DPU coverage after auto-fix: {verify['measured_after_pct']}%")
st.caption(f"For contrast — *estimated* coverage if **all** proposed changes (incl. manual ones) were applied: "
           f"**{est['coverage_after_pct']}%**. The number above is what auto-fix achieved and we re-measured.")

# ── what was applied ────────────────────────────────────────────────────────
st.markdown(f"### What was auto-applied · {fix['files_changed']} files, {fix['total_edits']} edits")
if not fix["edits"]:
    st.info("No safe activation swaps were applicable in this repo.")
for e in fix["edits"]:
    rules = "; ".join(f"{c['rule']} ×{c['count']}" for c in e["changes"])
    st.markdown(f"- `{e['file']}` — {rules}")
if fix["skipped_non_model"]:
    st.caption("Skipped (loss/training/util code — not the deployed model): " + ", ".join(fix["skipped_non_model"]))

# ── the patch ───────────────────────────────────────────────────────────────
st.markdown("### Patch (review before applying to your repo)")
if fix["patch"].strip():
    st.code(fix["patch"], language="diff")
else:
    st.caption("No diff produced.")

# ── what still needs manual work ────────────────────────────────────────────
st.markdown("### Still needs manual work")
if not verify["remaining_manual"]:
    st.success("Nothing left — fully DPU-mappable after the auto-fix.")
else:
    st.dataframe(
        [{"construct": r["construct"], "DPU op": r["dpu_op"], "severity": r["severity"], "uses": r["count"]}
         for r in verify["remaining_manual"]],
        use_container_width=True, hide_index=True)
    st.caption("These can't be auto-applied safely (e.g. Softmax→CPU, dilated conv, dynamic reshape, loss-fn ops).")

# ── downloads / persistence ─────────────────────────────────────────────────
if not saved.get("error"):
    st.caption(f"🗂️ Verified record saved → `{saved['json_path']}` · patch → `{saved['patch_path']}`")
    d1, d2 = st.columns(2)
    d1.download_button("⬇ Verified report (JSON)", saved["json_text"],
                       file_name=os.path.basename(saved["json_path"]), mime="application/json",
                       use_container_width=True)
    d2.download_button("⬇ Patch (.patch)", fix["patch"] or "# no changes\n",
                       file_name=os.path.basename(saved["patch_path"]), mime="text/x-patch",
                       use_container_width=True)
