"""Multi-Vendor page — score the same model against AMD DPU, NVIDIA TensorRT, Intel OpenVINO."""
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
from agents.agent_vendor_compat import compare_vendors

VENDOR_ICON = {"AMD Vitis DPU": "🟥", "NVIDIA TensorRT": "🟩", "Intel OpenVINO": "🟦"}
VERDICT_EMOJI = {"native": "🟢", "limited": "🟡", "unsupported": "🔴"}


def run_async(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


st.markdown("""
<div style="display:flex;align-items:center;gap:0.8rem;margin-bottom:0.25rem;">
  <div style="width:38px;height:38px;background:linear-gradient(135deg,#a855f7,#6d28d9);
              border-radius:10px;display:flex;align-items:center;justify-content:center;
              box-shadow:0 4px 14px rgba(168,85,247,0.45);"><span style="font-size:1.1rem;">🏁</span></div>
  <h1 style="margin:0;line-height:1.1;">Multi-Vendor Target Advisor</h1>
</div>
<p style="margin:0 0 1rem 0;color:rgba(100,116,139,0.9);font-size:0.85rem;padding-left:46px;">
  One model, three back-ends — <strong>AMD DPU · NVIDIA TensorRT · Intel OpenVINO</strong>.
  Where does it run as-is, and where's the porting cost? The question no single vendor answers.
</p>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("**Multi-Vendor**")
    src = repo_source_input()
    run = st.button("▶  Compare targets", type="primary", use_container_width=True)

if not run:
    st.info("Scan a model and compare how it maps onto each vendor's accelerator.", icon="🏁")
    st.stop()

with st.spinner("Resolving & scanning…"):
    resolved = resolve_repo(src)
    if resolved.get("error"):
        st.error(f"Could not load repo: {resolved['error']}")
        st.stop()
    scan = scan_repo({"repo_path": resolved["path"]})
    if scan.get("error"):
        st.error(f"Scan failed: {scan['error']}")
        st.stop()
    cmp = compare_vendors({"constructs": scan["constructs"]})

if cmp.get("error"):
    st.warning(cmp["error"]); st.stop()

# ── recommendation ──────────────────────────────────────────────────────────
st.success(cmp["recommendation"]["summary"], icon="🏁")

# ── per-vendor cards ────────────────────────────────────────────────────────
cols = st.columns(len(cmp["vendors"]))
for col, (name, v) in zip(cols, cmp["vendors"].items()):
    with col:
        st.markdown(f"#### {VENDOR_ICON.get(name,'▪️')} {name}")
        st.caption(f"{v['accel']} · power: {v['power']}")
        st.metric("Native (runs on accelerator)", f"{v['native_pct']}%")
        st.progress(min(v["native_pct"] / 100, 1.0))
        st.caption(f"Runs at all (native+limited): **{v['runs_pct']}%** · "
                   f"porting effort: **{v['port_effort_pct']}%**")

st.markdown("<br>", unsafe_allow_html=True)

# ── op-by-op matrix ─────────────────────────────────────────────────────────
st.markdown("### Op-by-op support matrix")
vendor_names = list(cmp["vendors"].keys())
# build construct → {vendor: verdict}
matrix = {}
for name in vendor_names:
    for o in cmp["vendors"][name]["ops"]:
        row = matrix.setdefault(o["construct"], {"construct": o["construct"], "op": o["op"], "uses": o["count"]})
        row[name] = f"{VERDICT_EMOJI.get(o['verdict'],'')} {o['verdict']}"
rows = sorted(matrix.values(), key=lambda r: -r["uses"])
st.dataframe(rows, use_container_width=True, hide_index=True)
st.caption("🟢 native · 🟡 limited (caveats) · 🔴 unsupported (host fallback / plugin). "
           "Curated op-support knowledge — TensorRT/OpenVINO accept a very broad ONNX opset; "
           "the AMD DPU is a fixed-function INT8 edge engine, so more falls back.")

with st.expander("How to read this"):
    st.markdown(
        "- **NVIDIA / Intel** typically run the model **as-is** (flexible, but higher power / cost).\n"
        "- **AMD DPU** needs the most refactoring but wins on **power for edge** deployment.\n"
        "- Use the **Auto-Fix & Verify** and **Structure Converter** tabs to close the AMD gap.")
