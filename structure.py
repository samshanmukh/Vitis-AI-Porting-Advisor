"""Structure Converter page — paste a GitHub model repo, see what changes to make it DPU-portable."""
import asyncio
import os
import sys

import streamlit as st

from agents.llm import PROVIDERS

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from agents.repo_fetch import resolve_repo
from agents.agent_repo_scanner import scan_repo
from agents.agent_two_RAG import search_vitis_compatibility
from agents.agent_observer import observe_and_capture
from agents.agent_converter import transform_structure, narrate_conversion, save_conversion

DPU_ARCHS = ["B512", "B800", "B1024", "B1152", "B1600", "B2304", "B3136", "B4096",
             "DPUCVDX8G", "DPUCAHX8H", "DPUCADF8H"]


def run_async(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


# ── header ──────────────────────────────────────────────────────────────────
st.markdown("""
<div style="display:flex;align-items:center;gap:0.8rem;margin-bottom:0.25rem;">
  <div style="width:38px;height:38px;background:linear-gradient(135deg,#0d9488,#0f766e);
              border-radius:10px;display:flex;align-items:center;justify-content:center;
              box-shadow:0 4px 14px rgba(13,148,136,0.45);"><span style="font-size:1.1rem;">🔀</span></div>
  <h1 style="margin:0;line-height:1.1;">Structure Converter</h1>
</div>
<p style="margin:0 0 1rem 0;color:rgba(100,116,139,0.9);font-size:0.85rem;padding-left:46px;">
  Paste a GitHub model repo &rarr; pick a DPU target &rarr; see <strong>what changes in the structure</strong>
  to make it DPU-portable, and the coverage it buys.
</p>
""", unsafe_allow_html=True)

st.caption("ℹ️ This shows the **proposed structural conversion** (the refactor toward the "
           "`float → INT8 → .xmodel` target) and the DPU coverage it would buy — not a literally compiled .xmodel.")

# ── inputs ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("**Structure Converter**")
    source = st.text_input("GitHub repo URL (or local path)",
                           value="https://github.com/owner/model-repo")
    target_arch = st.selectbox("Target DPU arch", DPU_ARCHS, index=DPU_ARCHS.index("B4096"))
    use_sample = st.checkbox("Use bundled sample repo (offline)", value=True,
                             help="Scan the in-repo USSFCNet example instead of cloning.")
    st.markdown("---")
    st.caption("AI narrative (optional)")
    provider = st.selectbox("LLM provider", list(PROVIDERS.keys()), index=0)
    model = st.selectbox("Model", PROVIDERS[provider]["models"], index=0)
    api_key = st.text_input(f"{provider} API Key", type="password",
                            placeholder=PROVIDERS[provider]["key_prefix"])
    run = st.button("▶  Convert structure", type="primary", use_container_width=True)

if not run:
    st.info("Paste a GitHub repo URL (e.g. a PyTorch model), or use the bundled sample, then convert.", icon="🔀")
    st.stop()

# ── resolve repo (clone if URL) ─────────────────────────────────────────────
src = "model_input/Transfer-Model_original" if use_sample else source
with st.spinner("Resolving repo…"):
    resolved = resolve_repo(src)
if resolved.get("error"):
    st.error(f"Could not load repo: {resolved['error']}")
    st.stop()
st.success(f"Loaded **{resolved['name']}** ({resolved['origin']}) → scanning architecture…", icon="📦")

# ── scan → compat → capture → transform (all deterministic) ─────────────────
scan = scan_repo({"repo_path": resolved["path"]})
if scan.get("error"):
    st.error(f"Scan failed: {scan['error']}")
    st.stop()
compat = run_async(search_vitis_compatibility({"operations": scan["operations"], "target_arch": target_arch}))
capture = observe_and_capture({"scan": scan, "compatibility": compat, "target_arch": target_arch})
transform = transform_structure({"capture": capture})
if transform.get("error"):
    st.warning("No incompatible constructs found — nothing to convert. The model already maps cleanly.")
    st.stop()

cov = transform["coverage"]

# persist the restructured output so it can be re-scanned / shown / downloaded
saved = save_conversion(transform, resolved["name"], target_arch, capture.get("architecture"))

# ── coverage before / after ─────────────────────────────────────────────────
st.markdown("### DPU coverage — before vs after")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Before", f"{cov['coverage_before_pct']}%")
m2.metric("After", f"{cov['coverage_after_pct']}%", delta=f"+{cov['delta_pct']}%")
m3.metric("Moved to CPU", cov["moved_to_cpu"])
m4.metric("Changes", len(transform["changes"]))
st.progress(min(cov["coverage_after_pct"] / 100, 1.0),
            text=f"{cov['dpu_after']} of {cov['total_construct_uses']} construct-uses on DPU after conversion")

# ── what changed in the structure ───────────────────────────────────────────
st.markdown("### What changes in the structure")
for c in transform["changes"]:
    land_color = "#0d9488" if c["lands"] == "dpu" else "#f59e0b"
    land_label = "→ stays on DPU" if c["lands"] == "dpu" else "→ moved to CPU"
    sev_color = "#ef4444" if c["severity"] == "critical" else "#f59e0b"
    with st.container(border=True):
        st.markdown(
            f"<span style='color:{sev_color};font-weight:700;'>●</span> "
            f"<code>{c['before']}</code> &nbsp;→&nbsp; <code style='color:#5eead4;'>{c['after']}</code> "
            f"<span style='color:{land_color};font-size:0.78rem;font-weight:600;'>&nbsp;{land_label}</span> "
            f"<span style='color:#64748b;font-size:0.78rem;'>· ×{c['count']}</span>",
            unsafe_allow_html=True)
        st.markdown(f"**Why change it:** {c['effect']}")
        if c.get("why"):
            st.caption(f"DPU rule: {c['why']}")
        if c.get("evidence"):
            st.caption("Where: " + ", ".join(c["evidence"]))

# ── converted structure (stored + downloadable) ─────────────────────────────
st.markdown("### 🗂️ Converted structure (stored)")
if saved.get("error"):
    st.warning(f"Could not save converted structure: {saved['error']}")
else:
    st.caption(f"Saved to `{saved['json_path']}` and `{saved['md_path']}` — re-scannable & downloadable.")
    d1, d2 = st.columns(2)
    d1.download_button("⬇ Download JSON", saved["json_text"],
                       file_name=os.path.basename(saved["json_path"]), mime="application/json",
                       use_container_width=True)
    d2.download_button("⬇ Download Markdown report", saved["markdown"],
                       file_name=os.path.basename(saved["md_path"]), mime="text/markdown",
                       use_container_width=True)

st.dataframe(
    [{"construct": c["construct"], "converted to": c["converted"],
      "changed": "✅" if c["changed"] else "—", "lands": c["lands"].upper(), "uses": c["count"]}
     for c in transform["converted_structure"]],
    use_container_width=True, hide_index=True)

# ── optional AI narrative of the converted structure ────────────────────────
st.markdown("### 🤖 Converted structure (AI summary)")
if not api_key:
    st.info("Enter an LLM API key in the sidebar for a narrative of the converted architecture. "
            "The before/after above is computed locally without a key.", icon="🔑")
else:
    os.environ["LLM_PROVIDER"] = provider
    os.environ["LLM_MODEL"] = model
    os.environ["LLM_API_KEY"] = api_key.strip()
    with st.spinner("Describing the converted structure…"):
        narr = run_async(narrate_conversion({"capture": capture, "transform": transform, "target_arch": target_arch}))
    if narr.get("error"):
        st.error(f"Narrative failed: {narr['error']}")
    else:
        st.markdown(narr.get("converted_summary", ""))
        for b in narr.get("structural_changes", []):
            st.markdown(f"- {b}")
        if narr.get("remaining_cpu"):
            st.caption(f"Stays on CPU: {narr['remaining_cpu']}")
        if narr.get("risk"):
            st.warning(f"Risk: {narr['risk']}")

with st.expander("Raw transform record"):
    st.json(transform)
