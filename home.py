"""Home page — feature overview. Rendered via st.navigation in app.py."""
import streamlit as st

from agents.llm import PROVIDERS

# ── shared card styling (each page run is independent, so inject here too) ──────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
footer { display: none !important; }
.home-card {
  background: rgba(15,23,42,0.55);
  border: 1px solid rgba(51,65,85,0.45);
  border-radius: 14px;
  padding: 1.25rem 1.35rem;
  height: 100%;
}
.home-card h4 { margin: 0 0 0.35rem 0; color: #e2e8f0; font-size: 0.95rem; font-weight: 700; }
.home-card p  { margin: 0; color: rgba(148,163,184,0.9); font-size: 0.82rem; line-height: 1.45; }
.home-pill {
  display: inline-block; padding: 0.2rem 0.6rem; margin: 0.15rem 0.25rem 0.15rem 0;
  border-radius: 999px; font-size: 0.72rem; font-weight: 600;
  background: rgba(37,99,235,0.14); color: #93c5fd; border: 1px solid rgba(37,99,235,0.3);
}
.step-num {
  display:inline-flex; align-items:center; justify-content:center;
  width: 26px; height: 26px; border-radius: 8px; font-size: 0.8rem; font-weight: 700;
  background: linear-gradient(135deg,#2563eb,#1e40af); color: #fff; margin-right: 0.6rem;
}
</style>
""", unsafe_allow_html=True)

# ── hero ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="display:flex; align-items:center; gap:0.8rem; margin-bottom:0.25rem;">
  <div style="width:44px;height:44px;background:linear-gradient(135deg,#2563eb,#1e40af);
              border-radius:12px;display:flex;align-items:center;justify-content:center;
              box-shadow:0 4px 14px rgba(37,99,235,0.45);">
    <span style="font-size:1.3rem;">⚡</span>
  </div>
  <div>
    <h1 style="margin:0;padding:0;line-height:1.1;">Vitis AI Porting Advisor</h1>
  </div>
</div>
<p style="margin:0.5rem 0 1.75rem 0; color:rgba(148,163,184,0.95); font-size:0.95rem; max-width:760px;">
  An AI pipeline that analyses a deep-learning model for <strong>Xilinx DPU</strong> hardware
  compatibility and generates two concrete refactoring strategies — one conservative,
  one aggressive INT8 — to maximise DPU subgraph coverage under Vitis&nbsp;AI&nbsp;3.x.
</p>
""", unsafe_allow_html=True)

st.page_link("advisor.py", label="Open the Advisor  →", icon="🔬")
st.markdown("<br>", unsafe_allow_html=True)

# ── how it works: 4-stage pipeline ──────────────────────────────────────────
st.subheader("How it works")
st.caption("Upload a YOLOv8 `.onnx` / `.pt` model, pick a DPU target, and run a 4-stage pipeline.")

stages = [
    ("🧩", "Model Extractor", "Traverses the ONNX graph and inventories every operator — names, types, and shapes."),
    ("🔎", "RAG Compatibility", "Cross-references each op against a ChromaDB knowledge base of Vitis AI DPU specs (93 entries, 12 archs) → severity: ok / warning / critical."),
    ("🛡️", "Conservative Proposal", "An LLM drafts a minimal-change porting strategy — accept a small CPU residual, maximise DPU coverage with low risk."),
    ("🚀", "Aggressive INT8 Proposal", "A second LLM call produces a full INT8 deployment plan — subgraph map, calibration pipeline, latency & power estimates."),
]
cols = st.columns(4)
for col, (icon, title, desc) in zip(cols, stages):
    with col:
        st.markdown(
            f'<div class="home-card"><div style="font-size:1.5rem;margin-bottom:0.4rem;">{icon}</div>'
            f'<h4>{title}</h4><p>{desc}</p></div>',
            unsafe_allow_html=True,
        )

st.markdown("<br>", unsafe_allow_html=True)

# ── features ────────────────────────────────────────────────────────────────
st.subheader("Features")

f1, f2 = st.columns(2)
with f1:
    st.markdown("""
    <div class="home-card">
      <h4>🔌 Bring-your-own LLM — 3 providers</h4>
      <p>Choose your provider and model in the sidebar, then paste your own API key.
      Calls go directly to the provider you pick — nothing is hardcoded.</p>
    </div>
    """, unsafe_allow_html=True)
with f2:
    st.markdown("""
    <div class="home-card">
      <h4>🎛️ Provider → Model → Key cascade</h4>
      <p>The model dropdown repopulates from the provider you select, and the key field's
      label/placeholder adapt to match (e.g. <code>sk-ant-…</code>, <code>xai-…</code>).</p>
    </div>
    """, unsafe_allow_html=True)

# provider/model matrix — pulled live from agents.llm so it never drifts
st.markdown("<br>", unsafe_allow_html=True)
pcols = st.columns(len(PROVIDERS))
for col, (name, cfg) in zip(pcols, PROVIDERS.items()):
    pills = "".join(f'<span class="home-pill">{m}</span>' for m in cfg["models"])
    with col:
        st.markdown(
            f'<div class="home-card"><h4>{name}</h4><p style="margin-bottom:0.5rem;">'
            f'<code>{cfg["base_url"]}</code></p>{pills}</div>',
            unsafe_allow_html=True,
        )

st.markdown("<br>", unsafe_allow_html=True)

g1, g2, g3 = st.columns(3)
with g1:
    st.markdown("""
    <div class="home-card">
      <h4>🎯 12 DPU targets</h4>
      <p>From embedded DPUCZDX8G (B512–B4096) and DPUCV2DX8G to data-centre Alveo
      DPUCAHX8H / DPUCADF8H — each scan is tailored to the chosen architecture.</p>
    </div>
    """, unsafe_allow_html=True)
with g2:
    st.markdown("""
    <div class="home-card">
      <h4>📚 RAG knowledge base</h4>
      <p>A local ChromaDB store of 93 Vitis AI operator-compatibility entries spanning
      30+ op types and all 12 DPU architectures grounds the compatibility scan.</p>
    </div>
    """, unsafe_allow_html=True)
with g3:
    st.markdown("""
    <div class="home-card">
      <h4>🗂️ Persistent run history</h4>
      <p>Every scan is saved to a Butterbase database — model name, op counts,
      unsupported ops, and both proposals — and listed in the sidebar for quick recall.</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.info("Head to the **Advisor** page (sidebar, or the button above) to run a scan.", icon="🔬")
