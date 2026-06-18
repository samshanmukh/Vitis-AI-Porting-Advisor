"""Advisor page — the 4-stage Vitis AI porting pipeline. Rendered via st.navigation in app.py."""
import asyncio
import json
import os
import sys

import streamlit as st
from butterbase import save_run, list_runs
from agents.llm import PROVIDERS

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# ── CSS — only custom components; native elements use Streamlit theme ─────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Streamlit chrome ─────────────────────────────────────────────────── */
footer { display: none !important; }
[data-testid="stDecoration"] { display: none !important; }

/* Sidebar always open — hide collapse/expand buttons */
[data-testid="stSidebarCollapseButton"] { display: none !important; }
[data-testid="collapsedControl"]        { display: none !important; }
.block-container {
    padding: 4rem 2rem 3rem !important;
    max-width: 1440px !important;
}

/* Running button spinner */
@keyframes spin { to { transform: rotate(360deg); } }
@keyframes btn-pulse {
    0%,100% { opacity: 1; } 50% { opacity: 0.75; }
}
.run-loading-btn {
    display: flex; align-items: center; justify-content: center; gap: 0.55rem;
    width: 100%; padding: 0.55rem 1rem;
    background: var(--primary-color, #ff4b4b);
    border-radius: 8px; cursor: not-allowed;
    font-size: 0.85rem; font-weight: 600; color: white;
    animation: btn-pulse 1.4s ease-in-out infinite;
    user-select: none;
}
.run-loading-btn .spin-ring {
    width: 14px; height: 14px;
    border: 2.5px solid rgba(255,255,255,0.3);
    border-top-color: white; border-radius: 50%;
    animation: spin 0.7s linear infinite; flex-shrink: 0;
}

/* Badges */
.neo-badge {
    display: inline-block; padding: 2px 10px; border-radius: 20px;
    font-size: 0.72rem; font-weight: 600; letter-spacing: 0.04em; text-transform: uppercase;
}
.neo-badge-blue  { background: rgba(37,99,235,0.15);  color: #93c5fd; border: 1px solid rgba(59,130,246,0.3); }
.neo-badge-red   { background: rgba(220,38,38,0.12);  color: #fca5a5; border: 1px solid rgba(239,68,68,0.3); }
.neo-badge-amber { background: rgba(217,119,6,0.12);  color: #fcd34d; border: 1px solid rgba(245,158,11,0.3); }
.neo-badge-green { background: rgba(5,150,105,0.12);  color: #6ee7b7; border: 1px solid rgba(16,185,129,0.3); }

/* Pipeline status rows */
.neo-stat-row {
    display: flex; align-items: center; gap: 0.6rem;
    padding: 0.55rem 0; border-bottom: 1px solid rgba(128,128,128,0.15);
}
.neo-stat-row:last-child { border-bottom: none; }
.neo-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
.neo-dot-blue  { background: #3b82f6; box-shadow: 0 0 6px rgba(59,130,246,0.7); }
.neo-dot-amber { background: #f59e0b; box-shadow: 0 0 6px rgba(245,158,11,0.7); }
.neo-dot-red   { background: #ef4444; box-shadow: 0 0 6px rgba(239,68,68,0.7); }
.neo-dot-green { background: #10b981; box-shadow: 0 0 6px rgba(16,185,129,0.7); }

/* Deployment pipeline steps */
.pipeline-step {
    display: flex; align-items: flex-start; gap: 0.9rem;
    padding: 0.7rem 0; border-bottom: 1px solid rgba(128,128,128,0.12);
}
.pipeline-step:last-child { border-bottom: none; }
.step-num {
    min-width: 26px; height: 26px;
    background: linear-gradient(135deg, #2563eb, #1e40af);
    border-radius: 50%; display: flex; align-items: center; justify-content: center;
    font-size: 0.72rem; font-weight: 700; color: white;
    box-shadow: 0 2px 8px rgba(37,99,235,0.4); flex-shrink: 0; margin-top: 1px;
}
.step-text { font-size: 0.85rem; line-height: 1.5; }

/* Glass cards — theme-neutral (no hardcoded dark colors) */
.neo-card {
    border-radius: 16px; padding: 1.25rem 1.5rem; margin-bottom: 1rem;
    border: 1px solid rgba(128,128,128,0.15);
    background: rgba(128,128,128,0.05);
}

/* Strategy / accent card */
.neo-card-accent {
    background: rgba(37,99,235,0.07);
    border: 1px solid rgba(59,130,246,0.2);
    border-radius: 12px; padding: 1rem 1.25rem; margin-bottom: 1rem;
}

/* CLI log panel */
.log-panel-header {
    display: flex; align-items: center; justify-content: space-between;
    background: #1a1a2e; border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px 10px 0 0; padding: 0.45rem 0.85rem; border-bottom: none;
}
.log-panel-dots { display: flex; gap: 5px; align-items: center; }
.log-panel-dots span { width: 10px; height: 10px; border-radius: 50%; display: block; }
.log-panel-title { font-size: 0.72rem; font-weight: 600; color: #64748b; font-family: 'JetBrains Mono', monospace; }
.log-panel-body {
    background: #0a0c10; border: 1px solid rgba(255,255,255,0.08);
    border-radius: 0 0 10px 10px;
    font-family: 'JetBrains Mono', 'Fira Code', 'Menlo', monospace;
    font-size: 0.72rem; line-height: 1.8; padding: 0.75rem 0.85rem;
    height: 420px; overflow-y: auto;
}
.log-panel-body .log-ok    { color: #6ee7b7; }
.log-panel-body .log-warn  { color: #fcd34d; }
.log-panel-body .log-err   { color: #fca5a5; }
.log-panel-body .log-info  { color: #93c5fd; }
.log-panel-body .log-dim   { color: #3d4a5c; }
.log-panel-body .log-white { color: #e2e8f0; }
.log-panel-idle {
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    height: 100%; gap: 0.5rem;
    color: #334155; font-size: 0.72rem; text-align: center;
    font-family: 'JetBrains Mono', monospace;
}
</style>
""", unsafe_allow_html=True)

# ── helpers ───────────────────────────────────────────────────────────────────


def run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def changes_table(changes: list):
    if not changes:
        return
    import pandas as pd
    df = pd.DataFrame([
        {
            "Op": c.get("op_name", ""),
            "Current": c.get("current", ""),
            "Replacement": c.get("replacement", ""),
            "Risk": c.get("risk", ""),
            "Effort": c.get("effort", ""),
        }
        for c in changes
    ])
    st.dataframe(df, use_container_width=True, hide_index=True)


def compat_table(report: list):
    if not report:
        return
    import pandas as pd
    df = pd.DataFrame([
        {
            "Op name": r.get("op_name", ""),
            "Type": r.get("op_type", ""),
            "Supported": "✅" if r.get("supported") else "❌",
            "Hint": r.get("optimization_hint", "")[:80],
        }
        for r in report
    ])
    st.dataframe(df, use_container_width=True, hide_index=True)


# ── page header ───────────────────────────────────────────────────────────────

st.markdown("""
<div style="display:flex; align-items:center; gap:0.8rem; margin-bottom:0.25rem;">
  <div style="width:38px;height:38px;background:linear-gradient(135deg,#2563eb,#1e40af);
              border-radius:10px;display:flex;align-items:center;justify-content:center;
              box-shadow:0 4px 14px rgba(37,99,235,0.45);">
    <span style="font-size:1.1rem;">⚡</span>
  </div>
  <div>
    <h1 style="margin:0;padding:0;line-height:1.1;">Vitis AI Porting Advisor</h1>
  </div>
</div>
<p style="margin:0 0 1.5rem 0; color:rgba(100,116,139,0.9); font-size:0.85rem; padding-left:46px;">
  Upload a YOLOv8 <code>.onnx</code> or <code>.pt</code> model &rarr; DPU compatibility scan &rarr; two AI-powered refactoring proposals
</p>
""", unsafe_allow_html=True)

# ── DPU targets ───────────────────────────────────────────────────────────────

DPU_TARGETS = {
    # ── DPUCZDX8G  — Zynq UltraScale+ ────────────────────────────────────
    "DPUCZDX8G B4096  · KV260 / ZCU102 / ZCU106 / Kria K26  [1024-ch, kernel≤16]":  "B4096",
    "DPUCZDX8G B3136  · custom Zynq UltraScale+ designs      [768-ch,  kernel≤12]":  "B3136",
    "DPUCZDX8G B2304  · ZCU104                               [512-ch,  kernel≤8]":   "B2304",
    "DPUCZDX8G B1600  · Ultra96-v2                           [384-ch,  kernel≤8]":   "B1600",
    "DPUCZDX8G B1152  · entry Zynq UltraScale+               [288-ch,  kernel≤8]":   "B1152",
    "DPUCZDX8G B1024  · Zynq UltraScale+ entry               [256-ch,  kernel≤8]":   "B1024",
    "DPUCZDX8G B800   · Zynq UltraScale+ low-power           [192-ch,  kernel≤6]":   "B800",
    "DPUCZDX8G B512   · Zynq UltraScale+ minimal             [128-ch,  kernel≤4]":   "B512",
    # ── DPUCVDX8G/H  — Versal ────────────────────────────────────────────
    "DPUCVDX8G        · VCK190  Versal AI Core  (AIE + PL DPU)":                      "DPUCVDX8G",
    "DPUCVDX8H        · VEK280  Versal AI Edge + HBM":                                "DPUCVDX8H",
    # ── DPUCAHX8H / DPUCADF8H  — Alveo PCIe ─────────────────────────────
    "DPUCAHX8H        · Alveo U50 / U200 / U250 / U280  (PCIe data-centre)":          "DPUCAHX8H",
    "DPUCADF8H        · Alveo U55C / V70  (data-centre, int4/fp16 support)":          "DPUCADF8H",
}

# ── sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:1rem;">
      <div style="width:6px;height:20px;background:linear-gradient(#2563eb,#1e40af);border-radius:3px;"></div>
      <span style="font-size:0.7rem;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:rgba(148,163,184,0.8);">Configuration</span>
    </div>
    """, unsafe_allow_html=True)

    provider     = st.selectbox("LLM provider", list(PROVIDERS.keys()), index=0)
    model        = st.selectbox("Model", PROVIDERS[provider]["models"], index=0)
    api_key      = st.text_input(
        f"{provider} API Key",
        type="password",
        placeholder=PROVIDERS[provider]["key_prefix"],
    )
    uploaded     = st.file_uploader("Model file (.onnx or .pt)", type=["onnx", "pt"])
    target_label = st.selectbox("Target DPU hardware", list(DPU_TARGETS.keys()), index=0)
    target_arch  = DPU_TARGETS[target_label]

    st.markdown("<br>", unsafe_allow_html=True)
    run_btn = st.button(
        "▶  Run Pipeline",
        type="primary",
        use_container_width=True,
        disabled=not (api_key and uploaded),
    )

    st.markdown("""<hr style="margin:1.25rem 0;">""", unsafe_allow_html=True)

    st.markdown("""
    <div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:0.75rem;">
      <div style="width:6px;height:20px;background:linear-gradient(#0f766e,#0d9488);border-radius:3px;"></div>
      <span style="font-size:0.7rem;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:rgba(148,163,184,0.8);">Run History</span>
    </div>
    """, unsafe_allow_html=True)

    past_runs = list_runs(limit=8)
    if past_runs:
        for r in past_runs:
            ts = r.get("created_at", "")[:16].replace("T", " ")
            unsup = r.get("unsupported_count", 0)
            badge_color = "#ef4444" if unsup > 5 else "#f59e0b" if unsup > 0 else "#10b981"
            st.markdown(f"""
            <div style="background:rgba(15,23,42,0.5);border:1px solid rgba(51,65,85,0.35);
                        border-radius:10px;padding:0.55rem 0.75rem;margin-bottom:0.4rem;">
              <div style="font-size:0.78rem;font-weight:600;color:#e2e8f0;margin-bottom:2px;">
                {r.get('model_name','?')}
              </div>
              <div style="display:flex;gap:0.4rem;align-items:center;">
                <span style="font-size:0.68rem;color:rgba(100,116,139,0.8);">{r.get('total_ops','?')} ops</span>
                <span style="color:rgba(51,65,85,0.8);">·</span>
                <span style="font-size:0.68rem;color:{badge_color};">{unsup} unsupported</span>
                <span style="color:rgba(51,65,85,0.8);">·</span>
                <span style="font-size:0.68rem;color:rgba(71,85,105,0.8);">{ts}</span>
              </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="color:rgba(71,85,105,0.7);font-size:0.75rem;text-align:center;
                    padding:1rem;border:1px dashed rgba(51,65,85,0.3);border-radius:10px;">
          No runs yet
        </div>
        """, unsafe_allow_html=True)

# ── idle state ────────────────────────────────────────────────────────────────

if not run_btn:
    st.markdown("""
    <div class="neo-card" style="text-align:center;padding:2.5rem 2rem;">
      <div style="font-size:2.5rem;margin-bottom:0.75rem;filter:drop-shadow(0 0 12px rgba(59,130,246,0.4));">🔬</div>
      <div style="font-size:1rem;font-weight:600;color:#e2e8f0;margin-bottom:0.4rem;">
        Ready to analyse
      </div>
      <div style="font-size:0.82rem;color:rgba(100,116,139,0.8);max-width:360px;margin:0 auto;">
        Pick an LLM provider and model, enter your API key, upload a <code>.onnx</code> or
        <code>.pt</code> model file, select a target DPU architecture, then hit <strong>Run Pipeline</strong>.
      </div>
    </div>
    """, unsafe_allow_html=True)

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("""
        <div class="neo-card" style="text-align:center;padding:1.25rem;">
          <div style="font-size:1.5rem;margin-bottom:0.4rem;">🧩</div>
          <div style="font-size:0.78rem;font-weight:700;color:#e2e8f0;margin-bottom:0.3rem;">Op Extraction</div>
          <div style="font-size:0.72rem;color:rgba(100,116,139,0.8);">Parses every ONNX node and counts op types</div>
        </div>""", unsafe_allow_html=True)
    with col_b:
        st.markdown("""
        <div class="neo-card" style="text-align:center;padding:1.25rem;">
          <div style="font-size:1.5rem;margin-bottom:0.4rem;">🔎</div>
          <div style="font-size:0.78rem;font-weight:700;color:#e2e8f0;margin-bottom:0.3rem;">RAG Scan</div>
          <div style="font-size:0.72rem;color:rgba(100,116,139,0.8);">Cross-references ops against Vitis AI DPU specs</div>
        </div>""", unsafe_allow_html=True)
    with col_c:
        st.markdown("""
        <div class="neo-card" style="text-align:center;padding:1.25rem;">
          <div style="font-size:1.5rem;margin-bottom:0.4rem;">🤖</div>
          <div style="font-size:0.78rem;font-weight:700;color:#e2e8f0;margin-bottom:0.3rem;">AI Proposals</div>
          <div style="font-size:0.72rem;color:rgba(100,116,139,0.8);">Two refactoring strategies — conservative & aggressive INT8</div>
        </div>""", unsafe_allow_html=True)

    st.stop()

# ── save upload ───────────────────────────────────────────────────────────────

import tempfile, pathlib
import pandas as pd

suffix    = pathlib.Path(uploaded.name).suffix
tmp       = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
tmp.write(uploaded.read())
tmp.flush()
model_path = tmp.name

os.environ["LLM_PROVIDER"] = provider
os.environ["LLM_MODEL"]    = model
os.environ["LLM_API_KEY"]  = api_key.strip()

# ── pipeline ──────────────────────────────────────────────────────────────────

from agents.agent_one_modelExtractor import extract_model
from agents.agent_two_RAG            import search_vitis_compatibility
from agents.agent_three_proposal_a   import generate_conservative_proposal
from agents.agent_three_proposal_b   import generate_aggressive_proposal

status = st.status("Running pipeline…", expanded=True)

with status:
    st.markdown(f"""
    <div class="neo-stat-row">
      <div class="neo-dot neo-dot-blue" style="animation:pulse 1.5s infinite;"></div>
      <span style="font-size:0.85rem;color:#e2e8f0;">
        <strong>Stage 1 / 4</strong> &mdash; Extracting model operations&hellip;
      </span>
    </div>""", unsafe_allow_html=True)
    extraction = run_async(extract_model({"model_path": model_path}))
    if "error" in extraction:
        st.error(f"Extraction failed: {extraction['error']}")
        st.stop()
    st.markdown(f"""
    <div class="neo-stat-row">
      <div class="neo-dot neo-dot-green"></div>
      <span style="font-size:0.82rem;color:rgba(110,231,183,0.9);">
        Found <strong>{extraction['total_ops']} operations</strong>
      </span>
    </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="neo-stat-row">
      <div class="neo-dot neo-dot-blue" style="animation:pulse 1.5s infinite;"></div>
      <span style="font-size:0.85rem;color:#e2e8f0;">
        <strong>Stage 2 / 4</strong> &mdash; Scanning against <code>{target_arch}</code> DPU&hellip;
      </span>
    </div>""", unsafe_allow_html=True)
    compatibility = run_async(search_vitis_compatibility({
        "operations":  extraction["operations"],
        "target_arch": target_arch,
    }))
    if "error" in compatibility:
        st.error(f"RAG scan failed: {compatibility['error']}")
        st.stop()
    crit = compatibility.get("critical_types", [])
    warn = compatibility.get("warning_types", [])
    st.markdown(f"""
    <div class="neo-stat-row">
      <div class="neo-dot neo-dot-green"></div>
      <span style="font-size:0.82rem;color:rgba(110,231,183,0.9);">
        Scanned <strong>{compatibility['total_checked']} ops</strong>
        ({compatibility.get('unique_op_types','?')} unique types) &mdash;
        <span style="color:#fca5a5;">{compatibility['unsupported_count']} unsupported</span>,
        🔴 {len(crit)} critical, 🟡 {len(warn)} warnings
      </span>
    </div>""", unsafe_allow_html=True)

    shared = {
        "model_name":    extraction["model_name"],
        "operations":    extraction["operations"],
        "unsupported_ops": compatibility["unsupported_ops"],
        "op_summary":    compatibility.get("op_summary", {}),
        "critical_types": crit,
        "warning_types":  warn,
        "target_arch":    target_arch,
    }

    st.markdown("""
    <div class="neo-stat-row">
      <div class="neo-dot neo-dot-blue" style="animation:pulse 1.5s infinite;"></div>
      <span style="font-size:0.85rem;color:#e2e8f0;">
        <strong>Stage 3 / 4</strong> &mdash; Running conservative hardware audit&hellip;
      </span>
    </div>""", unsafe_allow_html=True)
    proposal_a = run_async(generate_conservative_proposal(shared))
    st.markdown("""
    <div class="neo-stat-row">
      <div class="neo-dot neo-dot-green"></div>
      <span style="font-size:0.82rem;color:rgba(110,231,183,0.9);">Hardware audit complete</span>
    </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="neo-stat-row">
      <div class="neo-dot neo-dot-blue" style="animation:pulse 1.5s infinite;"></div>
      <span style="font-size:0.85rem;color:#e2e8f0;">
        <strong>Stage 4 / 4</strong> &mdash; Generating aggressive INT8 deployment plan&hellip;
      </span>
    </div>""", unsafe_allow_html=True)
    proposal_b = run_async(generate_aggressive_proposal(shared))
    st.markdown("""
    <div class="neo-stat-row">
      <div class="neo-dot neo-dot-green"></div>
      <span style="font-size:0.82rem;color:rgba(110,231,183,0.9);">Deployment plan ready</span>
    </div>""", unsafe_allow_html=True)

status.update(label="✅  Pipeline complete", state="complete")
save_run(extraction, compatibility, proposal_a, proposal_b)

# ── target hardware badge + top metrics ───────────────────────────────────────

st.markdown(f"""
<div style="display:flex;align-items:center;gap:0.6rem;margin:1.25rem 0 0.75rem;">
  <span style="font-size:0.7rem;font-weight:600;color:rgba(100,116,139,0.7);
               text-transform:uppercase;letter-spacing:0.06em;">Target hardware</span>
  <span class="neo-badge neo-badge-blue">{target_arch}</span>
  <span style="font-size:0.72rem;color:rgba(71,85,105,0.7);">{target_label.split('·')[-1].strip() if '·' in target_label else ''}</span>
</div>
""", unsafe_allow_html=True)

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Total ops",       extraction["total_ops"])
c2.metric("Unique types",    compatibility.get("unique_op_types", "—"))
c3.metric("Unsupported",     compatibility["unsupported_count"])
c4.metric("Critical flags",  len(crit))
c5.metric("Audit coverage",  proposal_a.get("vitis_compatibility",
          proposal_a.get("scan_summary", {}).get("dpu_eligible_after", "—")))
c6.metric("INT8 coverage",   proposal_b.get("vitis_compatibility", "—"))

st.markdown("<br>", unsafe_allow_html=True)

# ── tabs ──────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs([
    "🔬  Hardware Scan",
    "🛡️  Audit — Conservative",
    "⚡  Deployment — Aggressive INT8",
    "📄  Full JSON",
])

# ════════════════════════════════════════════════════════════════════════════
with tab1:
    op_sum = compatibility.get("op_summary", {})

    if op_sum:
        severity_order = {"critical": 0, "warning": 1, "ok": 2}
        sorted_ops = sorted(op_sum.items(), key=lambda x: severity_order.get(x[1].get("severity", "ok"), 2))

        # summary badges
        n_crit  = sum(1 for _, d in sorted_ops if d.get("severity") == "critical")
        n_warn  = sum(1 for _, d in sorted_ops if d.get("severity") == "warning")
        n_ok    = sum(1 for _, d in sorted_ops if d.get("severity") == "ok")
        st.markdown(f"""
        <div style="display:flex;gap:0.5rem;margin-bottom:1rem;flex-wrap:wrap;">
          <span class="neo-badge neo-badge-red">🔴 {n_crit} critical</span>
          <span class="neo-badge neo-badge-amber">🟡 {n_warn} warnings</span>
          <span class="neo-badge neo-badge-green">✅ {n_ok} clean</span>
        </div>
        """, unsafe_allow_html=True)

        severity_icon = {"critical": "🔴", "warning": "🟡", "ok": "✅"}
        rows = []
        for t, d in sorted_ops:
            rows.append({
                "Status":    severity_icon.get(d.get("severity", "ok"), "—"),
                "Op type":   t,
                "Count":     d.get("count", 0),
                "Supported": "Yes" if d.get("supported") else "No",
                "Hint":      d.get("optimization_hint", "")[:90],
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # alert strips
    if crit:
        st.markdown(f"""
        <div style="background:rgba(220,38,38,0.07);border:1px solid rgba(239,68,68,0.25);
                    border-radius:12px;padding:0.75rem 1rem;margin:0.5rem 0;">
          <span style="color:#fca5a5;font-size:0.82rem;font-weight:600;">🔴 Critical ops</span>
          <span style="color:rgba(252,165,165,0.7);font-size:0.82rem;"> — {', '.join(crit)}</span>
        </div>""", unsafe_allow_html=True)
    if warn:
        st.markdown(f"""
        <div style="background:rgba(217,119,6,0.07);border:1px solid rgba(245,158,11,0.25);
                    border-radius:12px;padding:0.75rem 1rem;margin:0.5rem 0;">
          <span style="color:#fcd34d;font-size:0.82rem;font-weight:600;">🟡 Warnings</span>
          <span style="color:rgba(252,211,77,0.7);font-size:0.82rem;"> — {', '.join(warn)}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div style="font-size:0.78rem;font-weight:600;color:rgba(100,116,139,0.7);
                text-transform:uppercase;letter-spacing:0.06em;margin:1.25rem 0 0.5rem;">
      Per-op scan — all {len(compatibility.get('compatibility_report',[]))} ops
    </div>""", unsafe_allow_html=True)
    compat_table(compatibility["compatibility_report"])

# ════════════════════════════════════════════════════════════════════════════
with tab2:
    if "error" in proposal_a and "scan_summary" not in proposal_a:
        st.error(proposal_a["error"])
        st.code(proposal_a.get("raw_response", ""), language="json")
    else:
        ss = proposal_a.get("scan_summary", {})

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("DPU coverage — before", ss.get("dpu_eligible_before", "—"))
        c2.metric("DPU coverage — after",  ss.get("dpu_eligible_after",  "—"))
        c3.metric("Risk level",            ss.get("risk_level", "—"))
        c4.metric("Dev effort",            proposal_a.get("dev_effort", "—"))

        st.markdown("<br>", unsafe_allow_html=True)

        if proposal_a.get("strategy"):
            st.markdown(f"""
            <div class="neo-card-accent">
              <div style="font-size:0.68rem;font-weight:700;text-transform:uppercase;
                          letter-spacing:0.08em;color:rgba(96,165,250,0.7);margin-bottom:0.4rem;">
                Strategy
              </div>
              <div style="font-size:0.88rem;color:#e2e8f0;line-height:1.6;">
                {proposal_a.get('strategy','—')}
              </div>
            </div>""", unsafe_allow_html=True)

        if ss.get("cpu_residual_ops"):
            ops_str = ", ".join(f"<code>{o}</code>" for o in ss["cpu_residual_ops"])
            st.markdown(f"""
            <div style="background:rgba(37,99,235,0.06);border:1px solid rgba(59,130,246,0.2);
                        border-radius:12px;padding:0.75rem 1rem;margin-bottom:0.75rem;">
              <span style="font-size:0.78rem;color:rgba(147,197,253,0.8);">
                CPU residual after fixes: {ops_str}
              </span>
            </div>""", unsafe_allow_html=True)

        if proposal_a.get("dpu_arch_notes"):
            st.markdown(f"""
            <div style="font-size:0.78rem;color:rgba(100,116,139,0.8);
                        font-style:italic;margin-bottom:1rem;padding:0 0.25rem;">
              🔧 {proposal_a['dpu_arch_notes']}
            </div>""", unsafe_allow_html=True)

        findings = proposal_a.get("critical_findings", [])
        if findings:
            st.markdown("""
            <div style="font-size:0.78rem;font-weight:600;color:rgba(100,116,139,0.7);
                        text-transform:uppercase;letter-spacing:0.06em;margin:1rem 0 0.5rem;">
              Critical findings
            </div>""", unsafe_allow_html=True)
            st.dataframe(pd.DataFrame(findings), use_container_width=True, hide_index=True)

        changes = proposal_a.get("recommended_changes", proposal_a.get("changes", []))
        if changes:
            st.markdown("""
            <div style="font-size:0.78rem;font-weight:600;color:rgba(100,116,139,0.7);
                        text-transform:uppercase;letter-spacing:0.06em;margin:1rem 0 0.5rem;">
              Recommended changes
            </div>""", unsafe_allow_html=True)
            changes_table(changes)

# ════════════════════════════════════════════════════════════════════════════
with tab3:
    if "error" in proposal_b and "dpu_subgraph" not in proposal_b:
        st.error(proposal_b["error"])
        st.code(proposal_b.get("raw_response", ""), language="json")
    else:
        perf = proposal_b.get("performance_estimates", {})

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("FPS @ 1080p",     perf.get("fps_1080p",        "—"))
        c2.metric("Latency",         perf.get("latency_ms",        "—"))
        c3.metric("Power draw",      perf.get("power_draw_w",      "—"))
        c4.metric("Memory",          perf.get("memory_mb",         "—"))
        c5.metric("vs CPU baseline", perf.get("vs_cpu_baseline",   "—"))

        st.markdown("<br>", unsafe_allow_html=True)

        sg = proposal_b.get("dpu_subgraph", {})
        if sg:
            st.markdown("""
            <div style="font-size:0.78rem;font-weight:600;color:rgba(100,116,139,0.7);
                        text-transform:uppercase;letter-spacing:0.06em;margin-bottom:0.75rem;">
              DPU subgraph map
            </div>""", unsafe_allow_html=True)

            sc1, sc2, sc3 = st.columns(3)
            sc1.metric("DPU ops",      sg.get("dpu_op_count", "—"))
            sc2.metric("CPU ops",      sg.get("cpu_op_count", "—"))
            sc3.metric("DPU coverage", sg.get("dpu_coverage", "—"))

            st.markdown(f"""
            <div class="neo-card" style="padding:0.75rem 1rem;margin-top:0.5rem;">
              <span style="font-size:0.78rem;color:rgba(100,116,139,0.7);">First DPU op:</span>
              <code style="margin-left:0.5rem;">{sg.get('first_dpu_op','—')}</code>
              <span style="margin:0 0.5rem;color:rgba(51,65,85,0.8);">→</span>
              <span style="font-size:0.78rem;color:rgba(100,116,139,0.7);">Last DPU op:</span>
              <code style="margin-left:0.5rem;">{sg.get('last_dpu_op','—')}</code>
            </div>""", unsafe_allow_html=True)

        cpu_res = proposal_b.get("cpu_residual", [])
        if cpu_res:
            st.markdown("""
            <div style="font-size:0.78rem;font-weight:600;color:rgba(100,116,139,0.7);
                        text-transform:uppercase;letter-spacing:0.06em;margin:1rem 0 0.5rem;">
              CPU residual ops
            </div>""", unsafe_allow_html=True)
            st.dataframe(pd.DataFrame(cpu_res), use_container_width=True, hide_index=True)

        qp = proposal_b.get("quantization_plan", {})
        if qp:
            st.markdown("""
            <div style="font-size:0.78rem;font-weight:600;color:rgba(100,116,139,0.7);
                        text-transform:uppercase;letter-spacing:0.06em;margin:1rem 0 0.75rem;">
              Quantization plan
            </div>""", unsafe_allow_html=True)
            qc1, qc2, qc3 = st.columns(3)
            qc1.metric("Method",         qp.get("method",             "—"))
            qc2.metric("Cal. images",    qp.get("calibration_images", "—"))
            qc3.metric("Expected SQNR",  qp.get("expected_sqnr_db",   "—"))
            if qp.get("calibration_cmd"):
                st.code(qp["calibration_cmd"], language="bash")
            if qp.get("sensitive_layers"):
                st.markdown(f"""
                <div style="font-size:0.78rem;color:rgba(100,116,139,0.8);margin-top:0.5rem;">
                  Sensitive layers: {", ".join(f"<code>{l}</code>" for l in qp["sensitive_layers"])}
                </div>""", unsafe_allow_html=True)

        dp = proposal_b.get("deployment_pipeline", [])
        if dp:
            st.markdown("""
            <div style="font-size:0.78rem;font-weight:600;color:rgba(100,116,139,0.7);
                        text-transform:uppercase;letter-spacing:0.06em;margin:1rem 0 0.75rem;">
              Deployment pipeline
            </div>""", unsafe_allow_html=True)
            steps_html = "".join(
                f"""<div class="pipeline-step">
                      <div class="step-num">{i}</div>
                      <div class="step-text">{step}</div>
                    </div>"""
                for i, step in enumerate(dp, 1)
            )
            st.markdown(f'<div class="neo-card" style="padding:0.75rem 1.1rem;">{steps_html}</div>',
                        unsafe_allow_html=True)

        arch_changes = proposal_b.get("architecture_changes", proposal_b.get("changes", []))
        if arch_changes:
            st.markdown("""
            <div style="font-size:0.78rem;font-weight:600;color:rgba(100,116,139,0.7);
                        text-transform:uppercase;letter-spacing:0.06em;margin:1rem 0 0.5rem;">
              Architecture changes
            </div>""", unsafe_allow_html=True)
            changes_table(arch_changes)

# ════════════════════════════════════════════════════════════════════════════
with tab4:
    full = {
        "model_name":    extraction["model_name"],
        "extraction":    extraction,
        "compatibility": compatibility,
        "proposal_a":    proposal_a,
        "proposal_b":    proposal_b,
    }
    st.code(json.dumps(full, indent=2), language="json")
    st.download_button(
        "⬇  Download JSON report",
        json.dumps(full, indent=2),
        "vitis_analysis.json",
        "application/json",
    )
