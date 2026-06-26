"""Phase 1 page — Preliminary investigation on model architecture (repo scan)."""
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
from agents.agent_arch_explainer import explain_architecture

DPU_ARCHS = ["B512", "B800", "B1024", "B1152", "B1600", "B2304", "B3136", "B4096",
             "DPUCVDX8G", "DPUCAHX8H", "DPUCADF8H"]
SEV_COLOR = {"critical": "#ef4444", "warning": "#f59e0b", "ok": "#10b981"}


def run_async(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


# ── header ──────────────────────────────────────────────────────────────────
st.markdown("""
<div style="display:flex;align-items:center;gap:0.8rem;margin-bottom:0.25rem;">
  <div style="width:38px;height:38px;background:linear-gradient(135deg,#7c3aed,#5b21b6);
              border-radius:10px;display:flex;align-items:center;justify-content:center;
              box-shadow:0 4px 14px rgba(124,58,237,0.45);"><span style="font-size:1.1rem;">🧬</span></div>
  <h1 style="margin:0;line-height:1.1;">Architecture Scan <span style="font-size:0.9rem;color:#a78bfa;">· Phase 1</span></h1>
</div>
<p style="margin:0 0 1.25rem 0;color:rgba(100,116,139,0.9);font-size:0.85rem;padding-left:46px;">
  Scan a model <strong>source repo</strong> &rarr; extract the architecture &amp; the functions it uses &rarr;
  compare against DPU docs &rarr; explain <strong>what is incompatible and why</strong>.
</p>
""", unsafe_allow_html=True)

# ── sidebar config ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("**Phase 1 · Repo Scan**")
    source = st.text_input("GitHub repo URL (or local path)", value="https://github.com/owner/model-repo")
    use_sample = st.checkbox("Use bundled sample repo (offline)", value=True,
                             help="Scan the in-repo USSFCNet example instead of cloning.")
    target_arch = st.selectbox("Target DPU arch", DPU_ARCHS, index=DPU_ARCHS.index("B4096"))

    st.markdown("---")
    st.caption("AI explanation (optional — scan + capture run without a key)")
    provider = st.selectbox("LLM provider", list(PROVIDERS.keys()), index=0)
    model = st.selectbox("Model", PROVIDERS[provider]["models"], index=0)
    api_key = st.text_input(f"{provider} API Key", type="password",
                            placeholder=PROVIDERS[provider]["key_prefix"])

    run = st.button("▶  Scan architecture", type="primary", use_container_width=True)

if not run:
    st.info("Paste a GitHub repo URL (a PyTorch model), or use the bundled USSFCNet sample, then scan.", icon="🧬")
    st.stop()

# ── resolve repo (clone if URL) ─────────────────────────────────────────────
src = "model_input/Transfer-Model_original" if use_sample else source
with st.spinner("Resolving repo…"):
    resolved = resolve_repo(src)
if resolved.get("error"):
    st.error(f"Could not load repo: {resolved['error']}")
    st.stop()
st.success(f"Loaded **{resolved['name']}** ({resolved['origin']}) → scanning architecture…", icon="📦")

# ── 1. static scan (deterministic) ──────────────────────────────────────────
scan = scan_repo({"repo_path": resolved["path"]})
if scan.get("error"):
    st.error(f"Scan failed: {scan['error']}")
    st.stop()

s = scan["summary"]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Files scanned", s["files_scanned"])
c2.metric("nn.Module classes", s["modules"])
c3.metric("Unique constructs", s["unique_constructs"])
c4.metric("Construct uses", s["total_construct_uses"])

# ── 2. compatibility (RAG) + 3. local capture (deterministic) ───────────────
compat = run_async(search_vitis_compatibility({"operations": scan["operations"], "target_arch": target_arch}))
capture = observe_and_capture({"scan": scan, "compatibility": compat, "target_arch": target_arch})

cc = capture["counts"]
st.markdown(f"""
<div style="display:flex;gap:0.6rem;align-items:center;margin:0.5rem 0 0.25rem;">
  <span class="neo-badge neo-badge-blue">{target_arch}</span>
  <span style="font-size:0.8rem;color:#94a3b8;">
    🔴 {cc['critical']} critical &nbsp;·&nbsp; 🟡 {cc['warning']} warning &nbsp;·&nbsp;
    {cc['issues']}/{cc['constructs']} constructs flagged</span>
</div>
""", unsafe_allow_html=True)
if capture.get("captured_path"):
    st.caption(f"🗂️ Local agent captured this scan → `{capture['captured_path']}`")

tab_why, tab_arch, tab_capture = st.tabs(["⚠️ Incompatibilities", "🧩 Architecture", "🗂️ Captured record"])

# ── what is incompatible & why ──────────────────────────────────────────────
with tab_why:
    if not capture["incompatible"]:
        st.success("No incompatibilities detected — every construct maps to a DPU-supported op.")
    for f in capture["incompatible"]:
        color = SEV_COLOR.get(f["severity"], "#94a3b8")
        with st.container(border=True):
            st.markdown(
                f"<span style='color:{color};font-weight:700;'>● {f['severity'].upper()}</span> "
                f"&nbsp;<code>{f['construct']}</code> → <code>{f['dpu_op']}</code> "
                f"<span style='color:#64748b;'>· used ×{f['count']}</span>",
                unsafe_allow_html=True)
            if f.get("dpu_rule"):
                st.markdown(f"**Why (DPU rule):** {f['dpu_rule']}")
            for cf in f.get("constraint_flags", []):
                st.markdown(f"- 🚩 {cf}")
            if f.get("optimization_hint"):
                st.markdown(f"**Fix hint:** {f['optimization_hint']}")
            st.caption("Where: " + ", ".join(f.get("evidence", [])))

    # ── 4. AI explanation (needs a key) ─────────────────────────────────────
    st.markdown("### 🤖 Agent explanation")
    if not api_key:
        st.info("Enter an LLM API key in the sidebar to generate the grounded AI explanation. "
                "The scan and capture above already ran locally without one.", icon="🔑")
    else:
        os.environ["LLM_PROVIDER"] = provider
        os.environ["LLM_MODEL"] = model
        os.environ["LLM_API_KEY"] = api_key.strip()
        with st.spinner("Agent explaining incompatibilities…"):
            exp = run_async(explain_architecture({"capture": capture, "target_arch": target_arch}))
        if exp.get("error"):
            st.error(f"Explanation failed: {exp['error']}")
        else:
            st.markdown(f"**Assessment:** {exp.get('architecture_assessment','')}")
            st.markdown(f"**Estimated DPU coverage:** {exp.get('estimated_dpu_coverage','—')}")
            if exp.get("top_priority"):
                st.warning(f"**Fix first:** {exp['top_priority']}")
            for inc in exp.get("incompatibilities", []):
                with st.container(border=True):
                    st.markdown(f"<code>{inc.get('construct','')}</code> → **{inc.get('verdict','')}** "
                                f"<span style='color:#64748b;'>({inc.get('impact','')})</span>",
                                unsafe_allow_html=True)
                    st.markdown(f"**What:** {inc.get('what','')}")
                    st.markdown(f"**Why:** {inc.get('why','')}")
                    st.markdown(f"**Fix:** {inc.get('fix','')}")
                    if inc.get("where"):
                        st.caption(f"Where: {inc['where']}")

# ── architecture map ────────────────────────────────────────────────────────
with tab_arch:
    st.markdown("**nn.Module classes** (the architecture)")
    for m in scan["modules"]:
        st.markdown(f"- `{m['name']}` &nbsp;<span style='color:#64748b;'>{m['file']}:{m['line']}</span>",
                    unsafe_allow_html=True)
    st.markdown("**Constructs used → DPU op**")
    st.dataframe(
        [{"construct": c["construct"], "DPU op": c["dpu_op"], "category": c["category"], "uses": c["count"]}
         for c in scan["constructs"]],
        use_container_width=True, hide_index=True)

# ── raw captured record ─────────────────────────────────────────────────────
with tab_capture:
    st.caption("This is exactly what the local observer agent persisted to disk.")
    st.json(capture)
