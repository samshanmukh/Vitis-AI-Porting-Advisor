"""Home page — feature overview. Rendered via st.navigation in app.py."""
import streamlit as st
import streamlit.components.v1 as components

from agents.llm import PROVIDERS

# ── dynamic constellation background ────────────────────────────────────────
# A canvas is injected behind the page content (z-index:-1). It twinkles a faint
# starfield and cycles through zodiac shapes, fading one in/out at a time so the
# connecting lines are visible but never dominate. Scoped to Home: the CSS below
# makes .stApp transparent only on this page, so on other pages the opaque app
# background covers the canvas.
components.html("""
<script>
(function () {
  const doc = window.parent.document;
  if (doc.getElementById('cosmos-canvas')) return;   // already running

  const cv = doc.createElement('canvas');
  cv.id = 'cosmos-canvas';
  cv.style.cssText =
    'position:fixed;inset:0;width:100vw;height:100vh;z-index:-1;' +
    'pointer-events:none;opacity:0.45;';
  doc.body.style.background = '#0e1117';
  doc.body.appendChild(cv);

  const ctx = cv.getContext('2d');
  let W, H, DPR;
  function resize() {
    DPR = window.parent.devicePixelRatio || 1;
    W = cv.clientWidth; H = cv.clientHeight;
    cv.width = W * DPR; cv.height = H * DPR;
    ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
  }
  window.parent.addEventListener('resize', resize);
  resize();

  // faint static starfield
  const bg = [];
  for (let i = 0; i < 170; i++)
    bg.push({ x: Math.random(), y: Math.random(), r: Math.random() * 1.1 + 0.3, ph: Math.random() * 6.28 });

  // stylized zodiac / constellation patterns: stars (0-100 box) + edges to connect
  const SIGNS = [
    { s: [[12,55],[38,48],[62,44],[86,52]], e: [[0,1],[1,2],[2,3]] },                                         // Aries
    { s: [[10,78],[34,58],[52,46],[72,34],[90,26],[58,60],[82,66]], e: [[0,1],[1,2],[2,3],[3,4],[2,5],[5,6]] }, // Taurus
    { s: [[22,18],[30,46],[32,74],[60,16],[66,46],[68,76]], e: [[0,1],[1,2],[3,4],[4,5],[1,4]] },               // Gemini
    { s: [[14,40],[30,30],[46,34],[50,52],[74,60],[88,50],[60,40]], e: [[0,1],[1,2],[2,3],[3,4],[4,5],[2,6],[6,5]] }, // Leo
    { s: [[14,24],[26,30],[40,34],[54,42],[62,56],[58,70],[44,80],[30,78]], e: [[0,1],[1,2],[2,3],[3,4],[4,5],[5,6],[6,7]] }, // Scorpius
    { s: [[20,42],[40,34],[58,40],[50,60],[34,62],[74,28],[66,54]], e: [[0,1],[1,2],[2,3],[3,4],[4,0],[1,5],[2,6]] }, // Sagittarius
    { s: [[50,12],[50,40],[50,70],[50,90],[22,46],[78,46]], e: [[0,1],[1,2],[2,3],[4,1],[1,5]] },               // Cygnus
    { s: [[24,16],[76,20],[40,46],[52,50],[64,54],[28,84],[80,86]], e: [[0,2],[1,4],[2,3],[3,4],[2,5],[4,6]] },  // Orion
  ];

  const PERIOD = 7000, FADE = 1500, t0 = performance.now();
  function draw(now) {
    const t = now - t0;
    ctx.clearRect(0, 0, W, H);

    for (const s of bg) {
      const a = 0.22 + 0.32 * Math.sin(t * 0.0011 + s.ph);
      if (a <= 0) continue;
      ctx.beginPath();
      ctx.arc(s.x * W, s.y * H, s.r, 0, 6.283);
      ctx.fillStyle = 'rgba(226,232,240,' + a + ')';
      ctx.fill();
    }

    const sign = SIGNS[Math.floor(t / PERIOD) % SIGNS.length];
    const tp = t % PERIOD;
    let f = 1;
    if (tp < FADE) f = tp / FADE;
    else if (tp > PERIOD - FADE) f = (PERIOD - tp) / FADE;

    const S = Math.min(W, H) * 0.42, cx = W * 0.5 - S / 2, cy = H * 0.42 - S / 2;
    const pt = (p) => [cx + p[0] / 100 * S, cy + p[1] / 100 * S];

    ctx.lineWidth = 1;
    ctx.strokeStyle = 'rgba(147,197,253,' + (0.55 * f) + ')';
    for (const [a, b] of sign.e) {
      const A = pt(sign.s[a]), B = pt(sign.s[b]);
      ctx.beginPath(); ctx.moveTo(A[0], A[1]); ctx.lineTo(B[0], B[1]); ctx.stroke();
    }
    for (let i = 0; i < sign.s.length; i++) {
      const P = pt(sign.s[i]), tw = 0.7 + 0.3 * Math.sin(t * 0.004 + i);
      ctx.beginPath(); ctx.arc(P[0], P[1], 4, 0, 6.283);
      ctx.fillStyle = 'rgba(96,165,250,' + (f * 0.14) + ')'; ctx.fill();       // glow
      ctx.beginPath(); ctx.arc(P[0], P[1], 1.8, 0, 6.283);
      ctx.fillStyle = 'rgba(191,219,254,' + (f * tw) + ')'; ctx.fill();        // core
    }
    requestAnimationFrame(draw);
  }
  requestAnimationFrame(draw);
})();
</script>
""", height=0)

# ── shared card styling (each page run is independent, so inject here too) ──────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
footer { display: none !important; }
/* let the constellation canvas (z-index:-1) show through on Home only */
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stHeader"] { background: transparent !important; }
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
<p style="margin:0.5rem 0 1.75rem 0; color:rgba(148,163,184,0.95); font-size:0.95rem; max-width:780px;">
  Point it at any model repo and it tells you <strong>why your model won't run on the accelerator,
  what to change, proves the fix by re-scanning</strong> — and compares
  <strong>AMD&nbsp;DPU · NVIDIA&nbsp;TensorRT · Intel&nbsp;OpenVINO</strong> so you pick the right target.
</p>
""", unsafe_allow_html=True)

st.markdown("""
<div style="border-left:3px solid #2563eb;background:rgba(37,99,235,0.08);
            border-radius:0 10px 10px 0;padding:0.9rem 1.1rem;margin:0.25rem 0 1.25rem 0;max-width:760px;">
  <p style="margin:0;font-size:1.0rem;line-height:1.5;color:#e2e8f0;">
    The <strong>Vitis AI Model Zoo</strong> tells you which models <em>already</em> run on the DPU.
    <span style="color:#93c5fd;font-weight:600;">We tell you why your model doesn't — and exactly what to change to fix it.</span>
  </p>
</div>
""", unsafe_allow_html=True)

# ── the product: five tools, one workflow ───────────────────────────────────
st.subheader("The workflow — five tools, one pipeline")
st.caption("Each tab is a step. Run them in order for a full port, or jump straight to the one you need.")

TOOLS = [
    ("🧬", "1 · Architecture Scan", "repo_scan.py",
     "**Why** it won't map. Scans your model's source and flags each incompatible op with the DPU rule + file:line."),
    ("🔀", "2 · Structure Converter", "structure.py",
     "**What** to change. Before→after restructure + DPU-coverage gain, stored & downloadable."),
    ("✅", "3 · Auto-Fix & Verify", "autofix.py",
     "**Proven** outcome. Applies safe fixes to a copy, re-scans, and *measures* the real coverage gain + patch."),
    ("🏁", "4 · Multi-Vendor", "multivendor.py",
     "**Which** target. Scores AMD DPU vs NVIDIA TensorRT vs Intel OpenVINO and recommends one."),
    ("🔬", "5 · Advisor", "advisor.py",
     "**Full** plan. ONNX op audit + conservative & aggressive INT8 porting proposals."),
]
cols = st.columns(5)
for col, (icon, title, page, desc) in zip(cols, TOOLS):
    with col:
        st.markdown(
            f'<div class="home-card" style="min-height:170px;">'
            f'<div style="font-size:1.5rem;margin-bottom:0.4rem;">{icon}</div>'
            f'<h4>{title}</h4><p>{desc}</p></div>',
            unsafe_allow_html=True,
        )
        st.page_link(page, label="Open →")

st.markdown("<br>", unsafe_allow_html=True)

# ── where we sit in the Vitis AI flow ───────────────────────────────────────
st.subheader("Where we sit in the Vitis AI flow")
st.caption("We're **upstream** of AMD's toolchain — analysis & refactor advice — not duplicating the Optimizer/Quantizer/Compiler.")

def _box(title, sub, kind):
    if kind == "us":
        style = ("background:linear-gradient(135deg,rgba(37,99,235,0.22),rgba(30,64,175,0.22));"
                 "border:1px solid #2563eb;box-shadow:0 0 16px rgba(37,99,235,0.35);")
        tcolor = "#bfdbfe"
    elif kind == "amd":
        style = "background:rgba(13,148,136,0.10);border:1px solid rgba(13,148,136,0.35);"
        tcolor = "#5eead4"
    else:
        style = "background:rgba(15,23,42,0.55);border:1px solid rgba(51,65,85,0.45);"
        tcolor = "#e2e8f0"
    return (f'<div style="{style}border-radius:10px;padding:0.55rem 0.7rem;min-width:118px;text-align:center;">'
            f'<div style="font-size:0.8rem;font-weight:700;color:{tcolor};">{title}</div>'
            f'<div style="font-size:0.66rem;color:rgba(148,163,184,0.85);margin-top:2px;">{sub}</div></div>')

_arrow = '<div style="display:flex;align-items:center;color:rgba(100,116,139,0.7);font-size:1.1rem;">→</div>'
flow = _arrow.join([
    _box("Your model", "PyTorch / ONNX", "in"),
    _box("⭐ Porting Advisor", "analyse + refactor (us)", "us"),
    _box("Optimizer", "prune", "amd"),
    _box("Quantizer", "float → INT8", "amd"),
    _box("Compiler", "→ .xmodel", "amd"),
    _box("Deploy", "DPU · VART", "in"),
])
st.markdown(
    f'<div style="display:flex;flex-wrap:wrap;align-items:stretch;gap:0.4rem;margin:0.25rem 0 0.6rem;">{flow}</div>',
    unsafe_allow_html=True)
st.markdown(
    '<p style="font-size:0.74rem;color:rgba(100,116,139,0.85);margin:0 0 0.5rem;">'
    '<span style="color:#5eead4;">●</span> Optimizer · Quantizer · Compiler = AMD Vitis AI toolchain &nbsp;·&nbsp; '
    '<span style="color:#93c5fd;">●</span> we catch incompatibilities <em>before</em> the model ever enters it.</p>',
    unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# ── why it's different (the value story) ────────────────────────────────────
st.subheader("Why it's different")
v1, v2, v3, v4 = st.columns(4)
with v1:
    st.markdown("""
    <div class="home-card"><div style="font-size:1.4rem;margin-bottom:0.3rem;">🎯</div>
      <h4>Analyzer → outcome</h4>
      <p>We don't just diagnose — we apply the fix and re-measure. <strong>Estimated vs verified</strong>
      coverage, side by side. Proof, not a promise.</p></div>
    """, unsafe_allow_html=True)
with v2:
    st.markdown("""
    <div class="home-card"><div style="font-size:1.4rem;margin-bottom:0.3rem;">🏁</div>
      <h4>Vendor-neutral</h4>
      <p>The only view that compares <strong>AMD · NVIDIA · Intel</strong> for <em>your</em> model.
      No single vendor will ever tell you to use a competitor.</p></div>
    """, unsafe_allow_html=True)
with v3:
    st.markdown("""
    <div class="home-card"><div style="font-size:1.4rem;margin-bottom:0.3rem;">🔒</div>
      <h4>IP-safe &amp; local</h4>
      <p>Runs locally — your model never leaves. Source scan &amp; capture work <strong>offline with
      no key</strong>; bring your own LLM key only for the AI explanations.</p></div>
    """, unsafe_allow_html=True)
with v4:
    st.markdown("""
    <div class="home-card"><div style="font-size:1.4rem;margin-bottom:0.3rem;">⚡</div>
      <h4>Upstream of the toolchain</h4>
      <p>Catches incompatibilities <strong>before</strong> quantize/compile — killing the
      trial-and-error loop teams burn weeks on with vendor tools.</p></div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── under the hood ──────────────────────────────────────────────────────────
st.subheader("Under the hood")

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
st.info("Start the demo on **🧬 Architecture Scan** — it runs offline on the bundled sample with no key.", icon="▶️")
st.page_link("repo_scan.py", label="Start here  →", icon="🧬")
