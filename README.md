# Vitis AI Porting Advisor

An AI-powered pipeline that analyses deep learning models for Xilinx DPU hardware compatibility and generates two concrete refactoring proposals — one conservative, one aggressive INT8 — to maximise DPU subgraph coverage under Vitis AI 3.x.

---

## Integrations

- **Butterbase**: AI Model Gateway (`https://api.butterbase.ai/v1`) for all LLM calls + REST database for persistent run history (`save_run` / `list_runs`)
- **XTrace**: Memory API used by agents to write and read persistent analysis history
- **Photon**: Agent delivered via `photon_send.mjs` messaging integration

---

## What It Does

1. **Upload** a YOLOv8 `.onnx` or `.pt` model
2. **Select** a target DPU architecture (12 supported)
3. **Run** a 4-stage AI pipeline
4. **Get** a hardware audit report + two AI-generated porting strategies

---

## Architecture

```
Model file (.onnx / .pt)
        │
        ▼
┌─────────────────────┐
│  Agent 1            │  ONNX graph traversal → op list with names/types/shapes
│  Model Extractor    │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Agent 2            │  ChromaDB RAG (93 entries, 12 DPU archs) + static
│  RAG Compatibility  │  partial-support map → severity: ok / warning / critical
└────────┬────────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌────────┐ ┌────────┐
│Agent 3A│ │Agent 3B│  Parallel LLM calls via Butterbase → Claude Sonnet 4.6
│Conserv.│ │Aggress.│
└────────┘ └────────┘
         │
         ▼
  Streamlit UI  +  Butterbase run history
```

---

## Pipeline Stages

| Stage | Agent | Output |
|-------|-------|--------|
| 1 | `agent_one_modelExtractor.py` | Op list: name, type, input/output shapes |
| 2 | `agent_two_RAG.py` | Compatibility report: severity per op, optimization hints |
| 3A | `agent_three_proposal_a.py` | Conservative audit: minimal code changes, CPU residual accepted |
| 3B | `agent_three_proposal_b.py` | Aggressive plan: full INT8, DPU subgraph map, FPS/power estimates |

---

## Supported DPU Architectures

| Board | DPU Arch |
|-------|----------|
| KV260 / ZCU102 / ZCU106 / Kria K26 | DPUCZDX8G B4096 |
| Custom Zynq UltraScale+ | DPUCZDX8G B3136 |
| ZCU104 | DPUCZDX8G B2304 |
| Ultra96-v2 | DPUCZDX8G B1600 |
| Entry Zynq UltraScale+ | DPUCZDX8G B1152 / B1024 / B800 / B512 |
| VCK190 Versal AI Core | DPUCVDX8G |
| VEK280 Versal AI Edge + HBM | DPUCVDX8H |
| Alveo U50 / U200 / U250 / U280 | DPUCAHX8H |
| Alveo U55C / V70 | DPUCADF8H |

---

## Setup

### 1. Install dependencies

```bash
pip install streamlit onnx torch torchvision openai \
            langchain-huggingface langchain-chroma \
            chromadb sentence-transformers pandas
```

### 2. Seed the RAG database

```bash
python setup_vitis_rag.py
```

Populates `./vitis_docs_db/` with 93 Vitis AI operator compatibility entries covering all 12 DPU architectures and 30+ op types.

### 3. Run the app

```bash
streamlit run app.py
```

### 4. CLI usage (no UI)

```bash
python -m agents.wire yolov8m.onnx
# Results saved to analysis_result.json
```

---

## Integrations

### Butterbase
- **AI Gateway**: All LLM calls route through `https://api.butterbase.ai/v1` (OpenAI-compatible), model `anthropic/claude-sonnet-4.6`
- **Database**: `butterbase.py` — `save_run()` / `list_runs()` for persistent run history shown in the sidebar

---

## Key Improvements

### Streamlit UI (`app.py`)
- **Neo-Tactile dark glassmorphism theme** — custom CSS with glass cards, glowing status dots, animated components, no native Streamlit element overrides (theme variables respected)
- **Run button spinner** — `st.session_state` + `st.rerun()` pattern; button shows animated "Running…" state during pipeline execution
- **Live CLI log panel** — right-hand column streams pipeline stage updates in real time with colour-coded log lines (`log-ok`, `log-warn`, `log-err`, `log-info`)
- **Persistent sidebar** — collapse/expand buttons hidden via CSS; sidebar always visible with Configuration + Run History sections
- **Top-right Streamlit menu preserved** — Rerun / Clear cache / Record screen accessible via native `⋮` menu
- **Run history** — last 8 runs fetched from Butterbase and shown in the sidebar with op count, unsupported badge, and timestamp
- **Target DPU selector** — 12 architecture options with board names and constraint annotations (channel width, kernel limits)

### RAG Knowledge Base (`setup_vitis_rag.py`)
- Expanded from 5 → **93 entries** across 12 DPU architectures
- Full per-arch constraint detail for 30+ op types: Conv, MaxPool, AveragePool, GlobalAveragePool, AdaptiveAvgPool, Sigmoid, SiLU, Softmax, Resize, Split, Slice, Concat, BatchNorm, Relu, LeakyRelu, Clip, HardSwish, Add, Mul, Sub, Div, Reshape, Transpose, Flatten, Gemm, Pad, ConvTranspose, NMS, TopK, LSTM, LayerNorm, Gather, Sqrt
- Board-to-arch mappings: KV260/ZCU102/ZCU106 → B4096, ZCU104 → B2304, Ultra96-v2 → B1600, VCK190 → DPUCVDX8G, VEK280 → DPUCVDX8H, Alveo U50–U280 → DPUCAHX8H, Alveo U55C/V70 → DPUCADF8H

### Agent 2 — RAG Compatibility (`agent_two_RAG.py`)
- Architecture-specific vector DB queries (`"Is {op_type} supported on {target_arch}?"`)
- Deduplication: hits DB once per unique op type, not once per op instance
- Prefers arch-matching docs, falls back to "ALL" arch docs
- Static `_PARTIAL_SUPPORT` override map for ops that cause DPU↔CPU transitions (Sigmoid, Reshape, Transpose, Slice, Div, Sub, Sqrt, etc.)
- Per-op optimization hints for 15+ op types

### Agent 3A — Conservative Proposal (`agent_three_proposal_a.py`)
- `target_arch` parameter passed through from UI selection
- `max_tokens` raised from 2048 → **4096** (previous value caused JSON truncation on large models)
- Robust JSON extraction: finds first `{` / last `}` in raw response, tolerates any LLM preamble or markdown fences

### Agent 3B — Aggressive Deployment Plan (`agent_three_proposal_b.py`)
- `target_arch` parameter passed through from UI selection
- Robust JSON extraction: same `{`/`}` approach — no longer fails when LLM adds a sentence before the JSON
- DPU subgraph partition, INT8 calibration pipeline, FPS/power/memory estimates in output

### Bug Fixes

| Bug | Fix |
|-----|-----|
| `"Failed to parse response"` on audit and deployment plan | Replace `startswith("```")` fence stripper with `raw[raw.find("{"):raw.rfind("}")+1]` — handles any preamble |
| Agent A JSON truncation on large models | `max_tokens` 2048 → 4096 |
| Sidebar disappearing | Root cause: `header[data-testid="stHeader"] { display:none }` breaks sidebar anchor; changed to `background: transparent` |
| Streamlit toolbar hidden | Removed `[data-testid="stToolbar"] { display:none }` rule |

---

## Project Structure

```
├── app.py                          # Streamlit UI (Neo-Tactile dark theme)
├── butterbase.py                   # Butterbase client (save_run / list_runs)
├── setup_vitis_rag.py              # Seeds ChromaDB with 93 Vitis AI op entries
├── photon_send.mjs                 # Photon messaging integration
├── agents/
│   ├── wire.py                     # 4-stage pipeline orchestrator
│   ├── agent_one_modelExtractor.py # ONNX/PT op extraction
│   ├── agent_two_RAG.py            # RAG compatibility scan
│   ├── agent_three_proposal_a.py   # Conservative porting strategy (LLM)
│   └── agent_three_proposal_b.py   # Aggressive INT8 deployment plan (LLM)
├── .streamlit/
│   └── config.toml                 # Dark theme + server config
└── model_input/                    # Sample models for testing
```
