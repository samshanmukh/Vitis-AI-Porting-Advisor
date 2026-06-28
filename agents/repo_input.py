"""Shared sidebar repo-source widget used by every scan tab.

Rules (consistent across Architecture Scan, Structure Converter, Auto-Fix,
Multi-Vendor):
  • A real GitHub URL / local path ALWAYS wins over the bundled sample — the
    offline sample is only used when no real source is provided.
  • The last real source is persisted (session + disk) and pre-filled on the
    next visit and on the other tabs.
"""
import os

import streamlit as st

PLACEHOLDER = "https://github.com/owner/model-repo"
SAMPLE_PATH = "model_input/Transfer-Model_original"

_STATE_FILE = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "scan_memory", "last_repo.txt"
)


def _load_saved() -> str:
    try:
        with open(_STATE_FILE, encoding="utf-8") as fh:
            return fh.read().strip()
    except OSError:
        return ""


def _save(url: str) -> None:
    try:
        os.makedirs(os.path.dirname(_STATE_FILE), exist_ok=True)
        with open(_STATE_FILE, "w", encoding="utf-8") as fh:
            fh.write((url or "").strip())
    except OSError:
        pass


def is_real_source(s: str) -> bool:
    """True when the user supplied an actual repo (not blank / not the hint)."""
    s = (s or "").strip()
    return bool(s) and s != PLACEHOLDER


def repo_source_input(label: str = "GitHub repo URL (or local path)") -> str:
    """Render the repo URL field + sample toggle and return the path to resolve.

    Call this inside a `with st.sidebar:` block. The returned value is what to
    hand to `resolve_repo()`.
    """
    # Seed once from disk; the shared key keeps it in sync across tabs after that.
    if "repo_source" not in st.session_state:
        st.session_state["repo_source"] = _load_saved() or PLACEHOLDER

    source = st.text_input(label, key="repo_source")
    has_url = is_real_source(source)

    use_sample = st.checkbox(
        "Use bundled sample repo (offline)",
        value=not has_url,
        disabled=has_url,
        help="Ignored when a GitHub URL / local path is provided above — the "
             "repo you enter always takes precedence over the sample.",
    )

    if has_url:
        _save(source)            # remember it for next time / other tabs
        return source            # a real source always wins
    if use_sample:
        return SAMPLE_PATH
    return source                # blank/placeholder → resolve_repo reports the error
