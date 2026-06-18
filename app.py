"""Entrypoint — multipage navigation for the Vitis AI Porting Advisor."""
import os
import sys

import streamlit as st

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

st.set_page_config(
    page_title="Vitis AI Porting Advisor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

pages = [
    st.Page("home.py", title="Home", icon="🏠", default=True),
    st.Page("advisor.py", title="Advisor", icon="🔬"),
]

st.navigation(pages).run()
