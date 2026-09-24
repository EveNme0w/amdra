"""AMDRA reviewer UI — entry point (M5).

    streamlit run src/amdra/ui/app.py

Use the sidebar to open a page: Investigate a dispute, or browse an eval run.
"""
from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="AMDRA", page_icon="🧾", layout="wide")
st.title("AMDRA — Auditable Multimodal Dispute Resolution Agent")
st.markdown(
    """
Use the sidebar to open a page:

- **Investigate** — run one dispute through the agent and inspect its evidence, citations,
  verification, and audit trail. Defaults to the free offline reasoner; a real Claude call is an
  explicit, one-case-at-a-time opt-in.
- **Eval Dashboard** — browse a saved `amdra eval` run: accuracy, cost, injection detection, and
  the authorization red-team probes.
"""
)
