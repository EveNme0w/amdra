"""M5a: browse a saved `amdra eval` run. Read-only — no live LLM calls, no cost.

    streamlit run src/amdra/ui/app.py   # then open this page from the sidebar
"""
from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from amdra.config import REPO_ROOT

st.set_page_config(page_title="AMDRA — Eval Dashboard", page_icon="📊", layout="wide")
st.title("Eval Dashboard")

RESULTS_DIR = REPO_ROOT / "evals" / "results"
files = sorted(RESULTS_DIR.glob("*.json"), reverse=True)
if not files:
    st.info("No eval results yet. Run `amdra eval --offline` (free) or `amdra eval --limit 14` "
            "first — results are written to `evals/results/`.")
    st.stop()

choice = st.selectbox("Run", [f.stem for f in files])
data = json.loads((RESULTS_DIR / f"{choice}.json").read_text())
meta, summary = data["meta"], data["summary"]
cases_df = pd.DataFrame(data["cases"])
probes_df = pd.DataFrame(data["authz_probes"])

st.caption(f"{meta['timestamp']} · reasoner `{meta['reasoner']}` · vector `{meta['vector_backend']}` "
          f"· embedder `{meta['embedder']}` · {summary['cases']} cases")

# ---------------------------------------------------------------------------------------- KPIs
kpis = [
    ("Decision accuracy", summary["decision_accuracy"], "pct"),
    ("Citation validity", summary["citation_validity"], "pct"),
    ("Injection detection", summary["injection_detection"], "pct"),
    ("Tool authz pass rate", summary["tool_authz_pass_rate"], "pct"),
    ("Cost / case", summary["cost_per_case_usd"], "usd"),
]
for col, (label, value, kind) in zip(st.columns(len(kpis)), kpis):
    if value is None:
        col.metric(label, "—")
    elif kind == "usd":
        col.metric(label, f"${value:.4f}")
    else:
        col.metric(label, f"{value:.1%}")

left, right = st.columns(2)
with left:
    st.subheader("Accuracy by scenario")
    acc = summary["accuracy_by_scenario"]
    st.bar_chart(pd.DataFrame({"accuracy": acc.values()}, index=list(acc.keys())))
with right:
    st.subheader("Confusion (expected → predicted)")
    st.dataframe(
        pd.DataFrame([{"pair": k, "count": v} for k, v in summary["confusion"].items()]),
        width='stretch', hide_index=True,
    )

# ------------------------------------------------------------------------------------- probes
st.subheader("Authorization probes")
if not probes_df.empty:
    n_failed = int((~probes_df["passed"]).sum())
    if n_failed:
        st.error(f"{n_failed} probe(s) failed — {summary['tool_authz_failures']}")
    else:
        st.success(f"All {len(probes_df)} probes passed (including over-blocking controls).")
    st.dataframe(probes_df, width='stretch', hide_index=True)

# -------------------------------------------------------------------------------------- cases
st.subheader("Cases")
f1, f2, f3, f4 = st.columns(4)
scenario_filter = f1.multiselect("Scenario", sorted(cases_df["scenario"].unique()))
all_tags = sorted({t for tags in cases_df["tags"] for t in tags})
tag_filter = f2.multiselect("Tag", all_tags)
only_incorrect = f3.checkbox("Only incorrect")
only_flagged = f4.checkbox("Only injection-flagged")

filtered = cases_df
if scenario_filter:
    filtered = filtered[filtered["scenario"].isin(scenario_filter)]
if tag_filter:
    filtered = filtered[filtered["tags"].apply(lambda ts: any(t in ts for t in tag_filter))]
if only_incorrect:
    filtered = filtered[~filtered["correct"]]
if only_flagged:
    filtered = filtered[filtered["injection_flagged"]]

st.dataframe(
    filtered[["dispute_id", "scenario", "tags", "expected", "predicted", "correct",
             "expected_section", "predicted_section", "section_correct", "injection_flagged",
             "classifier_only_flagged", "needs_human_review", "status", "cost_usd", "latency_s",
             "error"]],
    width='stretch', hide_index=True,
)
