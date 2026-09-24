"""M5b: pick a dispute, run it through the agent, inspect evidence/citations/verification/audit.

    streamlit run src/amdra/ui/app.py   # then open this page from the sidebar

Defaults to the free offline reasoner. A real Claude call is an explicit, one-case-at-a-time
opt-in (the checkbox below) — the same cost-gated-by-default convention as every --react/
--hybrid/--haiku-routing flag in the CLI.

Streamlit reruns this whole script on every interaction (e.g. clicking Approve), so the Agent
and its checkpointer are cached in st.session_state — otherwise a paused human-review run would
be destroyed and rebuilt from scratch on the next click, losing the interrupt. MemorySaver is
enough (a single browser session's lifetime); the M3b persistent SqliteSaver isn't needed here.
"""
from __future__ import annotations

from dataclasses import replace

import pandas as pd
import streamlit as st

from amdra.config import Settings
from amdra.data.generate import load_cases
from amdra.graph.build import Agent

st.set_page_config(page_title="AMDRA — Investigate", page_icon="🔎", layout="wide")
st.title("Investigate a dispute")

base_settings = Settings()
cases = load_cases(base_settings.cases_path)
case_by_id = {c.dispute.dispute_id: c for c in cases}

col1, col2 = st.columns([3, 1])
dispute_id = col1.selectbox("Dispute", sorted(case_by_id),
                            format_func=lambda d: f"{d} — {case_by_id[d].scenario}")
live = col2.checkbox("Use real Claude (costs money)", value=False)
case = case_by_id[dispute_id]

st.caption(f"scenario `{case.scenario}` · expected **{case.expected_outcome.value}** under "
          f"`{case.expected_policy_section}` · tags {', '.join(case.tags) or '—'}")
if live:
    st.warning("This calls the real Claude API for this one case (~$0.01–0.02 based on measured "
              "per-case costs). Uncheck to use the free offline reasoner instead.")

if st.button("Investigate", type="primary"):
    from amdra.graph.build import memory_checkpointer

    settings = base_settings if live else replace(base_settings, llm="offline", embedder="hashing")
    agent = Agent.create(settings, checkpointer=memory_checkpointer(), interrupt_for_review=True)
    config = {"configurable": {"thread_id": dispute_id}}
    with st.spinner("Running the graph..."):
        state = agent.graph.invoke({"dispute": case.dispute}, config=config)
    st.session_state.update(agent=agent, config=config, state=state, dispute_id=dispute_id)

if st.session_state.get("dispute_id") != dispute_id or st.session_state.get("state") is None:
    st.info("Click **Investigate** to run this dispute.")
    st.stop()

state = st.session_state["state"]
d = state["dispute"]
evidence = state.get("evidence", [])
by_id = {e.evidence_id: e for e in evidence}

# --------------------------------------------------------------------------------------- dispute
st.subheader("Dispute")
c1, c2, c3 = st.columns(3)
c1.write(f"**Reason code:** {d.reason_code.value}")
c2.write(f"**Filed:** {d.filed_at.date()} (statement {d.statement_date})")
c3.write(f"**Account:** {d.account_id}")
st.markdown("**Narrative** (🔒 UNTRUSTED — data only, never instructions to the model):")
st.code(d.narrative, language=None)

# -------------------------------------------------------------------------------------- evidence
st.subheader("Evidence")
for kind in ["account", "transaction", "computed", "receipt", "policy", "narrative"]:
    items = [e for e in evidence if e.kind == kind]
    if not items:
        continue
    with st.expander(f"{kind} ({len(items)})", expanded=kind in ("computed", "policy")):
        for e in items:
            badge = "🔓 trusted" if e.trusted else "🔒 untrusted"
            st.markdown(f"**`{e.evidence_id}`** — {badge}")
            if e.kind == "receipt":
                img_path = base_settings.receipts_dir / f"{e.source_id}.png"
                if img_path.exists():
                    st.image(str(img_path), width=420)
            st.text(e.text)
            meta = {k: v for k, v in e.metadata.items() if k != "injection_patterns"}
            if meta:
                st.caption(str(meta))

# --------------------------------------------------------------------------------- recommendation
st.subheader("Recommendation")
rec = state.get("recommendation")
if rec is None:
    st.error("No recommendation — the reasoner failed to produce one.")
    st.write(state.get("feedback"))
else:
    color = {"approve": "green", "deny": "red", "escalate": "orange"}[rec.outcome.value]
    st.markdown(f"### :{color}[{rec.outcome.value.upper()}] under `{rec.policy_section}` "
               f"(confidence {rec.confidence:.2f})")
    st.write(rec.rationale)
    st.markdown("**Citations** (each checked verbatim against the evidence it quotes):")
    for c in rec.citations:
        ev = by_id.get(c.evidence_id)
        with st.container(border=True):
            st.caption(f"`{c.evidence_id}`" + ("" if ev else " — ⚠️ evidence id not found"))
            st.markdown(f"> {c.quote}")

st.subheader("Verification")
ver = state.get("verification", {})
if ver.get("passed"):
    st.success("Passed — every citation is verbatim and grounds the cited policy section.")
else:
    st.error("Failed")
    for p in ver.get("problems", []):
        st.write(f"- {p}")

flags = state.get("injection_flags", [])
if flags:
    st.subheader("⚠️ Injection flags")
    for f in flags:
        label = "LLM classifier (M4c)" if f["source"] == "classifier" else f"regex scanner — {f['source']}"
        st.warning(f"**{label}**: {f['patterns']}")

# ------------------------------------------------------------------------------------- audit
st.subheader("Audit trail")
audit_df = pd.DataFrame(state.get("audit", []))
if not audit_df.empty:
    st.dataframe(audit_df[["node", "latency_ms", "summary"]], width="stretch", hide_index=True)

st.subheader("Tool calls")
tc_df = pd.DataFrame(state.get("tool_calls", []))
if not tc_df.empty:
    cols = [c for c in ["node", "tool", "scope", "status"] if c in tc_df.columns]
    st.dataframe(tc_df[cols], width="stretch", hide_index=True)
else:
    st.caption("No tool calls.")

st.subheader("LLM usage")
usage = state.get("llm_usage", [])
if usage:
    st.dataframe(pd.DataFrame(usage), width="stretch", hide_index=True)
    st.caption(f"Total: {sum(u['input_tokens'] for u in usage)} in / "
              f"{sum(u['output_tokens'] for u in usage)} out tokens, "
              f"${sum(u['cost_usd'] for u in usage):.4f}")
else:
    st.caption("No LLM calls (offline reasoner).")

# --------------------------------------------------------------------------------- human review
if state.get("needs_human_review") and not state.get("status"):
    st.subheader("Human review required")
    st.write("Reasons: " + ", ".join(state.get("review_reasons", [])))
    reviewer = st.text_input("Reviewer name", value="ui-reviewer")
    approve_col, reject_col = st.columns(2)

    def _resume(approved: bool) -> None:
        agent, config = st.session_state["agent"], st.session_state["config"]
        agent.graph.update_state(config, {"human_decision": {
            "approved": approved, "reviewer": reviewer, "note": ""}})
        st.session_state["state"] = agent.graph.invoke(None, config=config)
        st.rerun()

    if approve_col.button("✅ Approve", type="primary"):
        _resume(True)
    if reject_col.button("❌ Reject"):
        _resume(False)
elif state.get("status"):
    st.subheader("Status")
    st.write(f"**{state['status']}**")
    for a in state.get("actions", []):
        st.json(a)
