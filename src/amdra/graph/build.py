"""Assemble the LangGraph workflow.

Two static wirings, picked at build time by `Settings.investigator` (never a runtime choice):

    "fixed" (default):
      intake -> gather_transactions -> gather_documents -> classify_injection -> retrieve_policy -> decide -> verify
      gather_documents --(any receipt below ocr_confidence_threshold)--> vision_fallback -> classify_injection
                       \\--(otherwise)-----------------------------------------------------> classify_injection

    "react" (M3c, opt-in ablation — see nodes.py investigate()):
      intake -> investigate -> classify_injection -> retrieve_policy -> decide -> verify

classify_injection (M4c) is an unconditional second opinion on every untrusted evidence item —
not gated on whether the regex scanner already flagged something, and identical in both wirings.

Both converge on the same tail, unaffected by which pipeline gathered the evidence:
    verify --(failed, retries left)--> decide
    verify --> review_gate --(needs review)--> human_review -> finalize
                           \\--(auto)-------------------------> finalize
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langgraph.graph import END, START, StateGraph

from amdra.config import Settings
from amdra.graph.nodes import Nodes
from amdra.graph.state import DisputeState
from amdra.llm import InjectionClassifier, Reasoner, VisionReasoner, make_reasoner
from amdra.retrieval.store import PolicyIndex
from amdra.schemas import Dispute
from amdra.tools.toolbox import Toolbox


def _checkpoint_serde():
    """A JsonPlusSerializer that explicitly allows AMDRA's own Pydantic/Enum types during
    checkpoint deserialization. Without this, resuming a paused run (`interrupt_for_review`,
    exercised by `cmd_run --review` and the M5 UI) logs "Deserializing unregistered type ...
    will be blocked in a future version" for every one of them — harmless today, a hard failure
    once langgraph defaults LANGGRAPH_STRICT_MSGPACK to true. Explicit and enumerated, not a
    blanket bypass (`allowed_msgpack_modules=True`), consistent with this project's
    least-privilege posture elsewhere (NODE_SCOPES, @authorized)."""
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

    from amdra.schemas import (
        Account,
        AuditEvent,
        Citation,
        ClassifierResult,
        Evidence,
        LabeledCase,
        Outcome,
        ReasonCode,
        Receipt,
        Recommendation,
        Transaction,
        VisionResult,
    )

    return JsonPlusSerializer(allowed_msgpack_modules=[
        Account, AuditEvent, Citation, ClassifierResult, Dispute, Evidence, LabeledCase,
        Outcome, ReasonCode, Receipt, Recommendation, Transaction, VisionResult,
    ])


def memory_checkpointer():
    """MemorySaver configured to deserialize AMDRA's own state types without warnings — use this
    instead of a bare MemorySaver() wherever a paused (interrupt_for_review) run may be resumed."""
    from langgraph.checkpoint.memory import MemorySaver

    return MemorySaver(serde=_checkpoint_serde())


def sqlite_checkpointer(conn_string: str):
    """SqliteSaver configured the same way — a drop-in replacement for
    `SqliteSaver.from_conn_string(conn_string)` (M3b) with AMDRA's types allowlisted. Still a
    context manager, same usage: `with sqlite_checkpointer(path) as checkpointer: ...`."""
    import sqlite3
    from contextlib import closing, contextmanager

    from langgraph.checkpoint.sqlite import SqliteSaver

    @contextmanager
    def _open():
        with closing(sqlite3.connect(conn_string, check_same_thread=False)) as conn:
            yield SqliteSaver(conn, serde=_checkpoint_serde())

    return _open()


def build_graph(nodes: Nodes, checkpointer: Any = None, interrupt_for_review: bool = False):
    react = nodes.settings.investigator == "react"
    g = StateGraph(DisputeState)
    gather_names = ["investigate"] if react else ["gather_transactions", "gather_documents",
                                                   "vision_fallback"]
    for name in ["intake", *gather_names, "classify_injection", "retrieve_policy", "decide",
                 "verify", "review_gate", "human_review", "finalize"]:
        g.add_node(name, getattr(nodes, name))

    g.add_edge(START, "intake")
    if react:
        g.add_edge("intake", "investigate")
        g.add_edge("investigate", "classify_injection")
    else:
        g.add_edge("intake", "gather_transactions")
        g.add_edge("gather_transactions", "gather_documents")
        g.add_conditional_edges("gather_documents", nodes.route_after_documents,
                                {"vision_fallback": "vision_fallback",
                                 "retrieve_policy": "classify_injection"})
        g.add_edge("vision_fallback", "classify_injection")
    g.add_edge("classify_injection", "retrieve_policy")
    g.add_edge("retrieve_policy", "decide")
    g.add_edge("decide", "verify")
    g.add_conditional_edges("verify", nodes.route_after_verify,
                            {"decide": "decide", "review_gate": "review_gate"})
    g.add_conditional_edges("review_gate", nodes.route_after_gate,
                            {"human_review": "human_review", "finalize": "finalize"})
    g.add_edge("human_review", "finalize")
    g.add_edge("finalize", END)

    return g.compile(
        checkpointer=checkpointer,
        interrupt_before=["human_review"] if interrupt_for_review else None,
    )


@dataclass
class Agent:
    """Convenience wrapper that wires settings, tools, retrieval, reasoner, and graph."""

    settings: Settings
    toolbox: Toolbox
    reasoner: Reasoner
    graph: Any

    @classmethod
    def create(cls, settings: Settings | None = None, reasoner: Reasoner | None = None,
               vision_reasoner: VisionReasoner | None = None, investigator_model: Any = None,
               injection_classifier: InjectionClassifier | None = None, checkpointer: Any = None,
               interrupt_for_review: bool = False) -> Agent:
        settings = settings or Settings()
        index = PolicyIndex.from_settings(settings)
        toolbox = Toolbox(settings, index)
        reasoner = reasoner or make_reasoner(settings)
        nodes = Nodes(settings, toolbox, reasoner, vision_reasoner, investigator_model,
                      injection_classifier)
        return cls(settings, toolbox, reasoner,
                   build_graph(nodes, checkpointer, interrupt_for_review))

    def run(self, dispute: Dispute) -> dict:
        config = {"configurable": {"thread_id": dispute.dispute_id}}
        return self.graph.invoke({"dispute": dispute}, config=config)
