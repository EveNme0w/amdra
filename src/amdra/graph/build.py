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
