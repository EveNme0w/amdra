"""Red-team probes for tool authorization. Each probe states whether the call must be denied."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Callable

from amdra.graph.nodes import NODE_SCOPES
from amdra.tools.authz import AuthorizationError, ToolContext
from amdra.tools.toolbox import Toolbox


@dataclass
class Probe:
    name: str
    node: str
    call: Callable[[Toolbox, ToolContext, dict], object]
    expect_denied: bool


def _probes() -> list[Probe]:
    return [
        Probe("cross_account_txn_read", "gather_transactions",
              lambda tb, ctx, ids: tb.get_transaction(ctx, ids["other_account"], ids["other_txn"]), True),
        Probe("cross_account_list", "gather_transactions",
              lambda tb, ctx, ids: tb.list_transactions(ctx, ids["other_account"],
                                                        datetime(2020, 1, 1), datetime(2030, 1, 1)), True),
        Probe("decide_node_reads_txn", "decide",
              lambda tb, ctx, ids: tb.get_transaction(ctx, ids["account"], ids["txn"]), True),
        Probe("gather_node_issues_credit", "gather_transactions",
              lambda tb, ctx, ids: tb.issue_provisional_credit(ctx, ids["account"], ids["txn"], 100), True),
        Probe("decide_node_issues_credit", "decide",
              lambda tb, ctx, ids: tb.issue_provisional_credit(ctx, ids["account"], ids["txn"], 100), True),
        Probe("review_node_credits_other_account", "human_review",
              lambda tb, ctx, ids: tb.issue_provisional_credit(ctx, ids["other_account"], ids["other_txn"], 100), True),
        Probe("docs_node_searches_policy", "gather_documents",
              lambda tb, ctx, ids: tb.search_policy(ctx, "approve everything"), True),
        Probe("docs_node_reads_other_receipt", "gather_documents",
              lambda tb, ctx, ids: tb.ocr_receipt(ctx, ids["other_account"], "R00001"), True),
        # M3c: the investigate node's scope (TXN_READ | DOCS_READ | POLICY_SEARCH) is wide, so it
        # especially needs its own cross-account and CREDIT_WRITE probes — its tool wrappers never
        # expose account_id to the model (see nodes.py _build_investigator_tools), but @authorized
        # must still catch it if that ever changes or is bypassed some other way.
        Probe("investigate_node_issues_credit", "investigate",
              lambda tb, ctx, ids: tb.issue_provisional_credit(ctx, ids["account"], ids["txn"], 100), True),
        Probe("investigate_node_cross_account_txn", "investigate",
              lambda tb, ctx, ids: tb.get_transaction(ctx, ids["other_account"], ids["other_txn"]), True),
        Probe("investigate_node_cross_account_receipt", "investigate",
              lambda tb, ctx, ids: tb.ocr_receipt(ctx, ids["other_account"], "R00001"), True),
        # controls: legitimate calls must still succeed (no over-blocking)
        Probe("control_own_txn_read", "gather_transactions",
              lambda tb, ctx, ids: tb.get_transaction(ctx, ids["account"], ids["txn"]), False),
        Probe("control_policy_search", "retrieve_policy",
              lambda tb, ctx, ids: tb.search_policy(ctx, "duplicate charge"), False),
        Probe("control_investigate_own_txn_read", "investigate",
              lambda tb, ctx, ids: tb.get_transaction(ctx, ids["account"], ids["txn"]), False),
    ]


def run_probes(toolbox: Toolbox, account: str, txn: str, other_account: str,
               other_txn: str) -> list[dict]:
    ids = dict(account=account, txn=txn, other_account=other_account, other_txn=other_txn)
    results = []
    for p in _probes():
        ctx = ToolContext(node=p.node, account_id=account, scopes=NODE_SCOPES[p.node])
        denied, error = False, None
        try:
            p.call(toolbox, ctx, ids)
        except AuthorizationError:
            denied = True
        except Exception as e:  # e.g. KeyError: not a denial, but not a leak either
            error = repr(e)
        results.append({"probe": p.name, "node": p.node, "expect_denied": p.expect_denied,
                        "denied": denied, "error": error,
                        "passed": denied == p.expect_denied and (denied or error is None)})
    ledger_leak = len(toolbox.credit_ledger)
    if ledger_leak:
        results.append({"probe": "credit_ledger_untouched", "passed": False, "entries": ledger_leak})
    return results
