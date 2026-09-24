from dataclasses import replace
from datetime import date

import pytest

from amdra.graph.nodes import NODE_SCOPES
from amdra.guardrails import is_suspicious
from amdra.retrieval.bm25 import BM25Index
from amdra.retrieval.store import PolicyIndex, _match
from amdra.tools.authz import AuthorizationError, Scope, ToolContext
from amdra.tools.toolbox import Toolbox


def ctx_for(node, account):
    return ToolContext(node=node, account_id=account, scopes=NODE_SCOPES[node])


def test_generator_covers_every_scenario(cases):
    assert len(cases) == 66
    assert {c.expected_outcome.value for c in cases} == {"approve", "deny", "escalate"}
    assert sum("injection" in c.tags for c in cases) == 18
    assert sum("ocr_noise" in c.tags for c in cases) == 12
    assert sum("injection_adversarial" in c.tags for c in cases) == 12


def test_adversarial_narratives_evade_the_regex_scanner(cases):
    """M4b: these scenarios only measure anything if they genuinely evade guardrails.scan() —
    if a future regex change accidentally started catching them, the M4b/M4c comparison would
    silently stop meaning anything without this guard."""
    adversarial_narrative_scenarios = {
        "injection_narrative_obfuscated", "injection_narrative_homoglyph",
        "injection_narrative_multilingual",
    }
    cases_checked = [c for c in cases if c.scenario in adversarial_narrative_scenarios]
    assert len(cases_checked) == 9
    for c in cases_checked:
        assert not is_suspicious(c.dispute.narrative), c.scenario


def test_ocr_reads_receipt_total(settings, cases):
    tb = Toolbox(settings)
    case = next(c for c in cases if c.scenario == "amount_lower_receipt")
    r = tb.ocr_receipt(ctx_for("gather_documents", case.dispute.account_id),
                       case.dispute.account_id, case.dispute.receipt_ids[0])
    assert r["total_cents"] is not None and r["total_cents"] > 0
    assert r["ocr_confidence"] > 0.9  # clean receipt: high confidence


def test_mild_noise_keeps_total_readable_but_lowers_confidence(settings, cases):
    tb = Toolbox(settings)
    case = next(c for c in cases if c.scenario == "amount_lower_receipt_noisy_mild")
    r = tb.ocr_receipt(ctx_for("gather_documents", case.dispute.account_id),
                       case.dispute.account_id, case.dispute.receipt_ids[0])
    assert r["total_cents"] is not None and r["total_cents"] > 0
    assert r["ocr_confidence"] < 0.9


def test_severe_noise_breaks_extraction_and_forces_escalation(settings, cases):
    tb = Toolbox(settings)
    case = next(c for c in cases if c.scenario == "amount_matches_noisy_severe")
    r = tb.ocr_receipt(ctx_for("gather_documents", case.dispute.account_id),
                       case.dispute.account_id, case.dispute.receipt_ids[0])
    assert r["total_cents"] is None
    assert r["ocr_confidence"] < settings.ocr_confidence_threshold
    assert case.expected_outcome.value == "escalate"


def test_cross_account_access_denied(settings, cases):
    tb = Toolbox(settings)
    a, b = cases[0].dispute, cases[1].dispute
    ctx = ctx_for("gather_transactions", a.account_id)
    with pytest.raises(AuthorizationError):
        tb.get_transaction(ctx, b.account_id, b.txn_id)
    assert ctx.denials and "outside the case scope" in ctx.denials[0]["reason"]


def test_missing_scope_denied(settings, cases):
    tb = Toolbox(settings)
    d = cases[0].dispute
    with pytest.raises(AuthorizationError):
        tb.issue_provisional_credit(ctx_for("decide", d.account_id), d.account_id, d.txn_id, 1)
    assert tb.credit_ledger == []


def test_no_node_but_human_review_can_write():
    writers = [n for n, s in NODE_SCOPES.items() if Scope.CREDIT_WRITE in s]
    assert writers == ["human_review"]


def test_where_filter_matching():
    meta = {"product": "credit", "effective_from": 20250101, "rc_duplicate_charge": True}
    assert _match(meta, {"$and": [{"product": {"$eq": "credit"}},
                                  {"effective_from": {"$lte": 20260101}}]})
    assert not _match(meta, {"rc_amount_mismatch": {"$eq": True}})


@pytest.mark.parametrize("as_of,version", [(date(2026, 3, 1), "1"), (date(2026, 7, 1), "2")])
def test_retrieval_respects_effective_dates(settings, as_of, version):
    idx = PolicyIndex.from_settings(settings)
    hits = idx.search("filing window deadline", 10, reason_code="duplicate_charge", as_of=as_of)
    windows = [h for h in hits if h["metadata"]["policy_id"] == "POL-001"]
    assert [h["metadata"]["version"] for h in windows] == [version]
    assert all(h["metadata"]["product"] == "credit" for h in hits)  # debit distractor filtered


def test_bm25_index_filters_by_product_and_reason_code(settings):
    idx = BM25Index.from_dir(settings.policies_dir)
    hits = idx.search("provisional credit investigation", 10, product="debit")
    assert hits and {h["metadata"]["policy_id"] for h in hits} == {"POL-900"}
    hits = idx.search("duplicate charge processing error", 10, reason_code="amount_mismatch")
    assert all(h["metadata"].get("rc_amount_mismatch") for h in hits)


def test_hybrid_retrieval_still_finds_governing_sections(settings):
    idx = PolicyIndex.from_settings(replace(settings, hybrid_retrieval=True))
    hits = idx.search("duplicate charge same merchant same amount processing error", 4,
                      reason_code="duplicate_charge", as_of=date(2026, 7, 1))
    citations = {h["metadata"]["citation"] for h in hits}
    assert {"POL-003 §3.1", "POL-003 §3.3"} <= citations
