"""Graph state. Append-only channels (evidence, audit, tool_calls, usage) use reducers so the
full investigation history is preserved in every checkpoint."""
from __future__ import annotations

import operator
from typing import Annotated, Optional, TypedDict

from amdra.schemas import Account, Dispute, Evidence, Recommendation


def merge_evidence(left: list[Evidence] | None, right: list[Evidence] | None) -> list[Evidence]:
    """Append evidence, replacing items that reuse an evidence_id."""
    merged = {e.evidence_id: e for e in (left or [])}
    for e in right or []:
        merged[e.evidence_id] = e
    return list(merged.values())


class DisputeState(TypedDict, total=False):
    # inputs
    dispute: Dispute
    human_decision: Optional[dict]  # {"approved": bool, "reviewer": str, "note": str}
    # investigation
    account: Account
    evidence: Annotated[list[Evidence], merge_evidence]
    injection_flags: Annotated[list[dict], operator.add]
    # decision
    recommendation: Optional[Recommendation]
    verification: dict
    feedback: list[str]
    attempts: int
    # outputs
    needs_human_review: bool
    review_reasons: list[str]
    status: str
    actions: Annotated[list[dict], operator.add]
    # observability
    audit: Annotated[list[dict], operator.add]
    tool_calls: Annotated[list[dict], operator.add]
    llm_usage: Annotated[list[dict], operator.add]
