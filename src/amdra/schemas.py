"""Domain models shared by the data generator, tools, graph, and evals."""
from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field


class ReasonCode(str, Enum):
    FRAUD = "fraud_unauthorized"
    DUPLICATE = "duplicate_charge"
    AMOUNT_MISMATCH = "amount_mismatch"
    NOT_RECEIVED = "goods_not_received"


class Outcome(str, Enum):
    APPROVE = "approve"
    DENY = "deny"
    ESCALATE = "escalate"


class Account(BaseModel):
    account_id: str
    holder_name: str
    product: Literal["credit", "debit"] = "credit"
    home_city: str
    home_country: str
    card_reported_lost_at: Optional[datetime] = None
    known_devices: list[str] = Field(default_factory=list)


class Transaction(BaseModel):
    txn_id: str
    account_id: str
    posted_at: datetime
    amount_cents: int
    currency: str = "USD"
    merchant: str
    mcc: str
    channel: Literal["card_present", "online"]
    city: str
    country: str
    auth_method: Literal["chip_pin", "chip", "swipe", "3ds", "none"]
    device_id: Optional[str] = None


class Receipt(BaseModel):
    receipt_id: str
    account_id: str
    merchant: str
    kind: Literal["receipt", "order_confirmation"] = "receipt"
    issued_on: date
    total_cents: int
    lines: list[str]
    image_path: str
    noise_level: Literal["clean", "mild", "severe"] = "clean"


class Dispute(BaseModel):
    dispute_id: str
    account_id: str
    txn_id: str
    reason_code: ReasonCode
    filed_at: datetime
    statement_date: date
    narrative: str
    receipt_ids: list[str] = Field(default_factory=list)
    merchant_contacted: bool = False
    expected_delivery: Optional[date] = None


class LabeledCase(BaseModel):
    """A dispute plus its ground truth. Labels are set by the scenario template, never by agent code."""

    dispute: Dispute
    scenario: str
    expected_outcome: Outcome
    expected_policy_section: str
    tags: list[str] = Field(default_factory=list)


EvidenceKind = Literal["account", "transaction", "receipt", "policy", "computed", "narrative"]


class Evidence(BaseModel):
    evidence_id: str
    kind: EvidenceKind
    source_id: str
    text: str
    trusted: bool = True
    metadata: dict = Field(default_factory=dict)


class Citation(BaseModel):
    evidence_id: str = Field(description="ID of an evidence item provided in the prompt")
    quote: str = Field(description="Verbatim span copied from that evidence item's text")


class Recommendation(BaseModel):
    outcome: Outcome
    policy_section: str = Field(description="Governing policy section ID, e.g. 'POL-003 §3.1'")
    rationale: str = Field(description="2-4 sentences explaining the decision")
    citations: list[Citation] = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)


class VisionResult(BaseModel):
    """Structured output of the vision-model OCR fallback (graph-triggered, see M2d)."""

    transcribed_text: str = Field(description="Best-effort verbatim transcription of the image")
    total_cents: Optional[int] = Field(default=None, description="TOTAL amount in cents, if legible")
    expected_delivery: Optional[date] = Field(
        default=None, description="Expected-delivery date, if legible and present"
    )
    confidence: float = Field(ge=0.0, le=1.0,
                              description="Confidence that total_cents/expected_delivery are correct")


class AuditEvent(BaseModel):
    at: datetime
    node: str
    event: str
    detail: dict = Field(default_factory=dict)
