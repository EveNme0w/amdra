"""Prompt-injection defenses for untrusted content (OCR text, cardholder narratives).

Layers:
  1. Detection: regex heuristics flag likely instructions aimed at the model.
  2. Spotlighting: untrusted text is wrapped in tagged blocks with markup neutralized, and the
     system prompt tells the model that such blocks are data, never instructions.
  3. Architecture (enforced elsewhere): tools are called by graph nodes, not chosen by the LLM,
     and the decide node has no tool scopes at all, so injected text cannot trigger tool calls.
  4. Verification: every citation must quote real evidence, and policy must ground the outcome.
"""
from __future__ import annotations

import html
import re

from amdra.schemas import Evidence

INJECTION_PATTERNS: dict[str, re.Pattern] = {
    "ignore_instructions": re.compile(r"\b(ignore|disregard|forget)\b.{0,30}\b(instruction|prompt|rule)s?\b", re.I),
    "role_override": re.compile(r"\b(system\s*(override|prompt|message)|you are now|act as)\b", re.I),
    "addressed_to_model": re.compile(r"\b(note|message)\s+to\s+(the\s+)?(ai|agent|assistant|model|llm)\b", re.I),
    "decision_command": re.compile(r"\b(approve|refund|credit)\b.{0,20}\b(this|the|full|immediately)\b", re.I),
    "tool_request": re.compile(r"\b(look\s*up|call|query|access)\b.{0,40}\baccount\s+[A-Z]\d{3,}", re.I),
}


def scan(text: str) -> list[str]:
    """Return the names of injection patterns found in text."""
    return [name for name, pat in INJECTION_PATTERNS.items() if pat.search(text)]


def is_suspicious(text: str) -> bool:
    # decision_command alone is common in honest narratives ("please refund this"), so require
    # a second signal or a structural override before flagging.
    hits = set(scan(text))
    strong = hits & {"ignore_instructions", "role_override", "addressed_to_model", "tool_request"}
    return bool(strong)


def render_evidence(ev: Evidence) -> str:
    body = html.escape(ev.text, quote=False) if not ev.trusted else ev.text
    trust = "trusted" if ev.trusted else "UNTRUSTED-data-only"
    return f'<evidence id="{ev.evidence_id}" kind="{ev.kind}" trust="{trust}">\n{body}\n</evidence>'
