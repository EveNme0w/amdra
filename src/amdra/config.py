"""Runtime configuration, read from environment variables (and .env if python-dotenv is installed)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:  # optional
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

REPO_ROOT = Path(__file__).resolve().parents[2]

# USD per million tokens. Verify against current published pricing before trusting cost reports.
PRICING_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-sonnet-4-5": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
    "offline": (0.0, 0.0),
}


@dataclass
class Settings:
    llm: str = field(default_factory=lambda: os.getenv("AMDRA_LLM", "anthropic"))
    model: str = field(default_factory=lambda: os.getenv("AMDRA_MODEL", "claude-sonnet-4-5"))
    vector_backend: str = field(
        default_factory=lambda: os.getenv("AMDRA_VECTOR_BACKEND", "chroma")
    )
    embedder: str = field(default_factory=lambda: os.getenv("AMDRA_EMBEDDER", "chroma-default"))
    hybrid_retrieval: bool = field(
        default_factory=lambda: os.getenv("AMDRA_HYBRID_RETRIEVAL", "").lower() in ("1", "true", "yes")
    )
    ocr: str = field(default_factory=lambda: os.getenv("AMDRA_OCR", "auto"))
    # Below this, OCR text is unreliable enough that a fact derived from it shouldn't be trusted.
    # Drives gather_documents' routing to vision_fallback (M2d).
    ocr_confidence_threshold: float = 0.35
    # "fixed": today's deterministic gather_transactions/gather_documents/vision_fallback pipeline.
    # "react": M3c's opt-in ReAct investigator subgraph — a comparable, not a replacement; ablation
    # only, never the default, so tests and --offline stay on the deterministic, zero-cost path.
    investigator: str = field(default_factory=lambda: os.getenv("AMDRA_INVESTIGATOR", "fixed"))
    investigator_max_steps: int = 8
    data_dir: Path = field(
        default_factory=lambda: Path(os.getenv("AMDRA_DATA_DIR", REPO_ROOT / "data" / "synthetic"))
    )
    chroma_dir: Path = field(
        default_factory=lambda: Path(os.getenv("AMDRA_CHROMA_DIR", REPO_ROOT / ".chroma"))
    )
    retrieval_k: int = 4
    confidence_threshold: float = 0.7
    max_decide_attempts: int = 2

    @property
    def db_path(self) -> Path:
        return self.data_dir / "bank.db"

    @property
    def policies_dir(self) -> Path:
        return self.data_dir / "policies"

    @property
    def receipts_dir(self) -> Path:
        return self.data_dir / "receipts"

    @property
    def cases_path(self) -> Path:
        return self.data_dir / "cases.jsonl"


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    in_rate, out_rate = PRICING_PER_MTOK.get(model, (0.0, 0.0))
    return (input_tokens * in_rate + output_tokens * out_rate) / 1_000_000
