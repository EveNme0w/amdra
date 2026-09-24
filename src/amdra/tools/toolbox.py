"""Investigation tools: core-banking lookups (SQLite), receipt OCR, and policy search."""
from __future__ import annotations

import base64
import random
import re
import shutil
import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path

from amdra.config import Settings
from amdra.retrieval.store import PolicyIndex
from amdra.schemas import Account, Receipt, Transaction
from amdra.tools.authz import AuthorizationError, Scope, ToolContext, authorized

TOTAL_RE = re.compile(r"TOTAL\s*\$?\s*([\d,]+\.\d{2})", re.I)
DELIVERY_RE = re.compile(r"Expected delivery:\s*(\d{4}-\d{2}-\d{2})", re.I)

# Synthetic confidence for the sidecar (no-tesseract) path, keyed by Receipt.noise_level.
CONFIDENCE_BY_NOISE = {"clean": 0.99, "mild": 0.55, "severe": 0.15}

_LETTERS = "abcdefghijklmnopqrstuvwxyz"


def _corrupt_text(text: str, level: str, seed: str) -> str:
    """Deterministically simulate degraded OCR for the sidecar (no-tesseract) path.

    mild: garbles free-text lines (merchant name, item descriptions) but leaves lines containing
    the TOTAL or delivery-date fields untouched — on a real receipt those tend to be the most
    legible, high-contrast text — so extraction still succeeds; only confidence drops.
    severe: garbles every line, including the digits in the TOTAL/delivery lines, so extraction
    genuinely fails (TOTAL_RE / DELIVERY_RE stop matching).
    """
    if level == "clean":
        return text
    rng = random.Random(seed)
    protect_numeric_lines = level == "mild"
    rate = 0.15 if level == "mild" else 0.35
    out_lines = []
    for line in text.splitlines():
        if protect_numeric_lines and (TOTAL_RE.search(line) or DELIVERY_RE.search(line)):
            out_lines.append(line)
            continue
        chars = list(line)
        for i, ch in enumerate(chars):
            if ch.isalpha() and rng.random() < rate:
                chars[i] = rng.choice(_LETTERS).upper() if ch.isupper() else rng.choice(_LETTERS)
            elif level == "severe" and ch.isdigit() and rng.random() < rate:
                chars[i] = rng.choice("0123456789")
        out_lines.append("".join(chars))
    return "\n".join(out_lines)


def ocr_image(path: Path, mode: str = "auto", noise_level: str = "clean") -> tuple[str, str, float]:
    """Return (text, engine, confidence). Falls back to the generator's sidecar text — optionally
    corrupted to simulate OCR error, see `_corrupt_text` — when tesseract is absent."""
    use_tesseract = mode == "tesseract" or (mode == "auto" and shutil.which("tesseract"))
    if use_tesseract:
        import pytesseract
        from PIL import Image

        img = Image.open(path)
        text = pytesseract.image_to_string(img)
        text = "\n".join(l for l in text.splitlines() if l.strip())
        data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
        confs = [c for c in data["conf"] if isinstance(c, (int, float)) and c >= 0]
        confidence = (sum(confs) / len(confs) / 100.0) if confs else 0.0
        return text, "tesseract", confidence
    sidecar = path.with_suffix(".txt").read_text().strip()
    return _corrupt_text(sidecar, noise_level, seed=path.stem), "sidecar", \
        CONFIDENCE_BY_NOISE.get(noise_level, 0.99)


class Toolbox:
    def __init__(self, settings: Settings, policy_index: PolicyIndex | None = None):
        self.settings = settings
        self._db = sqlite3.connect(settings.db_path, check_same_thread=False)
        self._index = policy_index
        self.credit_ledger: list[dict] = []

    # ---- core banking -------------------------------------------------------------------

    @authorized(Scope.ACCOUNT_READ)
    def get_account(self, ctx: ToolContext, account_id: str) -> Account:
        row = self._db.execute("SELECT data FROM accounts WHERE account_id=?", (account_id,)).fetchone()
        if not row:
            raise KeyError(account_id)
        return Account.model_validate_json(row[0])

    @authorized(Scope.TXN_READ)
    def get_transaction(self, ctx: ToolContext, account_id: str, txn_id: str) -> Transaction:
        row = self._db.execute(
            "SELECT data FROM transactions WHERE txn_id=? AND account_id=?", (txn_id, account_id)
        ).fetchone()
        if not row:
            raise KeyError(f"{txn_id} not found on {account_id}")
        return Transaction.model_validate_json(row[0])

    @authorized(Scope.TXN_READ)
    def list_transactions(self, ctx: ToolContext, account_id: str, start: datetime,
                          end: datetime) -> list[Transaction]:
        rows = self._db.execute(
            "SELECT data FROM transactions WHERE account_id=? AND posted_at BETWEEN ? AND ? "
            "ORDER BY posted_at",
            (account_id, start.isoformat(), end.isoformat()),
        ).fetchall()
        return [Transaction.model_validate_json(r[0]) for r in rows]

    @authorized(Scope.TXN_READ)
    def find_similar_transactions(self, ctx: ToolContext, account_id: str, txn_id: str,
                                  window_hours: int = 48) -> list[Transaction]:
        """Other transactions at the same merchant within +/- window_hours of txn_id."""
        target = self.get_transaction(ctx, account_id, txn_id)
        lo = target.posted_at - timedelta(hours=window_hours)
        hi = target.posted_at + timedelta(hours=window_hours)
        return [t for t in self.list_transactions(ctx, account_id, lo, hi)
                if t.merchant == target.merchant and t.txn_id != txn_id]

    # ---- documents ----------------------------------------------------------------------

    @authorized(Scope.DOCS_READ)
    def ocr_receipt(self, ctx: ToolContext, account_id: str, receipt_id: str) -> dict:
        row = self._db.execute(
            "SELECT data FROM receipts WHERE receipt_id=? AND account_id=?", (receipt_id, account_id)
        ).fetchone()
        if not row:
            raise KeyError(f"{receipt_id} not found on {account_id}")
        meta = Receipt.model_validate_json(row[0])
        text, engine, confidence = ocr_image(self.settings.data_dir / meta.image_path,
                                             self.settings.ocr, meta.noise_level)
        total = TOTAL_RE.search(text)
        delivery = DELIVERY_RE.search(text)
        return {
            "receipt_id": receipt_id,
            "kind": meta.kind,
            "text": text,
            "ocr_engine": engine,
            "ocr_confidence": confidence,
            "total_cents": round(float(total.group(1).replace(",", "")) * 100) if total else None,
            "expected_delivery": date.fromisoformat(delivery.group(1)) if delivery else None,
        }

    @authorized(Scope.DOCS_READ)
    def read_receipt_image(self, ctx: ToolContext, account_id: str, receipt_id: str) -> dict:
        """Raw image bytes for the vision-model OCR fallback (M2d) — base64, ready to embed in
        a message content block. Distinct from `ocr_receipt`, which never returns image bytes."""
        row = self._db.execute(
            "SELECT data FROM receipts WHERE receipt_id=? AND account_id=?", (receipt_id, account_id)
        ).fetchone()
        if not row:
            raise KeyError(f"{receipt_id} not found on {account_id}")
        meta = Receipt.model_validate_json(row[0])
        image_bytes = (self.settings.data_dir / meta.image_path).read_bytes()
        return {
            "receipt_id": receipt_id,
            "kind": meta.kind,
            "media_type": "image/png",
            "image_b64": base64.b64encode(image_bytes).decode("ascii"),
        }

    # ---- policy -------------------------------------------------------------------------

    @authorized(Scope.POLICY_SEARCH, account_arg=None)
    def search_policy(self, ctx: ToolContext, query: str, reason_code: str | None = None,
                      as_of: date | None = None, k: int | None = None) -> list[dict]:
        if self._index is None:
            raise RuntimeError("policy index not configured")
        return self._index.search(query, k or self.settings.retrieval_k,
                                  reason_code=reason_code, as_of=as_of)

    # ---- side effects (human-gated) -----------------------------------------------------

    @authorized(Scope.CREDIT_WRITE)
    def issue_provisional_credit(self, ctx: ToolContext, account_id: str, txn_id: str,
                                 amount_cents: int) -> dict:
        record = {"account_id": account_id, "txn_id": txn_id, "amount_cents": amount_cents}
        self.credit_ledger.append(record)
        return record


__all__ = ["Toolbox", "ToolContext", "Scope", "AuthorizationError", "ocr_image"]
