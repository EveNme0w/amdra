"""Lexical (BM25) retrieval over the same policy chunks PolicyIndex embeds.

Used only for hybrid fusion (`store.PolicyIndex.search(..., hybrid=True)`) via reciprocal rank
fusion — not a standalone retrieval path. Scores the whole corpus, then applies the same
product/reason-code/effective-date filters as the dense side before ranking, so both rankers see
the same candidate pool.
"""
from __future__ import annotations

import re
from datetime import date
from pathlib import Path

from rank_bm25 import BM25Okapi

from amdra.retrieval.store import chunk_policies, date_int


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25Index:
    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        corpus = [_tokenize(f"{c['metadata']['title']} {c['metadata']['section']}: {c['text']}")
                 for c in chunks]
        self._bm25 = BM25Okapi(corpus)

    @classmethod
    def from_dir(cls, policies_dir: Path) -> BM25Index:
        return cls(chunk_policies(policies_dir))

    def search(self, query: str, k: int, *, reason_code: str | None = None,
              product: str = "credit", as_of: date | None = None) -> list[dict]:
        scores = self._bm25.get_scores(_tokenize(query))
        d = date_int(as_of) if as_of else None
        ranked = []
        for chunk, score in zip(self.chunks, scores):
            m = chunk["metadata"]
            if m["product"] != product:
                continue
            if reason_code and not m.get(f"rc_{reason_code}"):
                continue
            if d is not None and not (m["effective_from"] <= d < m["effective_to"]):
                continue
            ranked.append({"id": chunk["id"], "document": chunk["text"], "metadata": m,
                           "score": float(score)})
        ranked.sort(key=lambda r: -r["score"])
        return ranked[:k]
