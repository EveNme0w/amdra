"""Policy retrieval: embeddings, a vector-store interface (Chroma or in-memory), and indexing.

Metadata is stored flat so the same `where` filters work in Chroma and in memory:
  policy_id, version, section, product, effective_from/effective_to (YYYYMMDD ints),
  rc_<reason_code>: bool for each applicable reason code.
"""
from __future__ import annotations

import hashlib
import math
import re
from datetime import date
from pathlib import Path
from typing import Any, Protocol

from amdra.config import Settings
from amdra.data.policies import parse_markdown

OPEN_ENDED = 99991231


def date_int(d: date | str | None) -> int:
    if d is None:
        return OPEN_ENDED
    if isinstance(d, str):
        d = date.fromisoformat(d[:10])
    return d.year * 10000 + d.month * 100 + d.day


# --------------------------------------------------------------------------- embeddings

class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class HashingEmbedder:
    """Deterministic bag-of-words hashing embedder. Offline and good enough for tests."""

    def __init__(self, dim: int = 512):
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            v = [0.0] * self.dim
            for tok in re.findall(r"[a-z0-9]+", t.lower()):
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                v[h % self.dim] += 1.0 if (h >> 64) & 1 else -1.0
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / n for x in v])
        return out


class ChromaDefaultEmbedder:
    """all-MiniLM-L6-v2 via Chroma's bundled ONNX runtime (downloads the model on first use)."""

    def __init__(self):
        from chromadb.utils import embedding_functions

        self._fn = embedding_functions.DefaultEmbeddingFunction()

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [list(map(float, v)) for v in self._fn(texts)]


def make_embedder(settings: Settings) -> Embedder:
    return HashingEmbedder() if settings.embedder == "hashing" else ChromaDefaultEmbedder()


# --------------------------------------------------------------------------- stores

class VectorStore(Protocol):
    def upsert(self, ids: list[str], embeddings: list[list[float]], documents: list[str],
               metadatas: list[dict]) -> None: ...

    def query(self, embedding: list[float], k: int, where: dict | None) -> list[dict]: ...

    def count(self) -> int: ...


def _match(meta: dict, where: dict | None) -> bool:
    """Evaluate the subset of Chroma's `where` syntax used in this project."""
    if not where:
        return True
    if "$and" in where:
        return all(_match(meta, w) for w in where["$and"])
    if "$or" in where:
        return any(_match(meta, w) for w in where["$or"])
    ops = {"$eq": lambda a, b: a == b, "$ne": lambda a, b: a != b,
           "$lt": lambda a, b: a < b, "$lte": lambda a, b: a <= b,
           "$gt": lambda a, b: a > b, "$gte": lambda a, b: a >= b,
           "$in": lambda a, b: a in b}
    for key, cond in where.items():
        if key not in meta:
            return False
        if not isinstance(cond, dict):
            cond = {"$eq": cond}
        for op, val in cond.items():
            if not ops[op](meta[key], val):
                return False
    return True


class InMemoryStore:
    def __init__(self):
        self._rows: dict[str, tuple[list[float], str, dict]] = {}

    def upsert(self, ids, embeddings, documents, metadatas):
        for i, e, d, m in zip(ids, embeddings, documents, metadatas):
            self._rows[i] = (e, d, m)

    def query(self, embedding, k, where):
        scored = []
        for i, (e, d, m) in self._rows.items():
            if _match(m, where):
                sim = sum(a * b for a, b in zip(embedding, e))
                scored.append({"id": i, "document": d, "metadata": m, "score": sim})
        return sorted(scored, key=lambda r: -r["score"])[:k]

    def count(self):
        return len(self._rows)


class ChromaStore:
    def __init__(self, path: Path, collection: str = "policies"):
        import chromadb

        self._client = chromadb.PersistentClient(path=str(path))
        self._col = self._client.get_or_create_collection(
            collection, metadata={"hnsw:space": "cosine"}
        )

    def upsert(self, ids, embeddings, documents, metadatas):
        self._col.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)

    def query(self, embedding, k, where):
        n = self._col.count()
        if n == 0:
            return []
        res = self._col.query(query_embeddings=[embedding], n_results=min(k, n), where=where or None)
        return [
            {"id": i, "document": d, "metadata": m, "score": 1.0 - dist}
            for i, d, m, dist in zip(res["ids"][0], res["documents"][0], res["metadatas"][0],
                                     res["distances"][0])
        ]

    def count(self):
        return self._col.count()


def make_store(settings: Settings) -> VectorStore:
    if settings.vector_backend == "memory":
        return InMemoryStore()
    # one collection per embedder: vectors from different embedders are not comparable
    return ChromaStore(settings.chroma_dir, collection=f"policies_{settings.embedder}")


# --------------------------------------------------------------------------- indexing

def chunk_policies(policies_dir: Path) -> list[dict]:
    chunks = []
    for md in sorted(Path(policies_dir).glob("*.md")):
        meta, sections = parse_markdown(md.read_text())
        for s in sections:
            cid = f"{meta['policy_id']}-v{meta['version']}-{s['section'].replace('§', 'S')}"
            md_meta: dict[str, Any] = {
                "policy_id": meta["policy_id"],
                "title": meta["title"],
                "version": meta["version"],
                "section": s["section"],
                "citation": f"{meta['policy_id']} {s['section']}",
                "product": meta["product"],
                "effective_from": date_int(meta["effective_from"]),
                "effective_to": date_int(meta["effective_to"]),
                "source_file": md.name,
            }
            for rc in s["reason_codes"]:
                md_meta[f"rc_{rc}"] = True
            chunks.append({"id": cid, "text": s["text"], "metadata": md_meta})
    return chunks


RRF_K = 60  # standard reciprocal-rank-fusion damping constant


def _rrf_fuse(dense: list[dict], lexical: list[dict], k: int) -> list[dict]:
    """Reciprocal rank fusion: combine two rankings of the same candidate pool by id."""
    scores: dict[str, float] = {}
    docs: dict[str, dict] = {}
    for ranking in (dense, lexical):
        for rank, r in enumerate(ranking, start=1):
            scores[r["id"]] = scores.get(r["id"], 0.0) + 1.0 / (RRF_K + rank)
            docs.setdefault(r["id"], r)
    ordered = sorted(scores, key=lambda i: -scores[i])[:k]
    return [{**docs[i], "score": scores[i]} for i in ordered]


class PolicyIndex:
    def __init__(self, store: VectorStore, embedder: Embedder, bm25=None):
        self.store, self.embedder, self.bm25 = store, embedder, bm25

    @classmethod
    def from_settings(cls, settings: Settings, build: bool = True) -> PolicyIndex:
        bm25 = None
        if settings.hybrid_retrieval:
            from amdra.retrieval.bm25 import BM25Index

            bm25 = BM25Index.from_dir(settings.policies_dir)
        idx = cls(make_store(settings), make_embedder(settings), bm25)
        if build:  # upsert is idempotent and the corpus is small, so always sync
            idx.build(settings.policies_dir)
        return idx

    def build(self, policies_dir: Path) -> int:
        chunks = chunk_policies(policies_dir)
        # embed title + section text so topical queries match
        texts = [f"{c['metadata']['title']} {c['metadata']['section']}: {c['text']}" for c in chunks]
        self.store.upsert([c["id"] for c in chunks], self.embedder.embed(texts),
                          [c["text"] for c in chunks], [c["metadata"] for c in chunks])
        return len(chunks)

    def search(self, query: str, k: int, *, reason_code: str | None = None,
               product: str = "credit", as_of: date | None = None) -> list[dict]:
        clauses: list[dict] = [{"product": {"$eq": product}}]
        if reason_code:
            clauses.append({f"rc_{reason_code}": {"$eq": True}})
        if as_of:
            d = date_int(as_of)
            clauses += [{"effective_from": {"$lte": d}}, {"effective_to": {"$gt": d}}]
        where = clauses[0] if len(clauses) == 1 else {"$and": clauses}
        if self.bm25 is None:
            return self.store.query(self.embedder.embed([query])[0], k, where)
        pool = max(k * 3, 10)  # fetch a wider candidate pool from each ranker before fusing
        dense = self.store.query(self.embedder.embed([query])[0], pool, where)
        lexical = self.bm25.search(query, pool, reason_code=reason_code, product=product, as_of=as_of)
        return _rrf_fuse(dense, lexical, k)
