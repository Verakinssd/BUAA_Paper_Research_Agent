"""Pluggable relevance ranking for paper retrieval.

Four strategies are exposed through :func:`rank` which dispatches on the
``method`` argument. Each strategy assigns a numeric ``rank_score`` to every
paper in the input list, sorts by that score descending, and renumbers ``id``
to ``P1..Pn`` so downstream code (TaskAgent, server, web) stays stable.

method              | description
--------------------+------------------------------------------------------
basic               | original token-overlap heuristic (title × 2 + abstract)
bm25                | BM25Okapi on title + abstract
dense_linear        | BGE-M3 cosine similarity, no lexical signal
hybrid_rrf (default)| BM25 + BGE-M3 fused by Reciprocal Rank Fusion
"""
from __future__ import annotations

import logging
import os
import re
from typing import List, Dict, Tuple, Optional

log = logging.getLogger(__name__)


VALID_METHODS = ("basic", "bm25", "dense_linear", "hybrid_rrf")
DEFAULT_METHOD = "basic"

_RRF_K = 60


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def _doc_text(paper: Dict) -> str:
    title = paper.get("title", "") or ""
    abstract = paper.get("abstract", "") or ""
    return f"{title}. {abstract}" if abstract else title


def _renumber(papers: List[Dict]) -> List[Dict]:
    for i, p in enumerate(papers, 1):
        p["id"] = f"P{i}"
    return papers


# ---------------------------------------------------------------------------
# 1. basic — original token-overlap heuristic
# ---------------------------------------------------------------------------

def basic_rank(query: str, papers: List[Dict]) -> List[Dict]:
    tokens = set(_tokenize(query))
    for p in papers:
        title = set(_tokenize(p.get("title", "")))
        abstract = set(_tokenize(p.get("abstract", "")))
        p["rank_score"] = round(
            (2 * len(tokens & title) + len(tokens & abstract)) / max(1, 3 * len(tokens)), 4
        )
    ranked = sorted(papers, key=lambda p: p["rank_score"], reverse=True)
    return _renumber(ranked)


# ---------------------------------------------------------------------------
# 2. bm25 — BM25Okapi lexical ranking
# ---------------------------------------------------------------------------

def _bm25_scores(query: str, papers: List[Dict]) -> List[float]:
    from rank_bm25 import BM25Okapi
    docs = [_tokenize(_doc_text(p)) for p in papers]
    bm25 = BM25Okapi(docs)
    return list(bm25.get_scores(_tokenize(query)))


def bm25_rank(query: str, papers: List[Dict]) -> List[Dict]:
    scores = _bm25_scores(query, papers)
    max_s = max(scores) if scores else 1.0
    for p, s in zip(papers, scores):
        p["rank_score"] = round(float(s) / max(1e-9, max_s), 6) if max_s > 0 else 0.0
    ranked = sorted(papers, key=lambda p: p["rank_score"], reverse=True)
    return _renumber(ranked)


# ---------------------------------------------------------------------------
# 3. dense_linear — BGE-M3 cosine similarity
# ---------------------------------------------------------------------------

class _VectorRanker:
    _instance: Optional["_VectorRanker"] = None
    _error: Optional[str] = None

    def __init__(self):
        self.base_url = os.getenv("BGE_M3_SERVICE_URL", "").rstrip("/")
        self.timeout = float(os.getenv("BGE_M3_SERVICE_TIMEOUT", "30"))
        if self.base_url:
            log.info("BGE-M3 embedding service configured at %s", self.base_url)
        else:
            log.warning("BGE_M3_SERVICE_URL not set, dense ranking disabled")

    @classmethod
    def get(cls) -> "_VectorRanker":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @property
    def available(self) -> bool:
        return bool(self.base_url)

    def _encode_batch(self, texts: List[str]) -> Optional[List[List[float]]]:
        import httpx
        try:
            response = httpx.post(
                f"{self.base_url}/embed",
                json={"texts": texts},
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
            return data["embeddings"]
        except Exception as exc:
            _VectorRanker._error = str(exc)
            log.warning("BGE-M3 service request failed: %s", exc)
            return None

    def similarities(self, query: str, papers: List[Dict]) -> Optional[List[float]]:
        if not self.available or not papers:
            return None
        import torch
        import torch.nn.functional as F
        docs = [_doc_text(p) for p in papers]
        all_texts = [query] + docs
        all_vecs = self._encode_batch(all_texts)
        if all_vecs is None:
            return None
        if len(all_vecs) != len(all_texts):
            log.warning("BGE-M3 service returned %d embeddings for %d texts", len(all_vecs), len(all_texts))
            return None
        try:
            query_vec = F.normalize(torch.as_tensor(all_vecs[0], dtype=torch.float32).unsqueeze(0), p=2, dim=-1)
            doc_vecs = F.normalize(torch.as_tensor(all_vecs[1:], dtype=torch.float32), p=2, dim=-1)
            log.debug("embedding dim=%d, doc_count=%d", query_vec.shape[-1], doc_vecs.shape[0])
            sims = (doc_vecs @ query_vec.squeeze(0)).tolist()
            return [float(s) for s in sims]
        except Exception as exc:
            _VectorRanker._error = str(exc)
            log.warning("BGE-M3 similarity calc failed: %s", exc)
            return None


def dense_rank(query: str, papers: List[Dict]) -> List[Dict]:
    ranker = _VectorRanker.get()
    sims = ranker.similarities(query, papers) if ranker.available else None
    if sims is None:
        log.warning("dense_rank falling back to original order (vector unavailable)")
        for p in papers:
            p["rank_score"] = 0.0
        return _renumber(list(papers))
    log.debug("dense similarities: %s", [round(s, 4) for s in sims])
    for p, s in zip(papers, sims):
        p["rank_score"] = round(max(0.0, float(s)), 6)
    ranked = sorted(papers, key=lambda p: p["rank_score"], reverse=True)
    return _renumber(ranked)


# ---------------------------------------------------------------------------
# 4. hybrid_rrf — BM25 + vector, fused by Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

def _rrf_fuse(rank_lists: List[List[int]], k: int = _RRF_K) -> List[Tuple[int, float]]:
    scores: Dict[int, float] = {}
    for ranks in rank_lists:
        for position, doc_idx in enumerate(ranks, start=1):
            scores[doc_idx] = scores.get(doc_idx, 0.0) + 1.0 / (k + position)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


def hybrid_rank(query: str, papers: List[Dict], rrf_k: int = _RRF_K) -> List[Dict]:
    scores = _bm25_scores(query, papers)
    bm25_order = sorted(range(len(papers)), key=lambda i: scores[i], reverse=True)

    vector_order: Optional[List[int]] = None
    ranker = _VectorRanker.get()
    if ranker.available:
        sims = ranker.similarities(query, papers)
        if sims is not None:
            vector_order = sorted(range(len(papers)), key=lambda i: sims[i], reverse=True)

    if vector_order is not None:
        log.debug("hybrid_rrf: fusing BM25 + vector rankings")
        fused = _rrf_fuse([bm25_order, vector_order], k=rrf_k)
    else:
        log.warning("hybrid_rrf: vector unavailable, falling back to pure BM25")
        fallback = {}
        for pos, idx in enumerate(bm25_order, start=1):
            fallback[idx] = 1.0 / (rrf_k + pos)
        fused = sorted(fallback.items(), key=lambda kv: kv[1], reverse=True)

    ranked = []
    for new_pos, (orig_idx, score) in enumerate(fused, start=1):
        paper = dict(papers[orig_idx])
        paper["rank_score"] = round(score, 6)
        paper["id"] = f"P{new_pos}"
        ranked.append(paper)
    return ranked


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def rank(query: str, papers: List[Dict],
         method: Optional[str] = None) -> List[Dict]:
    """Dispatch to the chosen ranking strategy.

    ``method`` is resolved in this order: explicit argument >
    ``PAPER_RANKING_METHOD`` env var > :data:`DEFAULT_METHOD`. An unknown value
    silently falls back to ``hybrid_rrf`` so a typo in production config can't
    break retrieval.
    """
    if not papers:
        return papers

    resolved = (method
                or os.getenv("PAPER_RANKING_METHOD", DEFAULT_METHOD)
                or DEFAULT_METHOD)
    resolved = resolved.strip().lower()
    if resolved not in VALID_METHODS:
        log.warning("unknown PAPER_RANKING_METHOD '%s', falling back to '%s'",
                    resolved, DEFAULT_METHOD)
        resolved = DEFAULT_METHOD

    log.info("ranking %d papers via '%s' (query=%r)", len(papers), resolved, query)
    if resolved == "basic":
        return basic_rank(query, papers)
    if resolved == "bm25":
        return bm25_rank(query, papers)
    if resolved == "dense_linear":
        return dense_rank(query, papers)
    return hybrid_rank(query, papers)