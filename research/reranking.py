"""Optional second-stage Cross-Encoder reranking with safe fallbacks."""
from __future__ import annotations

import asyncio
import logging
import math
import os
import threading
import time
from typing import Any

import httpx


log = logging.getLogger(__name__)
VALID_METHODS = ("none", "local", "http")
DEFAULT_METHOD = "none"

_local_model = None
_local_model_key = None
_local_model_lock = threading.Lock()


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default
    return value if value >= minimum else default


def _method() -> str:
    value = os.getenv("PAPER_RERANKING_METHOD", DEFAULT_METHOD).strip().lower()
    if value not in VALID_METHODS:
        log.warning("unknown PAPER_RERANKING_METHOD '%s', disabling reranking", value)
        return DEFAULT_METHOD
    return value


def _document_text(paper: dict[str, Any]) -> str:
    title = str(paper.get("title") or "").strip()
    abstract = str(paper.get("abstract") or "").strip()
    return f"{title}\n{abstract}" if abstract else title


def _copy_with_base_rank(papers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    copied = []
    for position, paper in enumerate(papers, start=1):
        item = dict(paper)
        item.setdefault("base_rank", position)
        item.setdefault("base_score", item.get("rank_score"))
        copied.append(item)
    return copied


def _validate_scores(scores: Any, expected: int) -> list[float]:
    if not isinstance(scores, list) or len(scores) != expected:
        raise ValueError(f"Reranker 返回 {len(scores) if isinstance(scores, list) else 0} 个分数，预期 {expected} 个")
    result = []
    for value in scores:
        score = float(value)
        if not math.isfinite(score):
            raise ValueError("Reranker 返回了无效分数")
        result.append(score)
    return result


def _load_local_model():
    global _local_model, _local_model_key
    model_name = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3").strip()
    device = os.getenv("RERANKER_DEVICE", "cpu").strip() or "cpu"
    max_length = _env_int("RERANKER_MAX_LENGTH", 512, minimum=32)
    key = (model_name, device, max_length)
    if _local_model is not None and _local_model_key == key:
        return _local_model
    with _local_model_lock:
        if _local_model is not None and _local_model_key == key:
            return _local_model
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError(
                "local Reranker 需要可选依赖，请安装 requirements-reranker.txt"
            ) from exc
        log.info("loading Cross-Encoder model=%s device=%s", model_name, device)
        _local_model = CrossEncoder(model_name, device=device, max_length=max_length)
        _local_model_key = key
        return _local_model


def _local_scores_sync(query: str, papers: list[dict[str, Any]]) -> list[float]:
    model = _load_local_model()
    pairs = [(query, _document_text(paper)) for paper in papers]
    values = model.predict(
        pairs,
        batch_size=_env_int("RERANKER_BATCH_SIZE", 8),
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return _validate_scores(list(values), len(papers))


async def _local_scores(query: str, papers: list[dict[str, Any]]) -> list[float]:
    return await asyncio.to_thread(_local_scores_sync, query, papers)


def _http_url() -> str:
    base = os.getenv("RERANKER_SERVICE_URL", "http://127.0.0.1:8003").rstrip("/")
    return base if base.endswith("/rerank") else base + "/rerank"


async def _http_scores(query: str, papers: list[dict[str, Any]]) -> list[float]:
    documents = [{
        "id": paper.get("id"),
        "title": paper.get("title") or "",
        "abstract": paper.get("abstract") or "",
    } for paper in papers]
    headers = {}
    key = os.getenv("RERANKER_API_KEY", "").strip()
    if key:
        headers["Authorization"] = "Bearer " + key
    timeout = max(1.0, float(os.getenv("RERANKER_TIMEOUT", "30")))
    trust_env = os.getenv("RERANKER_TRUST_ENV", "0") == "1"
    async with httpx.AsyncClient(timeout=timeout, trust_env=trust_env) as client:
        response = await client.post(_http_url(), json={"query": query, "documents": documents}, headers=headers)
    if response.is_error:
        raise RuntimeError(f"Reranker HTTP {response.status_code}")
    try:
        payload = response.json()
        values = payload["scores"]
    except (ValueError, KeyError, TypeError) as exc:
        raise ValueError("Reranker 响应格式无效") from exc
    if isinstance(values, list) and values and isinstance(values[0], dict):
        by_id = {}
        for item in values:
            if not isinstance(item, dict) or "id" not in item or "score" not in item:
                raise ValueError("Reranker 分数项格式无效")
            by_id[item["id"]] = item["score"]
        scores = [by_id.get(paper.get("id")) for paper in papers]
        if any(value is None for value in scores):
            raise ValueError("Reranker 未返回所有候选论文的分数")
        values = scores
    return _validate_scores(values, len(papers))


async def rerank(query: str, papers: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Rerank first-stage candidates while preserving a safe fallback path."""
    started = time.perf_counter()
    result = _copy_with_base_rank(papers)
    method = _method()
    limit = min(len(result), _env_int("RERANKER_TOP_N", 16))
    stats: dict[str, Any] = {
        "method": method,
        "status": "ok" if method == "none" else "pending",
        "fallback": False,
        "candidate_count": len(result),
        "processed_count": 0,
    }
    if not result or method == "none":
        for position, paper in enumerate(result, start=1):
            paper["final_rank"] = position
        stats.update(processed_count=0, seconds=round(time.perf_counter() - started, 3))
        log.info(
            "Cross-Encoder reranking skipped: method=%s candidates=%d",
            method, len(result),
        )
        return result, stats

    selected = result[:limit]
    log.info(
        "Cross-Encoder reranking started: method=%s candidates=%d processing=%d",
        method, len(result), len(selected),
    )
    try:
        scores = await (_local_scores(query, selected) if method == "local"
                       else _http_scores(query, selected))
        scores = _validate_scores(scores, len(selected))
        for paper, score in zip(selected, scores):
            paper["rerank_score"] = round(score, 6)
        selected.sort(key=lambda paper: (-paper["rerank_score"], paper["base_rank"]))
        for position, paper in enumerate(selected, start=1):
            paper["rerank_rank"] = position
        ordered = selected + result[limit:]
        stats.update(status="ok", processed_count=len(selected))
    except Exception as exc:
        if os.getenv("RERANKER_FALLBACK", "1") != "1":
            raise
        ordered = result
        stats.update(status="fallback", fallback=True, error=str(exc), processed_count=0)

    for position, paper in enumerate(ordered, start=1):
        paper["final_rank"] = position
    stats["seconds"] = round(time.perf_counter() - started, 3)
    stats["top_changes"] = [{
        "id": paper.get("id"),
        "base_rank": paper.get("base_rank"),
        "final_rank": paper.get("final_rank"),
        "rerank_score": paper.get("rerank_score"),
    } for paper in ordered[:5]]
    summary = ", ".join(
        f'{item["id"]}:{item["base_rank"]}->{item["final_rank"]}'
        + (f'({item["rerank_score"]:.4f})' if item["rerank_score"] is not None else "")
        for item in stats["top_changes"]
    ) or "none"
    if stats["status"] == "fallback":
        log.warning(
            "Cross-Encoder reranking fallback: method=%s candidates=%d processing=%d "
            "seconds=%.3f reason=%s",
            method, len(result), len(selected), stats["seconds"], stats.get("error", "unknown"),
        )
    else:
        log.info(
            "Cross-Encoder reranking completed: method=%s processed=%d/%d "
            "seconds=%.3f top=%s",
            method, len(selected), len(result), stats["seconds"], summary,
        )
    return ordered, stats
