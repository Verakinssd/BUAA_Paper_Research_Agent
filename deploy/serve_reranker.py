"""Serve an optional Cross-Encoder reranker over a local HTTP endpoint."""
import asyncio
import os
import threading
from pathlib import Path

from dotenv import load_dotenv
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route


PROJECT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT / ".env", override=False)

_model = None
_model_key = None
_model_lock = threading.Lock()


def env_int(name, default, minimum=1):
    try:
        value = int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default
    return value if value >= minimum else default


def load_model():
    global _model, _model_key
    model_name = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3").strip()
    device = os.getenv("RERANKER_DEVICE", "cpu").strip() or "cpu"
    max_length = env_int("RERANKER_MAX_LENGTH", 512, minimum=32)
    key = (model_name, device, max_length)
    if _model is not None and _model_key == key:
        return _model
    with _model_lock:
        if _model is not None and _model_key == key:
            return _model
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError("请先安装 requirements-reranker.txt") from exc
        _model = CrossEncoder(model_name, device=device, max_length=max_length)
        _model_key = key
        return _model


def score(query, documents):
    model = load_model()
    pairs = []
    for document in documents:
        title = str(document.get("title") or "").strip()
        abstract = str(document.get("abstract") or "").strip()
        pairs.append((query, f"{title}\n{abstract}" if abstract else title))
    values = model.predict(
        pairs,
        batch_size=env_int("RERANKER_BATCH_SIZE", 8),
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return [float(value) for value in values]


async def rerank_endpoint(request):
    if request.method != "POST":
        return JSONResponse({"error": "请使用 POST"}, status_code=405)
    try:
        body = await request.json()
    except ValueError:
        return JSONResponse({"error": "JSON 格式不正确"}, status_code=400)
    query = body.get("query") if isinstance(body, dict) else None
    documents = body.get("documents") if isinstance(body, dict) else None
    if not isinstance(query, str) or not query.strip():
        return JSONResponse({"error": "query 必须为非空字符串"}, status_code=400)
    if not isinstance(documents, list) or not 1 <= len(documents) <= 128:
        return JSONResponse({"error": "documents 数量必须为 1–128"}, status_code=400)
    if any(not isinstance(document, dict) or not isinstance(document.get("id"), str)
           for document in documents):
        return JSONResponse({"error": "每个 document 必须包含字符串 id"}, status_code=400)
    try:
        scores = await asyncio.to_thread(score, query.strip(), documents)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=503)
    return JSONResponse({
        "model": os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"),
        "scores": [{"id": document["id"], "score": score_value}
                   for document, score_value in zip(documents, scores)],
    })


app = Starlette(routes=[Route("/rerank", rerank_endpoint, methods=["POST"])])


if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8003)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("port 必须为 1–65535")
    uvicorn.run(app, host=args.host, port=args.port)
