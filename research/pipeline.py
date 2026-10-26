import asyncio
import contextlib
import hashlib
import io
import json
import os
import shutil
import time
from pathlib import Path
import httpx

_convert_lock = asyncio.Lock()
_arxiv_pdf_lock = asyncio.Lock()
_last_pdf_request = 0.0


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _cache_paths(url):
    root = Path(os.getenv("PAPER_CACHE_DIR", "outputs/_cache/fulltext"))
    key = hashlib.sha256(url.encode("utf-8")).hexdigest()
    return root / f"{key}.pdf", root / f"{key}.md"


def restore_cached_fulltext(url, pdf):
    cached_pdf, cached_md = _cache_paths(url)
    try:
        if cached_pdf.stat().st_size < 5 or cached_md.stat().st_size < 80:
            return None
        with cached_pdf.open("rb") as stream:
            if stream.read(5) != b"%PDF-":
                return None
        text = cached_md.read_text(encoding="utf-8")
        if len(text.strip()) < 80:
            return None
        pdf.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cached_pdf, pdf)
        return text
    except (OSError, UnicodeError):
        return None


def store_cached_fulltext(url, pdf, text):
    cached_pdf, cached_md = _cache_paths(url)
    cached_pdf.parent.mkdir(parents=True, exist_ok=True)
    suffix = f".{os.getpid()}.{time.time_ns()}.tmp"
    temp_pdf = cached_pdf.with_name(cached_pdf.name + suffix)
    temp_md = cached_md.with_name(cached_md.name + suffix)
    try:
        shutil.copy2(pdf, temp_pdf)
        temp_md.write_text(text, encoding="utf-8")
        os.replace(temp_pdf, cached_pdf)
        os.replace(temp_md, cached_md)
    finally:
        temp_pdf.unlink(missing_ok=True)
        temp_md.unlink(missing_ok=True)


def convert_pdf(path):
    layout_mode = os.getenv("PDF_LAYOUT_MODE", "legacy").lower()
    if layout_mode not in ("legacy", "onnx"):
        raise ValueError("PDF_LAYOUT_MODE 必须为 legacy 或 onnx")
    # Keep third-party progress/informational prints out of MCP's stdio stdout.
    with contextlib.redirect_stdout(io.StringIO()):
        import pymupdf
        with pymupdf.open(str(path)) as doc:
            if doc.page_count > 120:
                raise ValueError("初版限制为 120 页以内；请人工处理超长文档")
            if layout_mode == "legacy":
                # Importing pymupdf4llm activates its ONNX layout module before
                # use_layout(False) can take effect. Direct PyMuPDF extraction is
                # faster and avoids invalid CPU-affinity attempts inside a Slurm
                # cpuset while retaining complete page-labelled text for the LLM.
                chunks = [page.get_text("text", sort=True) for page in doc]
            else:
                import pymupdf4llm
                pymupdf4llm.use_layout(True)
                converted = pymupdf4llm.to_markdown(
                    doc, page_chunks=True, show_progress=False
                )
                if not isinstance(converted, list):
                    raise ValueError("PDF 转换未返回预期分页文本")
                chunks = [chunk["text"] for chunk in converted]
        text = "\n\n".join(f"[Page {i}]\n{chunk}" for i, chunk in enumerate(chunks, 1))
        if len(text.strip()) < 80:
            raise ValueError("全文文字过少，可能需要 OCR，未生成总结")
        return text


async def download_pdf(url, path, progress=None):
    global _last_pdf_request
    async def fetch():
        proxy = os.getenv("PAPER_PROXY_URL", "").strip() or None
        async with httpx.AsyncClient(timeout=60, follow_redirects=True, proxy=proxy,
                                    trust_env=(not proxy and os.getenv("PAPER_TRUST_ENV", "1") == "1")) as client:
            async with client.stream("GET", url) as response:
                if response.is_error:
                    raise RuntimeError(f"PDF HTTP {response.status_code}")
                total = 0
                try:
                    expected = int(response.headers.get("content-length", 0)) or None
                except ValueError:
                    expected = None
                last_reported = 0
                last_reported_at = time.monotonic()
                with path.open("wb") as target:
                    async for block in response.aiter_bytes():
                        total += len(block)
                        if total > 20 * 1024 * 1024:
                            raise ValueError("PDF 大于初版 20MB 限制")
                        target.write(block)
                        now = time.monotonic()
                        if progress and (total - last_reported >= 1024 * 1024 or
                                         now - last_reported_at >= 5):
                            progress(total, expected)
                            last_reported, last_reported_at = total, now
                if progress and total != last_reported:
                    progress(total, expected)
        with path.open("rb") as stream:
            if stream.read(5) != b"%PDF-":
                raise ValueError("下载内容不是 PDF")
    # Conservative arXiv throttling even in concurrent mode.
    if "arxiv.org/" in url:
        async with _arxiv_pdf_lock:
            await asyncio.sleep(max(0, 3.1 - (time.monotonic() - _last_pdf_request)))
            try:
                await fetch()
            finally:
                _last_pdf_request = time.monotonic()
    else:
        await fetch()


async def process_papers(papers, output, agent, concurrent=False, trace=None):
    output = Path(output)
    (output / "fulltext").mkdir(parents=True, exist_ok=True)
    (output / "summaries").mkdir(exist_ok=True)
    semaphore = asyncio.Semaphore(2 if concurrent else 1)
    def emit(event, data):
        if trace:
            trace(event, data)

    async def one(p):
        async with semaphore:
            started = time.perf_counter()
            result = {key: p.get(key) for key in (
                "id", "title", "url", "sources", "selection_reason",
                "base_rank", "base_score", "rerank_score", "rerank_rank", "final_rank",
            )}
            result["status"] = "failed"
            pdf = output / "fulltext" / (p["id"] + ".pdf")
            errors = []
            emit("paper_started", {"id": p["id"], "title": p["title"]})
            try:
                text = None
                pdf_urls = p.get("pdf_urls", [])[:3]
                # Check every equivalent source before downloading any of them.
                # This lets a cached arXiv copy satisfy a paper even when a new
                # publisher/CVF URL has since become the preferred first choice.
                for url in pdf_urls:
                    cache_started = time.perf_counter()
                    text = await asyncio.to_thread(restore_cached_fulltext, url, pdf)
                    if text is not None:
                        emit("paper_cache_hit", {"id": p["id"], "pdf_url": url,
                                                 "pdf_bytes": pdf.stat().st_size,
                                                 "markdown_chars": len(text),
                                                 "seconds": round(time.perf_counter() - cache_started, 3)})
                        break
                for url in ([] if text is not None else pdf_urls):
                    try:
                        download_started = time.perf_counter()
                        emit("pdf_download_started", {"id": p["id"], "pdf_url": url})
                        await download_pdf(
                            url, pdf,
                            lambda downloaded, expected: emit(
                                "pdf_download_progress",
                                {"id": p["id"], "pdf_url": url,
                                 "downloaded_bytes": downloaded,
                                 "total_bytes": expected},
                            ),
                        )
                        emit("pdf_download_completed", {"id": p["id"], "pdf_url": url,
                                                         "pdf_bytes": pdf.stat().st_size,
                                                         "seconds": round(time.perf_counter() - download_started, 3)})
                        convert_started = time.perf_counter()
                        emit("pdf_conversion_started", {"id": p["id"],
                                                         "mode": os.getenv("PDF_LAYOUT_MODE", "legacy")})
                        async with _convert_lock:
                            text = await asyncio.to_thread(convert_pdf, pdf)
                        emit("pdf_conversion_completed", {"id": p["id"],
                                                           "markdown_chars": len(text),
                                                           "seconds": round(time.perf_counter() - convert_started, 3)})
                        await asyncio.to_thread(store_cached_fulltext, url, pdf, text)
                        break
                    except Exception as exc:
                        errors.append(str(exc))
                        pdf.unlink(missing_ok=True)
                if text is None:
                    raise ValueError("没有可用的公开 PDF；" + "; ".join(errors))
                md = output / "fulltext" / (p["id"] + ".md")
                md.write_text(text, encoding="utf-8")
                result["markdown_path"] = str(md)
                result["summary"] = await agent.summarize(p, text, progress=emit)
                summary = output / "summaries" / (p["id"] + ".md")
                summary.write_text(result["summary"], encoding="utf-8")
                result.update(status="ok", summary_path=str(summary),
                              evidence="extracted_fulltext")
            except Exception as exc:
                result["error"] = str(exc)
                result["evidence"] = "no_completed_summary"
            result["seconds"] = round(time.perf_counter() - started, 3)
            emit("paper_processed", {key: v for key, v in result.items() if key != "summary"})
            return result
    started = time.perf_counter()
    results = await asyncio.gather(*(one(p) for p in papers))
    metrics = {
        "processing_seconds": round(time.perf_counter() - started, 3),
        "selected_count": len(papers), "completed_count": sum(r["status"] == "ok" for r in results),
        "paper_success_rate": sum(r["status"] == "ok" for r in results) / max(1, len(papers)),
        "concurrent": concurrent,
        "measured_scope": "PDF download + Markdown conversion + per-paper summaries",
    }
    return results, metrics
