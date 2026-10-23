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

PIPELINE_MODES = {"legacy", "staged"}


def _positive_int(name, default):
    raw = os.getenv(name, str(default))
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{name} 必须为正整数") from None
    if value <= 0:
        raise ValueError(f"{name} 必须为正整数")
    return value


def pipeline_settings(concurrent=False, mode=None):
    """Resolve scheduling without changing the legacy default entry points.

    With no explicit PIPELINE_MODE, the existing `concurrent` flag keeps its
    original meaning: false is serial and true uses the whole-paper limit of 2.
    An explicit mode is used by reproducible benchmarks and deployments that
    want to opt in to staged scheduling.
    """
    explicit = mode if mode is not None else os.getenv("PIPELINE_MODE")
    explicit = explicit.strip().lower() if explicit else ""
    if explicit and explicit not in PIPELINE_MODES:
        raise ValueError("PIPELINE_MODE 必须为 legacy 或 staged")
    selected = explicit or "legacy"
    enabled = bool(concurrent) or bool(explicit)
    return {
        "mode": selected,
        "enabled": enabled,
        "legacy_concurrency": 2 if enabled else 1,
        "download_concurrency": _positive_int("DOWNLOAD_CONCURRENCY", 4),
        "convert_concurrency": _positive_int("CONVERT_CONCURRENCY", 1),
        "summary_concurrency": _positive_int("SUMMARY_CONCURRENCY", 2),
        "max_in_flight": _positive_int("PIPELINE_MAX_IN_FLIGHT", 8),
    }


def _stage_timings():
    return {
        name: {"seconds": 0.0, "wait_seconds": 0.0, "attempts": 0, "failures": 0}
        for name in ("cache", "download", "convert", "summary")
    }


async def _run_stage(semaphore, timing, operation):
    wait_started = time.perf_counter()
    if semaphore is not None:
        await semaphore.acquire()
    timing["wait_seconds"] += time.perf_counter() - wait_started if semaphore is not None else 0
    timing["attempts"] += 1
    active_started = time.perf_counter()
    try:
        return await operation()
    except Exception:
        timing["failures"] += 1
        raise
    finally:
        timing["seconds"] += time.perf_counter() - active_started
        if semaphore is not None:
            semaphore.release()


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


async def process_papers(papers, output, agent, concurrent=False, trace=None, mode=None):
    """Download, convert and summarize papers with legacy or staged scheduling."""
    papers = list(papers)
    settings = pipeline_settings(concurrent, mode)
    output = Path(output)
    (output / "fulltext").mkdir(parents=True, exist_ok=True)
    (output / "summaries").mkdir(exist_ok=True)

    # Legacy deliberately keeps one semaphore around the complete per-paper
    # workflow. Staged mode owns a different semaphore for each expensive stage.
    whole_paper_gate = (asyncio.Semaphore(settings["legacy_concurrency"])
                        if settings["mode"] == "legacy" else None)
    download_gate = (asyncio.Semaphore(settings["download_concurrency"])
                     if settings["mode"] == "staged" else None)
    convert_gate = (asyncio.Semaphore(settings["convert_concurrency"])
                    if settings["mode"] == "staged" else _convert_lock)
    summary_gate = (asyncio.Semaphore(settings["summary_concurrency"])
                    if settings["mode"] == "staged" else None)

    def emit(event, data):
        if trace:
            trace(event, data)

    async def process_one(p):
        scheduler_wait_started = time.perf_counter()
        if whole_paper_gate is not None:
            await whole_paper_gate.acquire()
        scheduler_wait = (time.perf_counter() - scheduler_wait_started
                          if whole_paper_gate is not None else 0.0)
        started = time.perf_counter()
        timings = _stage_timings()
        result = {key: p.get(key) for key in ("id", "title", "url", "sources", "selection_reason")}
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
                timings["cache"]["attempts"] += 1
                try:
                    text = await asyncio.to_thread(restore_cached_fulltext, url, pdf)
                except Exception:
                    timings["cache"]["failures"] += 1
                    raise
                finally:
                    timings["cache"]["seconds"] += time.perf_counter() - cache_started
                if text is not None:
                    emit("paper_cache_hit", {"id": p["id"], "pdf_url": url,
                                             "pdf_bytes": pdf.stat().st_size,
                                             "markdown_chars": len(text),
                                             "seconds": round(time.perf_counter() - cache_started, 3)})
                    break
            for url in ([] if text is not None else pdf_urls):
                try:
                    emit("pdf_download_started", {"id": p["id"], "pdf_url": url})

                    async def download_operation():
                        await download_pdf(
                            url, pdf,
                            lambda downloaded, expected: emit(
                                "pdf_download_progress",
                                {"id": p["id"], "pdf_url": url,
                                 "downloaded_bytes": downloaded,
                                 "total_bytes": expected},
                            ),
                        )

                    before = dict(timings["download"])
                    await _run_stage(download_gate, timings["download"], download_operation)
                    emit("pdf_download_completed", {
                        "id": p["id"], "pdf_url": url, "pdf_bytes": pdf.stat().st_size,
                        "seconds": round(timings["download"]["seconds"] - before["seconds"], 3),
                        "wait_seconds": round(timings["download"]["wait_seconds"] - before["wait_seconds"], 3),
                    })
                    emit("pdf_conversion_started", {"id": p["id"],
                                                     "mode": os.getenv("PDF_LAYOUT_MODE", "legacy")})

                    async def convert_operation():
                        return await asyncio.to_thread(convert_pdf, pdf)

                    before = dict(timings["convert"])
                    text = await _run_stage(convert_gate, timings["convert"], convert_operation)
                    emit("pdf_conversion_completed", {
                        "id": p["id"], "markdown_chars": len(text),
                        "seconds": round(timings["convert"]["seconds"] - before["seconds"], 3),
                        "wait_seconds": round(timings["convert"]["wait_seconds"] - before["wait_seconds"], 3),
                    })
                    async def store_cache_operation():
                        await asyncio.to_thread(store_cached_fulltext, url, pdf, text)

                    await _run_stage(None, timings["cache"], store_cache_operation)
                    break
                except Exception as exc:
                    errors.append(str(exc))
                    pdf.unlink(missing_ok=True)
            if text is None:
                raise ValueError("没有可用的公开 PDF；" + "; ".join(errors))
            md = output / "fulltext" / (p["id"] + ".md")
            md.write_text(text, encoding="utf-8")
            result["markdown_path"] = str(md)

            async def summary_operation():
                return await agent.summarize(p, text, progress=emit)

            result["summary"] = await _run_stage(
                summary_gate, timings["summary"], summary_operation
            )
            summary = output / "summaries" / (p["id"] + ".md")
            summary.write_text(result["summary"], encoding="utf-8")
            result.update(status="ok", summary_path=str(summary),
                          evidence="extracted_fulltext")
        except Exception as exc:
            result["error"] = str(exc)
            result["evidence"] = "no_completed_summary"
        result["seconds"] = round(time.perf_counter() - started, 3)
        result["timings"] = {
            "scheduler_wait_seconds": round(scheduler_wait, 6),
            "stages": {
                name: {key: round(value, 6) if isinstance(value, float) else value
                       for key, value in timing.items()}
                for name, timing in timings.items()
            },
        }
        try:
            emit("paper_processed", {key: value for key, value in result.items() if key != "summary"})
        finally:
            if whole_paper_gate is not None:
                whole_paper_gate.release()
        return result

    # A fixed worker pool avoids creating one asyncio Task per input paper. The
    # queue is implicit in the protected index and bounded by max_in_flight.
    results = [None] * len(papers)
    next_index = 0
    index_lock = asyncio.Lock()

    async def worker():
        nonlocal next_index
        while True:
            async with index_lock:
                if next_index >= len(papers):
                    return
                index = next_index
                next_index += 1
            results[index] = await process_one(papers[index])

    started = time.perf_counter()
    worker_count = min(len(papers), settings["max_in_flight"])
    if worker_count:
        await asyncio.gather(*(worker() for _ in range(worker_count)))
    elapsed = time.perf_counter() - started
    completed = sum(result["status"] == "ok" for result in results)
    stage_metrics = {}
    for name in ("cache", "download", "convert", "summary"):
        rows = [result["timings"]["stages"][name] for result in results]
        stage_metrics[name] = {
            "seconds": round(sum(row["seconds"] for row in rows), 6),
            "wait_seconds": round(sum(row["wait_seconds"] for row in rows), 6),
            "attempts": sum(row["attempts"] for row in rows),
            "failures": sum(row["failures"] for row in rows),
        }
    metrics = {
        "processing_seconds": round(elapsed, 3),
        "selected_count": len(papers), "attempted_count": len(papers),
        "completed_count": completed, "failed_count": len(papers) - completed,
        "failure_rate": (len(papers) - completed) / max(1, len(papers)),
        "paper_success_rate": completed / max(1, len(papers)),
        "throughput_papers_per_second": round(len(papers) / elapsed, 6) if elapsed else 0.0,
        "successful_papers_per_second": round(completed / elapsed, 6) if elapsed else 0.0,
        "concurrent": settings["enabled"], "pipeline_mode": settings["mode"],
        "concurrency": {key: settings[key] for key in (
            "legacy_concurrency", "download_concurrency", "convert_concurrency",
            "summary_concurrency", "max_in_flight",
        )},
        "scheduler_wait_seconds": round(sum(
            result["timings"]["scheduler_wait_seconds"] for result in results
        ), 6),
        "stage_metrics": stage_metrics,
        "measured_scope": "PDF download + Markdown conversion + per-paper summaries",
    }
    return results, metrics
