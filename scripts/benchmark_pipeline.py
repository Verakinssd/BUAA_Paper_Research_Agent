"""Run legacy and staged pipeline benchmarks sequentially on one manifest."""
import argparse
import asyncio
import hashlib
import json
import os
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
from research.benchmarking import build_comparison, render_markdown, write_visualizations
from research.pipeline import process_papers, save_json
from research.task_agent import TaskAgent


MODEL_ENV_NAMES = (
    "LLM_BACKEND", "LLM_BASE_URL", "LLM_MODEL", "LLM_MAX_TOKENS",
    "LLM_CONTEXT_TOKENS", "LLM_TEMPERATURE", "LLM_TOP_P", "LLM_TOP_K",
    "LLM_ENABLE_THINKING", "SUMMARY_CHUNK_CHARS", "SUMMARY_NOTE_MAX_TOKENS",
    "SUMMARY_FINAL_MAX_TOKENS", "PDF_LAYOUT_MODE",
)


@contextmanager
def temporary_environment(updates):
    previous = {key: os.environ.get(key) for key in updates}
    os.environ.update({key: str(value) for key, value in updates.items()})
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def load_manifest(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    papers = value.get("papers") if isinstance(value, dict) else None
    if not isinstance(value.get("name"), str) or not value["name"].strip() or not papers:
        raise ValueError("benchmark manifest 需要非空 name 和 papers")
    seen = set()
    for paper in papers:
        if (
            not isinstance(paper, dict)
            or not isinstance(paper.get("id"), str)
            or not isinstance(paper.get("title"), str)
            or not isinstance(paper.get("pdf_urls"), list)
            or not paper["pdf_urls"]
        ):
            raise ValueError("每篇论文需要 id、title 和非空 pdf_urls")
        if paper["id"] in seen:
            raise ValueError("benchmark manifest 的论文 ID 不得重复")
        seen.add(paper["id"])
    return value


def model_config():
    result = {name: os.getenv(name, "") for name in MODEL_ENV_NAMES}
    base = result.get("LLM_BASE_URL", "")
    if base:
        parsed = urlsplit(base)
        result["LLM_BASE_URL"] = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
    result["llm_retry_policy"] = "transport/429/5xx：最多 3 次，退避 1s/2s"
    result["llm_timeout_seconds"] = 300
    result["pdf_timeout_seconds"] = 60
    for name in ("task_agent.py", "llm.py"):
        content = (PROJECT_ROOT / "research" / name).read_bytes()
        result[name + "_sha256"] = hashlib.sha256(content).hexdigest()
    return result


def compact_paper_result(result):
    return {
        key: result.get(key)
        for key in ("id", "title", "status", "seconds", "error", "evidence", "timings")
        if key in result
    }


async def run_one(manifest, session_root, sequence, repeat, mode, config):
    run_root = session_root / f"{sequence:02d}-{mode}-repeat-{repeat}"
    output = run_root / "artifacts"
    cache = run_root / "cache"
    environment = {
        "PIPELINE_MODE": mode,
        "DOWNLOAD_CONCURRENCY": config["download_concurrency"],
        "CONVERT_CONCURRENCY": config["convert_concurrency"],
        "SUMMARY_CONCURRENCY": config["summary_concurrency"],
        "PIPELINE_MAX_IN_FLIGHT": config["max_in_flight"],
        "PAPER_CACHE_DIR": cache,
    }
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    # Recreate the input list so neither mode can inherit in-memory mutations
    # from a previous round. The canonical manifest and order remain identical.
    papers = json.loads(json.dumps(manifest["papers"], ensure_ascii=False))
    with temporary_environment(environment):
        results, metrics = await process_papers(
            papers, output, TaskAgent(), concurrent=True, mode=mode
        )
    wall_seconds = time.perf_counter() - started
    row = {
        "sequence": sequence,
        "repeat": repeat,
        "mode": mode,
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": round(wall_seconds, 6),
        "output_dir": str(output.resolve()),
        "cache_dir": str(cache.resolve()),
        "metrics": metrics,
        "papers": [compact_paper_result(result) for result in results],
    }
    save_json(run_root / "results.json", results)
    save_json(run_root / "metrics.json", metrics)
    save_json(run_root / "benchmark_run.json", row)
    return row


async def run(args):
    load_dotenv(PROJECT_ROOT / ".env")
    manifest_path = Path(args.manifest).resolve()
    manifest = load_manifest(manifest_path)
    canonical = json.dumps(manifest, ensure_ascii=False, sort_keys=True).encode("utf-8")
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    session_root = (
        Path(args.output_root).resolve()
        / f"benchmark-{stamp}-{time.time_ns() % 1_000_000:06d}"
    )
    session_root.mkdir(parents=True)
    config = {
        "legacy_concurrency": 2,
        "download_concurrency": args.download_concurrency,
        "convert_concurrency": args.convert_concurrency,
        "summary_concurrency": args.summary_concurrency,
        "max_in_flight": args.max_in_flight,
    }
    selected_modes = [args.mode] if args.mode != "both" else ["legacy", "staged"]
    runs = []
    sequence = 0
    for repeat in range(1, args.repeats + 1):
        order = list(selected_modes)
        if len(order) == 2 and repeat % 2 == 0:
            order.reverse()
        for mode in order:
            sequence += 1
            print(
                f"[{sequence}] 开始 {mode}，第 {repeat} 轮（两种模式不会并行运行）",
                flush=True,
            )
            row = await run_one(manifest, session_root, sequence, repeat, mode, config)
            runs.append(row)
            print(
                json.dumps(
                    {
                        "mode": mode,
                        "repeat": repeat,
                        "wall_seconds": row["wall_seconds"],
                        "completed": row["metrics"]["completed_count"],
                        "failed": row["metrics"]["failed_count"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    raw = {
        "schema_version": 1,
        "started_from": str(PROJECT_ROOT),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "manifest_path": str(manifest_path),
        "manifest_sha256": hashlib.sha256(canonical).hexdigest(),
        "manifest": manifest,
        "pipeline_config": config,
        "model_config": model_config(),
        "runs": runs,
        "limitations": [
            "结果受论文站点、API 服务商负载、网络质量和限流策略影响。",
            "arXiv PDF 下载仍严格串行并保持至少 3.1 秒间隔，因此下载并发不会绕过其限速。",
            "每轮使用独立输出与缓存；服务商侧缓存、连接预热或动态路由无法由本脚本清除。",
            "若服务商动态路由实际承载模型，请将这种变化视为外部实验变量。",
            "本报告只测量 PDF 下载、Markdown 转换和逐篇总结，不包含检索、筛选和综合报告生成。",
        ],
    }
    raw["comparison"] = build_comparison(raw)
    save_json(session_root / "benchmark_results.json", raw)
    write_visualizations(raw, session_root)
    (session_root / "benchmark_report.md").write_text(
        render_markdown(raw), encoding="utf-8"
    )
    print("原始数据：" + str(session_root / "benchmark_results.json"))
    print("对比报告：" + str(session_root / "benchmark_report.md"))


def positive_int(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("必须为正整数")
    return parsed


def main():
    parser = argparse.ArgumentParser(description="顺序比较 legacy 与 staged 论文流水线")
    parser.add_argument("--manifest", default=str(PROJECT_ROOT / "benchmark_manifest_10.json"))
    parser.add_argument(
        "--output-root", default=str(PROJECT_ROOT / "outputs" / "benchmarks")
    )
    parser.add_argument("--repeats", type=positive_int, default=3)
    parser.add_argument("--mode", choices=("both", "legacy", "staged"), default="both")
    parser.add_argument("--download-concurrency", type=positive_int, default=4)
    parser.add_argument("--convert-concurrency", type=positive_int, default=1)
    parser.add_argument("--summary-concurrency", type=positive_int, default=2)
    parser.add_argument("--max-in-flight", type=positive_int, default=8)
    parser.add_argument(
        "--acknowledge-api-cost",
        action="store_true",
        help="确认真实 PDF/API 调用可能产生费用并触发外部限流",
    )
    args = parser.parse_args()
    print(
        "警告：真实 Benchmark 会下载公开 PDF，并多次调用当前模型 API；"
        "可能产生费用、消耗额度或触发限流。"
    )
    if not args.acknowledge_api_cost:
        parser.error("请确认风险后增加 --acknowledge-api-cost")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
