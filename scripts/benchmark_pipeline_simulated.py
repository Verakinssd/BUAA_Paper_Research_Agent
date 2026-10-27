"""Run a deterministic, fully offline benchmark of pipeline scheduling."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.pipeline import process_papers, save_json
from research.simulated_benchmarking import (
    build_simulation_comparison,
    render_simulation_markdown,
    write_simulation_visualizations,
)
from scripts.benchmark_pipeline import load_manifest, temporary_environment


SCENARIOS = {
    "balanced": {
        "label": "均衡流水线",
        "download_seconds": 0.040,
        "convert_seconds": 0.015,
        "summary_seconds": 0.120,
        "description": "三个阶段均有可观耗时，检验流水线重叠收益",
    },
    "api_dominated": {
        "label": "API 主导",
        "download_seconds": 0.010,
        "convert_seconds": 0.005,
        "summary_seconds": 0.200,
        "description": "总结远慢于其他阶段，模拟当前真实系统瓶颈",
    },
    "download_dominated": {
        "label": "下载主导",
        "download_seconds": 0.120,
        "convert_seconds": 0.010,
        "summary_seconds": 0.030,
        "description": "下载占主导，检验下载并发 4 的收益",
    },
}

PAPER_FACTORS = (1.00, 1.15, 0.85, 1.20, 0.90, 1.10, 0.80, 1.25, 0.95, 1.05)


class AsyncTracker:
    def __init__(self):
        self.active = 0
        self.maximum = 0

    async def pause(self, seconds):
        self.active += 1
        self.maximum = max(self.maximum, self.active)
        try:
            await asyncio.sleep(seconds)
        finally:
            self.active -= 1


class ThreadTracker:
    def __init__(self):
        self.active = 0
        self.maximum = 0
        self.lock = threading.Lock()

    def pause(self, seconds):
        with self.lock:
            self.active += 1
            self.maximum = max(self.maximum, self.active)
        try:
            time.sleep(seconds)
        finally:
            with self.lock:
                self.active -= 1


def factor_for(paper_id, papers):
    index = next(index for index, paper in enumerate(papers) if paper["id"] == paper_id)
    return PAPER_FACTORS[index % len(PAPER_FACTORS)]


async def run_once(papers, scenario_name, scenario, mode, repeat, sequence, session_root, config):
    run_root = session_root / "runs" / scenario_name / f"{sequence:03d}-{mode}-repeat-{repeat}"
    output = run_root / "artifacts"
    download_tracker = AsyncTracker()
    convert_tracker = ThreadTracker()
    summary_tracker = AsyncTracker()
    url_to_id = {paper["pdf_urls"][0]: paper["id"] for paper in papers}

    async def fake_download(url, path, progress=None):
        await download_tracker.pause(
            scenario["download_seconds"] * factor_for(url_to_id[url], papers)
        )
        path.write_bytes(b"%PDF-synthetic")
        if progress:
            progress(path.stat().st_size, path.stat().st_size)

    def fake_convert(path):
        convert_tracker.pause(
            scenario["convert_seconds"] * factor_for(path.stem, papers)
        )
        return "[Page 1]\n" + ("deterministic simulated paper text " * 20)

    class FakeAgent:
        async def summarize(self, paper, markdown, progress=None):
            await summary_tracker.pause(
                scenario["summary_seconds"] * factor_for(paper["id"], papers)
            )
            return "# Deterministic simulated summary\n\nNo external API was called."

    environment = {
        "PIPELINE_MODE": mode,
        "DOWNLOAD_CONCURRENCY": str(config["download_concurrency"]),
        "CONVERT_CONCURRENCY": str(config["convert_concurrency"]),
        "SUMMARY_CONCURRENCY": str(config["summary_concurrency"]),
        "PIPELINE_MAX_IN_FLIGHT": str(config["max_in_flight"]),
    }
    started = time.perf_counter()
    with temporary_environment(environment), \
            patch("research.pipeline.restore_cached_fulltext", return_value=None), \
            patch("research.pipeline.store_cached_fulltext", return_value=None), \
            patch("research.pipeline.download_pdf", side_effect=fake_download), \
            patch("research.pipeline.convert_pdf", side_effect=fake_convert):
        results, metrics = await process_papers(
            json.loads(json.dumps(papers)), output, FakeAgent(), concurrent=True, mode=mode
        )
    wall_seconds = time.perf_counter() - started
    if metrics["failed_count"]:
        raise RuntimeError(f"模拟运行不应失败：{scenario_name}/{mode}/repeat-{repeat}")
    return {
        "sequence": sequence,
        "scenario": scenario_name,
        "repeat": repeat,
        "mode": mode,
        "wall_seconds": round(wall_seconds, 6),
        "metrics": metrics,
        "observed_max_concurrency": {
            "download": download_tracker.maximum,
            "convert": convert_tracker.maximum,
            "summary": summary_tracker.maximum,
        },
        "papers": [
            {"id": result["id"], "status": result["status"], "seconds": result["seconds"]}
            for result in results
        ],
    }


async def run(args):
    manifest_path = Path(args.manifest).resolve()
    manifest = load_manifest(manifest_path)
    papers = manifest["papers"]
    if len(papers) != 10:
        raise ValueError("模拟实验固定要求 10 篇论文")
    selected = SCENARIOS if args.scenario == "all" else {args.scenario: SCENARIOS[args.scenario]}
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    session_root = Path(args.output_root).resolve() / f"simulation-{stamp}-{time.time_ns() % 1_000_000:06d}"
    session_root.mkdir(parents=True)
    config = {
        "legacy_concurrency": 2,
        "download_concurrency": args.download_concurrency,
        "convert_concurrency": args.convert_concurrency,
        "summary_concurrency": args.summary_concurrency,
        "max_in_flight": args.max_in_flight,
    }
    runs = []
    sequence = 0
    for scenario_name, scenario in selected.items():
        for repeat in range(1, args.repeats + 1):
            order = ["legacy", "staged"]
            if repeat % 2 == 0:
                order.reverse()
            for mode in order:
                sequence += 1
                row = await run_once(
                    papers, scenario_name, scenario, mode, repeat, sequence, session_root, config
                )
                runs.append(row)
                print(json.dumps({
                    "scenario": scenario_name,
                    "repeat": repeat,
                    "mode": mode,
                    "seconds": row["wall_seconds"],
                    "max_concurrency": row["observed_max_concurrency"],
                }, ensure_ascii=False), flush=True)
    raw = {
        "schema_version": 1,
        "benchmark_kind": "deterministic_offline_simulation",
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "manifest_path": str(manifest_path),
        "manifest_name": manifest["name"],
        "paper_count": len(papers),
        "paper_ids": [paper["id"] for paper in papers],
        "repeats": args.repeats,
        "scenarios": selected,
        "pipeline_config": config,
        "paper_duration_factors": list(PAPER_FACTORS),
        "runs": runs,
    }
    raw["comparison"] = build_simulation_comparison(raw)
    save_json(session_root / "simulation_results.json", raw)
    write_simulation_visualizations(raw, session_root)
    (session_root / "simulation_report.md").write_text(
        render_simulation_markdown(raw), encoding="utf-8"
    )
    print("模拟原始数据：" + str(session_root / "simulation_results.json"))
    print("模拟报告：" + str(session_root / "simulation_report.md"))


def positive_int(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("必须为正整数")
    return parsed


def main():
    parser = argparse.ArgumentParser(description="完全离线、固定延迟的流水线调度模拟实验")
    parser.add_argument("--manifest", default=str(PROJECT_ROOT / "benchmark_manifest_10.json"))
    parser.add_argument(
        "--output-root", default=str(PROJECT_ROOT / "outputs" / "benchmarks" / "simulated")
    )
    parser.add_argument("--repeats", type=positive_int, default=20)
    parser.add_argument("--scenario", choices=("all", *SCENARIOS), default="all")
    parser.add_argument("--download-concurrency", type=positive_int, default=4)
    parser.add_argument("--convert-concurrency", type=positive_int, default=1)
    parser.add_argument("--summary-concurrency", type=positive_int, default=2)
    parser.add_argument("--max-in-flight", type=positive_int, default=8)
    args = parser.parse_args()
    print("离线模拟：不会访问论文网站，不会调用模型 API，也不会产生 API 费用。")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
