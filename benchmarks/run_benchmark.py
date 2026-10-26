"""Local-inference benchmark over the fixed 100-query set.

stage mode (default): select -> per-paper summaries -> report, all inputs frozen
    (candidates, reference Top-k, cached full text), so no network is touched and
    every model/quantization sees exactly the same workload.
e2e mode: spawn main_agent.py per query (includes network); use a small --limit.

Run: python -m benchmarks.run_benchmark --label q4
"""
import argparse
import asyncio
import contextvars
import json
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from dotenv import load_dotenv
from research.pipeline import process_papers, save_json
from research.task_agent import TaskAgent
from benchmarks.build_dataset import QUERIES_PATH, load_queries
from benchmarks.corpus import Corpus, CORPUS_DIR
from benchmarks.summary import render_markdown, summarize

REPO = Path(__file__).resolve().parent.parent
stage_var = contextvars.ContextVar("bench_stage", default="other")
collector_var = contextvars.ContextVar("bench_collector", default=None)
ENV_KEYS = ("PAPER_MODEL_ID", "PAPER_MODEL_LABEL", "PDF_LAYOUT_MODE", "SELECTION_MAX_TOKENS",
            "SUMMARY_CHUNK_CHARS", "SUMMARY_NOTE_MAX_TOKENS", "SUMMARY_FINAL_MAX_TOKENS",
            "REPORT_MAX_TOKENS", "PAPER_RANKING_METHOD")


class StubAgent:
    """Used only to populate the full-text cache without calling an LLM."""
    async def summarize(self, paper, markdown, progress=None):
        return "stub"


class Collector:
    """LLM observer: records every completed call under the current stage."""
    def __init__(self):
        self.calls = []

    def __call__(self, elapsed, usage, timings, finish_reason):
        usage, timings = usage or {}, timings or {}
        self.calls.append({"stage": stage_var.get(), "seconds": elapsed, "finish": finish_reason,
                           "prompt_tokens": usage.get("prompt_tokens") or 0,
                           "completion_tokens": usage.get("completion_tokens") or 0,
                           "prompt_n": timings.get("prompt_n") or 0,
                           "prompt_ms": timings.get("prompt_ms") or 0.0,
                           "predicted_n": timings.get("predicted_n") or 0,
                           "predicted_ms": timings.get("predicted_ms") or 0.0})

    def drain(self):
        calls, self.calls = self.calls, []
        stages = {}
        for c in calls:
            s = stages.setdefault(c["stage"], dict(calls=0, seconds=0.0, prompt_tokens=0,
                                                    completion_tokens=0, truncated=0, prompt_n=0,
                                                    prompt_ms=0.0, predicted_n=0, predicted_ms=0.0))
            s["calls"] += 1
            s["seconds"] += c["seconds"]
            s["truncated"] += c["finish"] == "length"
            for key in ("prompt_tokens", "completion_tokens", "prompt_n", "prompt_ms",
                        "predicted_n", "predicted_ms"):
                s[key] += c[key]
        return {name: {k: round(v, 3) if isinstance(v, float) else v for k, v in s.items()}
                for name, s in stages.items()}


class Router:
    """The one observer installed on the LLM client. With --workers > 1 several queries
    share it, so each call goes to the Collector of the query (asyncio task) that made it."""
    def __call__(self, *call):
        target = collector_var.get()
        if target is not None:
            target(*call)


def shuffled(rows, seed):
    rows = list(rows)
    random.Random(seed).shuffle(rows)
    return rows


def stratified(rows, seed):
    """Shuffled round-robin over top_k so a small --limit still covers every k."""
    groups = {}
    for row in shuffled(rows, seed):
        groups.setdefault(row["top_k"], []).append(row)
    ordered = []
    while any(groups.values()):
        for k in sorted(groups):
            if groups[k]:
                ordered.append(groups[k].pop(0))
    return ordered


def row_key(row):
    return (row["repeat"], row["qid"], bool(row.get("warmup")))


async def run_query(query, agent, corpus, collector, concurrent, work_dir):
    """One frozen-workload query; returns a result row. Never touches the network."""
    candidates = corpus.candidates(query)
    reference = corpus.reference_papers(query)
    reference_ids = [p["id"] for p in reference]
    row = {"qid": query["qid"], "topic": query["topic"], "lang": query["lang"],
           "top_k": query["top_k"]}
    collector_var.set(collector)
    collector.drain()

    stage_var.set("select")
    started = time.perf_counter()
    try:
        live = await agent.select(query["idea"], candidates, query["top_k"])
        row["select_ids"] = [p["id"] for p in live]
        row["select_overlap"] = round(len(set(row["select_ids"]) & set(reference_ids))
                                      / len(reference_ids), 4)
    except Exception as exc:
        row["select_error"] = str(exc)
    select_seconds = time.perf_counter() - started

    stage_var.set("summarize")
    events = []
    started = time.perf_counter()
    results, metrics = await process_papers(reference, work_dir, agent, concurrent,
                                            lambda event, data: events.append((event, data)))
    process_wall = time.perf_counter() - started
    cache_seconds = sum(d.get("seconds", 0) for e, d in events if e == "paper_cache_hit")
    if any(e == "pdf_download_started" for e, _ in events):
        row["network_used"] = True
    summarize_seconds = max(0.0, process_wall - cache_seconds)

    stage_var.set("report")
    report_success, started = False, time.perf_counter()
    if any(r["status"] == "ok" for r in results):
        try:
            report = await agent.report(query["idea"], results)
            (Path(work_dir) / "report.md").write_text(report, encoding="utf-8")
            report_success = True
        except Exception as exc:
            row["report_error"] = str(exc)
    report_seconds = time.perf_counter() - started

    stages = {"select": select_seconds, "summarize": summarize_seconds, "report": report_seconds}
    row.update(
        stages={k: round(v, 3) for k, v in stages.items()},
        query_seconds=round(sum(stages.values()), 3),
        cache_seconds=round(cache_seconds, 3),
        completed_count=metrics["completed_count"], report_success=report_success,
        task_success=(report_success and metrics["completed_count"] == query["top_k"]
                      and not row.get("network_used")),
        paper_errors=[r.get("error") for r in results if r["status"] != "ok"],
        llm=collector.drain())
    shutil.rmtree(Path(work_dir) / "fulltext", ignore_errors=True)
    return row


async def run_query_e2e(query, out_dir, concurrent, timeout):
    """Spawn the real main_agent.py; the whole wall time includes network."""
    root = Path(out_dir) / "e2e" / query["qid"]
    root.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PAPER_OUTPUT_ROOT=str(root), PYTHONIOENCODING="utf-8")
    cmd = [sys.executable, "main_agent.py", "--idea", query["idea"], "--top-k", str(query["top_k"])]
    if concurrent:
        cmd.append("--concurrent")
    row = {"qid": query["qid"], "topic": query["topic"], "lang": query["lang"],
           "top_k": query["top_k"]}
    started = time.perf_counter()
    proc = await asyncio.create_subprocess_exec(
        *cmd, cwd=str(REPO), env=env, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
    try:
        output, _ = await asyncio.wait_for(proc.communicate(), timeout)
    except asyncio.TimeoutError:
        proc.kill()
        output, _ = await proc.communicate()
        row["error"] = f"超时（>{timeout}s）"
    wall = time.perf_counter() - started
    (root / "main.log").write_bytes(output or b"")
    metrics = {}
    found = sorted(root.glob("*/metrics.json"))
    if found:
        metrics = json.loads(found[0].read_text(encoding="utf-8"))
    search = metrics.get("search_seconds", 0.0)
    processing = metrics.get("processing_seconds", 0.0)
    row.update(exit_code=proc.returncode, query_seconds=round(wall, 3),
               stages={"search": search, "processing": processing,
                       "other": round(max(0.0, wall - search - processing), 3)},
               task_success=bool(metrics.get("task_success")) and proc.returncode == 0)
    return row


def collect_environment():
    env = {k: os.environ[k] for k in os.environ if k.startswith("LLM_") and k != "LLM_API_KEY"}
    env.update({k: os.environ[k] for k in ENV_KEYS if k in os.environ})
    info = {"env": env, "python": sys.version.split()[0]}
    try:
        info["git_sha"] = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
                                         capture_output=True, text=True, timeout=10).stdout.strip()
        info["git_dirty"] = bool(subprocess.run(["git", "status", "--porcelain"], cwd=REPO,
                                                capture_output=True, text=True, timeout=10).stdout.strip())
    except Exception:
        pass
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10)
        if gpu.returncode == 0:
            info["gpu"] = gpu.stdout.strip().splitlines()
    except Exception:
        pass
    try:
        import httpx
        base = os.getenv("LLM_BASE_URL", "").rstrip("/").removesuffix("/v1")
        if base:
            props = httpx.get(base + "/props", timeout=5, trust_env=False).json()
            info["server"] = {
                "n_ctx": props.get("default_generation_settings", {}).get("n_ctx"),
                "total_slots": props.get("total_slots"),
                "model": Path(str(props.get("model_path") or props.get("model_alias") or "")).name,
                "build_info": props.get("build_info")}
    except Exception:
        pass
    return info


def read_rows(path):
    if not Path(path).is_file():
        return []
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--label", default="local", help="结果标签，如 q4 / q5")
    p.add_argument("--mode", choices=("stage", "e2e"), default="stage")
    p.add_argument("--out", help="输出目录；指向已有目录并配合已有 results.jsonl 即断点续跑")
    p.add_argument("--queries", default=str(QUERIES_PATH))
    p.add_argument("--corpus-dir", default=str(CORPUS_DIR))
    p.add_argument("--concurrent", action="store_true", help="单查询内 2 篇论文并发")
    p.add_argument("--workers", type=int, default=1,
                   help="同时跑几条查询（服务端 --parallel 需 >= workers，--concurrent 时需 >= 2*workers）。"
                        "用来缩短总时长；单条查询耗时会因争用变长，只能和相同 workers 的结果对比")
    p.add_argument("--repeats", type=int, choices=range(1, 6), default=1)
    p.add_argument("--warmup", type=int, default=None, help="额外预热查询数，不计入统计（stage 默认 3，e2e 默认 0）")
    p.add_argument("--seed", type=int, default=0, help="查询执行顺序的随机种子")
    p.add_argument("--limit", type=int, help="只跑前 N 条（调试用，结果不是完整数据集）")
    p.add_argument("--top-k", type=int, choices=range(1, 6), help="只跑某个 top_k（调试用）")
    p.add_argument("--lang", choices=("zh", "en"), help="只跑某种语言（调试用）")
    p.add_argument("--e2e-timeout", type=int, default=3600)
    return p.parse_args(argv)


async def run(args, agent_factory=TaskAgent):
    load_dotenv()
    all_rows, dataset_hash = load_queries(args.queries)
    selected = [q for q in all_rows if (not args.top_k or q["top_k"] == args.top_k)
                and (not args.lang or q["lang"] == args.lang)]
    if args.limit:
        selected = selected[:args.limit]
    if not selected:
        raise SystemExit("过滤后没有查询")
    if args.workers < 1:
        raise SystemExit("--workers 必须 >= 1")
    full = len(selected) == len(all_rows)
    stage_mode = args.mode == "stage"
    warmup = args.warmup if args.warmup is not None else (3 if stage_mode else 0)

    corpus = Corpus(args.corpus_dir)
    if stage_mode:
        corpus.check_ready(selected, dataset_hash)  # fail before creating any output

    out = Path(args.out) if args.out else Path("outputs") / f"bench-{args.label}-{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    results_path = out / "results.jsonl"
    done = {row_key(r) for r in read_rows(results_path)}

    meta_path = out / "run_meta.json"
    meta = {"label": args.label, "mode": args.mode, "workload": "frozen" if stage_mode else "live",
            "concurrent": args.concurrent, "workers": args.workers, "repeats": args.repeats,
            "warmup": warmup, "seed": args.seed, "dataset_hash": dataset_hash, "full_dataset": full,
            "filters": {"limit": args.limit, "top_k": args.top_k, "lang": args.lang}}
    if meta_path.is_file():
        old = json.loads(meta_path.read_text(encoding="utf-8"))
        old.setdefault("workers", 1)  # written before --workers existed
        for key in ("mode", "dataset_hash", "seed", "repeats", "concurrent", "workers", "filters"):
            if old.get(key) != meta[key]:
                raise SystemExit(f"断点续跑参数与已有 run_meta.json 不一致：{key}")
        meta = old
    else:
        meta.update(started_at=time.strftime("%Y-%m-%d %H:%M:%S"), environment=collect_environment())
        save_json(meta_path, meta)
    slots = ((meta.get("environment") or {}).get("server") or {}).get("total_slots")
    need = args.workers * (2 if args.concurrent else 1)
    if stage_mode and slots and slots < need:
        print(f"警告：服务端只有 {slots} 个并行槽位，少于需要的 {need}；多出的请求会在服务端排队，"
              f"单条查询耗时会被拉长。请用 LLAMA_PARALLEL={need} 重启模型服务。", file=sys.stderr, flush=True)

    agent = collector = None
    previous_cache = os.environ.get("PAPER_CACHE_DIR")
    if stage_mode:
        os.environ["PAPER_CACHE_DIR"] = str(corpus.cache_dir)
        agent = agent_factory()
        if hasattr(getattr(agent, "llm", None), "observer"):
            agent.llm.observer = Router()
    plan = []
    if stage_mode and warmup:
        plan += [(True, 0, q) for q in shuffled(selected, args.seed)[:warmup]]
    for repeat in range(args.repeats):
        order = shuffled(selected, args.seed + repeat) if stage_mode else stratified(selected, args.seed + repeat)
        plan += [(False, repeat, q) for q in order]

    async def execute(index, is_warmup, repeat, query):
        started, started_at = time.perf_counter(), time.time()
        try:
            if stage_mode:
                work = out / "work" / f"{'w' if is_warmup else repeat}-{query['qid']}"
                work.mkdir(parents=True, exist_ok=True)
                row = await run_query(query, agent, corpus, Collector(), args.concurrent, work)
            else:
                row = await run_query_e2e(query, out, args.concurrent, args.e2e_timeout)
        except Exception as exc:
            row = {"qid": query["qid"], "topic": query["topic"], "lang": query["lang"],
                   "top_k": query["top_k"], "error": str(exc), "task_success": False,
                   "query_seconds": round(time.perf_counter() - started, 3), "stages": {}}
        row.update(repeat=repeat, warmup=is_warmup, started_at=round(started_at, 3),
                   ended_at=round(time.time(), 3))
        with results_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[{index}/{len(plan)}] {query['qid']} k={query['top_k']} "
              f"{'warmup ' if is_warmup else ''}{row['query_seconds']}s "
              f"{'ok' if row.get('task_success') else 'FAILED'}", flush=True)

    pending = [(index, is_warmup, repeat, query)
               for index, (is_warmup, repeat, query) in enumerate(plan, 1)
               if (repeat, query["qid"], is_warmup) not in done]
    gate = asyncio.Semaphore(args.workers)

    async def guarded(item):
        async with gate:
            await execute(*item)

    try:
        for item in pending:  # warmup always runs alone, before the measured phase
            if item[1]:
                await execute(*item)
        await asyncio.gather(*(guarded(item) for item in pending if not item[1]))
    finally:
        if previous_cache is None:
            os.environ.pop("PAPER_CACHE_DIR", None)
        else:
            os.environ["PAPER_CACHE_DIR"] = previous_cache

    summary = summarize(read_rows(results_path), dict(
        {k: meta[k] for k in ("label", "mode", "workload", "concurrent", "dataset_hash",
                              "full_dataset", "repeats", "warmup", "seed", "filters")},
        workers=meta.get("workers", 1)))
    save_json(out / "summary.json", summary)
    (out / "summary.md").write_text(render_markdown(summary), encoding="utf-8")
    print(render_markdown(summary))
    print("结果目录：" + str(out))
    return summary


if __name__ == "__main__":
    # Windows consoles default to gbk; never let a summary glyph crash a finished run.
    sys.stdout.reconfigure(errors="replace")
    asyncio.run(run(parse_args()))
