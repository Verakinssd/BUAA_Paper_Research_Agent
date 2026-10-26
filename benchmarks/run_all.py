"""One command for a formal baseline run: corpus -> preflight -> smoke -> full 100 queries.

    python -m benchmarks.run_all --label baseline-q4
    python -m benchmarks.run_all --label baseline-q4-w4 --workers 4     # throughput run

Start the model server first (deploy/serve_gguf.py; LLAMA_PARALLEL >= workers) and set
LLM_TEMPERATURE=0 and LLM_ENABLE_THINKING in .env. The smoke run (3 queries) must produce a
valid result before the full run starts. If the full run is interrupted, rerun the printed
command with the same --out to resume.
"""
import argparse
import asyncio
import sys
import time
from pathlib import Path
from dotenv import load_dotenv
from benchmarks import preflight, prepare_corpus, run_benchmark
from benchmarks.build_dataset import QUERIES_PATH, load_queries
from benchmarks.corpus import Corpus, CORPUS_DIR
from research.task_agent import TaskAgent


def ensure_corpus(corpus_dir, queries_path):
    queries, digest = load_queries(queries_path)
    corpus = Corpus(corpus_dir)
    try:
        corpus.check_ready(queries, digest)
    except RuntimeError:
        print("冻结语料未就绪，从 corpus.tar.gz 解压……", flush=True)
        prepare_corpus.unpack(root=corpus_dir)
        corpus.check_ready(queries, digest)  # still broken -> let the error through


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--label", required=True, help="结果标签，如 baseline-q4")
    p.add_argument("--workers", type=int, default=1, help="同时跑几条查询（吞吐测试）；基线延迟请用默认 1")
    p.add_argument("--concurrent", action="store_true", help="单查询内 2 篇论文并发")
    p.add_argument("--repeats", type=int, choices=range(1, 6), default=1)
    p.add_argument("--out", help="完整运行的输出目录；指向已有目录即续跑（同时跳过冒烟）")
    p.add_argument("--skip-smoke", action="store_true")
    p.add_argument("--limit", type=int, help="只跑前 N 条（演练用，结果不是完整数据集，不能用于对比）")
    p.add_argument("--queries", default=str(QUERIES_PATH))
    p.add_argument("--corpus-dir", default=str(CORPUS_DIR))
    return p.parse_args(argv)


def main(argv=None, agent_factory=TaskAgent):
    args = parse_args(argv)
    load_dotenv()
    ensure_corpus(args.corpus_dir, args.queries)
    report = preflight.check(full=not args.limit, workers=args.workers, concurrent=args.concurrent)
    print(preflight.render(report), flush=True)
    if not report.ok:
        return 1

    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = Path(args.out) if args.out else Path("outputs") / f"bench-{args.label}-{stamp}"
    common = ["--queries", args.queries, "--corpus-dir", args.corpus_dir,
              "--workers", str(args.workers)] + (["--concurrent"] if args.concurrent else [])

    resuming = (out / "results.jsonl").is_file()
    if not (args.skip_smoke or resuming):
        smoke_out = Path("outputs") / f"bench-{args.label}-smoke-{stamp}"
        print("== 冒烟：3 条查询，不计入正式结果 ==", flush=True)
        smoke = asyncio.run(run_benchmark.run(run_benchmark.parse_args(
            common + ["--label", args.label + "-smoke", "--limit", "3", "--warmup", "0",
                      "--out", str(smoke_out)]), agent_factory=agent_factory))
        if not smoke["validity"]["ok"]:
            print("冒烟结果无效，已停止：" + "；".join(smoke["validity"]["reasons"]),
                  file=sys.stderr, flush=True)
            return 2
        print("冒烟通过。", flush=True)

    resume = f"python -m benchmarks.run_all --label {args.label} --out {out}" + \
             (f" --workers {args.workers}" if args.workers != 1 else "") + \
             (" --concurrent" if args.concurrent else "") + \
             (f" --repeats {args.repeats}" if args.repeats != 1 else "") + \
             (f" --limit {args.limit}" if args.limit else "")
    print(f"== 完整运行（若中断，用同一命令续跑）：\n   {resume}", flush=True)
    summary = asyncio.run(run_benchmark.run(run_benchmark.parse_args(
        common + (["--limit", str(args.limit)] if args.limit else [])
        + ["--label", args.label, "--repeats", str(args.repeats), "--out", str(out)]),
        agent_factory=agent_factory))
    if not summary["validity"]["ok"]:
        print("完整运行结果无效：" + "；".join(summary["validity"]["reasons"]), file=sys.stderr, flush=True)
        return 2
    print(f"完成。请保存 {out / 'summary.json'}、{out / 'summary.md'}、{out / 'run_meta.json'}。", flush=True)
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(errors="replace")
    sys.exit(main())
