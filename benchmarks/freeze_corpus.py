"""One-time, networked step: freeze candidates, reference Top-5 and full text.

For each of the 20 ideas: retrieve candidates once, let the configured model pick
a ranked Top-5, make sure every picked paper's full text is in the cache (dropping
and re-selecting around papers without an accessible PDF), and record the result.
The reference Top-k of a query is the first k of its idea's ranked Top-5.

Run: python -m benchmarks.freeze_corpus [--refresh]
"""
import argparse
import asyncio
import os
import shutil
import time
from pathlib import Path
from dotenv import load_dotenv
from research.pipeline import process_papers
from research.sources import retrieve
from research.task_agent import TaskAgent
from benchmarks.build_dataset import QUERIES_PATH, TOP_KS, load_queries
from benchmarks.corpus import Corpus, CORPUS_DIR, query_key
from benchmarks.run_benchmark import StubAgent

MAX_ROUNDS = 3


async def freeze_idea(idea, search_query, papers, work):
    """Select a ranked Top-5 whose full texts are all cached. Returns manifest entry."""
    pool, k = list(papers), max(TOP_KS)
    for round_number in range(1, MAX_ROUNDS + 1):
        if len(pool) < k:
            break
        agent = TaskAgent()
        selected = await agent.select(idea, pool, k)
        results, _ = await process_papers(selected, work, StubAgent(), False)
        failed = {r["id"] for r in results if r["status"] != "ok"}
        if not failed:
            sizes = {r["id"]: len(Path(r["markdown_path"]).read_text(encoding="utf-8")) for r in results}
            return {"search_query": search_query, "rounds": round_number,
                    "ranked_ids": [p["id"] for p in selected],
                    "papers": {p["id"]: {"title": p["title"], "markdown_chars": sizes[p["id"]]}
                               for p in selected}}
        print(f"  第 {round_number} 轮有 {len(failed)} 篇无可用全文，剔除后重选：{sorted(failed)}", flush=True)
        pool = [p for p in pool if p["id"] not in failed]
    raise RuntimeError(f"无法为 {search_query!r} 选出 {k} 篇都有全文的论文")


async def retrieve_complete(search_query, limit, attempts=5, pause=30):
    """Retrieve, but never freeze a pool where a source failed or came back empty.

    arXiv rate-limits (HTTP 429) and drops connections; freezing an OpenAlex-only
    pool would silently make the whole benchmark corpus lopsided.
    """
    for attempt in range(1, attempts + 1):
        papers, sources = await retrieve(search_query, limit)
        bad = {name: s for name, s in sources.items()
               if isinstance(s, dict) and (s.get("status") != "ok" or not s.get("count"))}
        if papers and not bad:
            return papers
        print(f"  检索不完整（第 {attempt}/{attempts} 次）：{bad or '无结果'}", flush=True)
        if attempt < attempts:
            await asyncio.sleep(pause * attempt)
    raise RuntimeError(f"检索始终不完整：{search_query!r}；请稍后重试（已冻结的想法会自动跳过）")


async def freeze(args):
    load_dotenv()
    queries, dataset_hash = load_queries(args.queries)
    corpus = Corpus(args.corpus_dir)
    corpus.cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["PAPER_CACHE_DIR"] = str(corpus.cache_dir)
    manifest = {"dataset_hash": dataset_hash, "ideas": {}}
    if corpus.manifest_path.is_file() and not args.refresh:
        old = corpus.manifest()
        if old.get("dataset_hash") == dataset_hash:
            manifest = old
    manifest.update(created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                    ranking_method=os.getenv("PAPER_RANKING_METHOD") or "default",
                    reference_model=os.getenv("PAPER_MODEL_ID") or os.getenv("LLM_MODEL"))

    ideas = {}
    for q in queries:
        ideas.setdefault(q["search_query"], q["idea"])
    work = corpus.root / "_work"
    for number, (search_query, idea) in enumerate(ideas.items(), 1):
        key = query_key(search_query)
        print(f"[{number}/{len(ideas)}] {search_query}", flush=True)
        if key in manifest["ideas"] and not args.refresh:
            print("  已冻结，跳过")
            continue
        path = corpus.candidates_path(search_query)
        if path.is_file() and not args.refresh:
            papers = corpus.candidates({"search_query": search_query})
        else:
            papers = await retrieve_complete(search_query, args.max_per_source)
            corpus.save_candidates(search_query, papers)
        shutil.rmtree(work, ignore_errors=True)
        manifest["ideas"][key] = await freeze_idea(idea, search_query, papers, work)
        corpus.save_manifest(manifest)
    shutil.rmtree(work, ignore_errors=True)
    corpus.save_manifest(manifest)
    print(f"完成：{len(manifest['ideas'])} 个想法；冻结全文位于 {corpus.cache_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", default=str(QUERIES_PATH))
    parser.add_argument("--corpus-dir", default=str(CORPUS_DIR))
    parser.add_argument("--max-per-source", type=int, default=8)
    parser.add_argument("--refresh", action="store_true", help="忽略已有冻结结果，全部重做")
    asyncio.run(freeze(parser.parse_args()))


if __name__ == "__main__":
    main()
