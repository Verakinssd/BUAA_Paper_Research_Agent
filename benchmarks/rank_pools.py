"""Raw per-source search results for the ranking evaluation (networked, one-time).

The ranking evaluation replays `research.sources.retrieve` against these frozen lists, so the
real merge / rank / rerank code of whichever checkout is under test runs unchanged and two code
versions always see exactly the same papers. `limit` is deliberately larger than production's
8 per source: `replay(..., limit=8)` slices the production pool out of it, `limit=25` gives a
pool where ranking quality actually matters.

Run: python -m benchmarks.rank_pools [--limit 25]
"""
import argparse
import asyncio
import contextlib
import copy
import gzip
import hashlib
import json
import re
import time
from pathlib import Path
from unittest.mock import patch
from dotenv import load_dotenv
from research import sources
from benchmarks.build_dataset import QUERIES_PATH, load_queries
from benchmarks.corpus import query_key

POOLS_PATH = Path(__file__).with_name("rank_pools.json.gz")
SOURCES = ("arxiv", "openalex")


def paper_key(paper):
    """Stable identity for relevance labels: the normalized title.

    merge_dedup may merge one paper found by both sources into a single record, so ids
    (P1..Pn) and source-specific ids cannot identify it across runs.
    """
    normalized = re.sub(r"[^a-z0-9一-鿿]+", "", (paper.get("title") or "").lower())
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:12]


def unique_ideas(queries):
    """One entry per distinct search_query (the 100 queries are 20 ideas x 5 top_k)."""
    ideas = {}
    for row in queries:
        ideas.setdefault(row["search_query"], {
            "qkey": query_key(row["search_query"]), "search_query": row["search_query"],
            "idea": row["idea"], "topic": row["topic"], "lang": row["lang"]})
    return list(ideas.values())


async def fetch_idea(search_query, limit, attempts=5, pause=30):
    """Both sources, each complete. Never freeze a pool where a source failed or was empty."""
    for attempt in range(1, attempts + 1):
        got, bad = {}, {}
        for name in SOURCES:
            try:
                data = await getattr(sources, name + "_search")(search_query, limit)
            except Exception as exc:
                bad[name] = str(exc)
                continue
            if data:
                got[name] = data
            else:
                bad[name] = "empty"
        if not bad:
            return got
        print(f"  检索不完整（第 {attempt}/{attempts} 次）：{bad}", flush=True)
        if attempt < attempts:
            await asyncio.sleep(pause)
    raise RuntimeError(f"无法为 {search_query!r} 取得两个来源都完整的结果")


def pools_hash(ideas):
    blob = json.dumps(ideas, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def save_pools(pools, path=POOLS_PATH):
    """Reproducible bytes: sorted keys and a zeroed gzip timestamp."""
    pools = dict(pools, hash=pools_hash(pools["ideas"]))
    raw = json.dumps(pools, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    with open(path, "wb") as stream:
        stream.write(gzip.compress(raw, mtime=0))
    return pools["hash"]


def load_pools(path=POOLS_PATH):
    if not Path(path).is_file():
        raise FileNotFoundError(f"缺少 {path}；请先运行 python -m benchmarks.rank_pools（联网），或从仓库取得该文件")
    pools = json.loads(gzip.decompress(Path(path).read_bytes()).decode("utf-8"))
    if pools.get("hash") != pools_hash(pools["ideas"]):
        raise ValueError(f"{path} 内容与其哈希不符，文件可能被改动")
    return pools


@contextlib.contextmanager
def replay(idea, limit):
    """Make research.sources serve this idea's frozen lists, sliced to `limit` per source.

    Fresh copies on every call, because ranking functions annotate the dicts they receive.
    """
    def fake(name):
        async def search(query, requested):
            return copy.deepcopy(idea[name][:min(limit, requested)])
        return search
    with patch.object(sources, "arxiv_search", fake("arxiv")), \
            patch.object(sources, "openalex_search", fake("openalex")):
        yield


async def freeze(limit, queries_path=QUERIES_PATH, out=POOLS_PATH, only=None):
    queries, dataset_hash = load_queries(queries_path)
    ideas = {}
    for index, item in enumerate(unique_ideas(queries), 1):
        if only and item["qkey"] not in only:
            continue
        print(f"[{index}] {item['search_query']}", flush=True)
        got = await fetch_idea(item["search_query"], limit)
        ideas[item["qkey"]] = dict(item, **{name: got[name] for name in SOURCES})
        print("  " + ", ".join(f"{name}={len(got[name])}" for name in SOURCES), flush=True)
    digest = save_pools({"version": 1, "limit": limit, "dataset_hash": dataset_hash,
                         "frozen_at": time.strftime("%Y-%m-%d"), "ideas": ideas}, out)
    return digest, len(ideas)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=25, help="每个来源取多少篇（默认 25，生产为 8）")
    parser.add_argument("--out", default=str(POOLS_PATH))
    args = parser.parse_args()
    load_dotenv()
    digest, count = asyncio.run(freeze(args.limit, out=args.out))
    print(f"已冻结 {count} 个想法的原始检索结果，哈希 {digest} -> {args.out}")


if __name__ == "__main__":
    main()
