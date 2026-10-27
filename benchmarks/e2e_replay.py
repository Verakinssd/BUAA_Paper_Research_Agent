"""main_agent.py with frozen search results, for end-to-end A/B runs.

Live search is a poor basis for comparing pipeline versions: arXiv is queried with an exact-phrase match,
so the multi-word query the main agent writes often returns nothing, and every run would see different
candidates. This wrapper keeps everything else real - the main agent's tool-calling loop, the MCP server,
the task agent, PDF download, conversion and summaries - but makes the MCP server answer arXiv / OpenAlex
searches from the frozen raw results of the idea named by BENCH_REPLAY_QKEY (benchmarks/rank_pools).

Same CLI as main_agent.py; it must be started with the checkout under test as the working directory:
    BENCH_REPLAY_QKEY=<qkey> python benchmarks/e2e_replay.py --idea "..." --top-k 3 [--concurrent]
"""
import argparse
import asyncio
import copy
import os
import sys
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != HERE]  # keep benchmarks/ off the import path
sys.path.insert(0, os.getcwd())


def replay_searches(qkey):
    """(arxiv_search, openalex_search) replacements serving the frozen lists of one idea."""
    from benchmarks.rank_pools import load_pools
    pools = load_pools()
    if qkey not in pools["ideas"]:
        raise SystemExit(f"BENCH_REPLAY_QKEY={qkey!r} 不在冻结的候选池里")
    idea = pools["ideas"][qkey]

    def make(name):
        async def search(query, limit):
            return copy.deepcopy(idea[name][:limit])
        return search
    return make("arxiv"), make("openalex")


def serve():
    import server
    from research import sources
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", action="store_true")
    parser.add_argument("--concurrent", action="store_true")
    parser.add_argument("--ranking-method", choices=("basic", "bm25", "dense_linear", "hybrid_rrf"))
    args = parser.parse_args()
    server.config.update(concurrent=args.concurrent, ranking_method=args.ranking_method)
    if args.ranking_method:
        os.environ["PAPER_RANKING_METHOD"] = args.ranking_method
    arxiv, openalex = replay_searches(os.environ["BENCH_REPLAY_QKEY"])
    with patch.object(sources, "arxiv_search", arxiv), patch.object(sources, "openalex_search", openalex):
        server.mcp.run(transport="stdio")


def client():
    import main_agent
    original = main_agent.StdioServerParameters

    def through_wrapper(**kwargs):  # start the MCP server as `this file --server <original server args>`
        kwargs["args"] = [str(Path(__file__).resolve()), "--server"] + kwargs["args"][1:]
        return original(**kwargs)
    main_agent.StdioServerParameters = through_wrapper
    parser = argparse.ArgumentParser()
    parser.add_argument("--idea", default="我想研究大语言模型的参数高效微调方法。")
    parser.add_argument("--top-k", type=int, choices=range(1, 6), default=3)
    parser.add_argument("--concurrent", action="store_true")
    parser.add_argument("--ranking-method", choices=("basic", "bm25", "dense_linear", "hybrid_rrf"))
    args = parser.parse_args()
    if args.ranking_method:
        os.environ["PAPER_RANKING_METHOD"] = args.ranking_method
    asyncio.run(main_agent.run(args))


if __name__ == "__main__":
    serve() if "--server" in sys.argv else client()
