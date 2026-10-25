"""Our own MCP service. Run through main_agent.py, not as a separate first step."""
import argparse
import json
import logging
import os
import sys
import time
import uuid
from pathlib import Path
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from research.sources import retrieve
from research.pipeline import process_papers, save_json
from research.task_agent import TaskAgent

load_dotenv()

# On Windows, a lazy numpy import inside a running MCP stdio session hangs forever;
# import it here before the transport starts reading stdin.
import rank_bm25  # noqa: E402,F401

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stderr,
)
log = logging.getLogger("server")

mcp = FastMCP("Paper Research MCP")
state = {}
config = {"concurrent": False}


def trace(output, event, data):
    row = {"event": event, "time": time.time(), **data}
    with (output / "trace.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    log.info(event)


@mcp.tool()
async def search_papers(query: str, max_per_source: int = 8) -> dict:
    """Search arXiv and OpenAlex with one short English topic phrase; merge, deduplicate and rank. Returns run_id for build_report."""
    if not query.strip() or not 1 <= max_per_source <= 20:
        raise ValueError("需要非空检索词，每源候选数量 1~20")
    run_id = uuid.uuid4().hex[:12]
    output = Path(os.getenv("PAPER_OUTPUT_ROOT", "outputs")) / run_id
    output.mkdir(parents=True)
    save_json(output / "run_meta.json", {"run_id": run_id, "query": query,
                                        "created_at": time.time()})
    trace(output, "search_started", {"query": query})
    started = time.perf_counter()
    papers, sources = await retrieve(query, max_per_source, **config)
    elapsed = time.perf_counter() - started
    state[run_id] = {"papers": papers, "output": output, "sources": sources, "search_seconds": elapsed}
    save_json(output / "candidates.json", papers)
    trace(output, "sources_searched", {"sources": sources, "deduplicated_count": len(papers)})
    # Main Agent receives bounded candidate metadata; full records remain server-side.
    candidates = [{k: p.get(k) for k in ("id", "title", "abstract", "year", "sources", "rank_score")}
                  for p in papers]
    for candidate in candidates:
        abstract = candidate.get("abstract") or ""
        candidate["abstract_is_excerpt"] = len(abstract) > 1000
        candidate["abstract"] = abstract[:1000]
    return {"run_id": run_id, "sources": sources, "candidates": candidates}


@mcp.tool()
async def build_report(run_id: str, idea: str, top_k: int = 3) -> dict:
    """Use the internal task Agent to select Top-k, read accessible full PDFs and create summaries plus a research report. Requires run_id from search_papers."""
    if run_id not in state or not 1 <= top_k <= 5:
        raise ValueError("run_id 不存在，或 top_k 不在 1~5")
    run = state[run_id]
    output, papers = run["output"], run["papers"]
    if not papers:
        raise ValueError("没有候选论文，请缩短英文检索词后重新检索")
    started = time.perf_counter()
    metadata = json.loads((output / "run_meta.json").read_text(encoding="utf-8"))
    metadata.update(idea=idea, requested_top_k=top_k, concurrent=config["concurrent"],
                    model_id=os.getenv("PAPER_MODEL_ID"), model_label=os.getenv("PAPER_MODEL_LABEL"))
    save_json(output / "run_meta.json", metadata)
    trace(output, "selection_started", {"requested_top_k": top_k})
    agent = TaskAgent()
    selected = await agent.select(idea, papers, top_k)
    for paper in selected:
        paper["research_idea"] = idea
    save_json(output / "selection.json", selected)
    trace(output, "task_agent_selected", {"selection": [{"id": p["id"], "reason": p["selection_reason"]} for p in selected]})
    results, metrics = await process_papers(selected, output, agent, config["concurrent"],
                                            lambda e, d: trace(output, e, d))
    save_json(output / "topk.json", results)
    trace(output, "report_started", {"completed_count": metrics["completed_count"]})
    if not any(r["status"] == "ok" for r in results):
        report = "# 未完成研究报告\n\n全部 Top-k 全文处理/总结失败；请查看 topk.json 中的错误。"
        report_success = False
    else:
        try:
            report = await agent.report(idea, results)
            report_success = True
        except Exception as exc:
            report = "# 报告生成失败\n\n" + str(exc)
            report_success = False
    (output / "report.md").write_text(report, encoding="utf-8")
    metrics.update(search_seconds=round(run["search_seconds"], 3),
                   server_total_seconds=round(run["search_seconds"] + time.perf_counter() - started, 3),
                   report_success=report_success,
                   requested_top_k=top_k,
                   model_id=os.getenv("PAPER_MODEL_ID"), model_label=os.getenv("PAPER_MODEL_LABEL"),
                   task_success=(report_success and metrics["completed_count"] == len(selected) == top_k
                                 and all(s["status"] == "ok" for s in run["sources"].values()
                                         if isinstance(s, dict))),
                   sources=run["sources"])
    save_json(output / "metrics.json", metrics)
    trace(output, "report_completed", {"metrics": metrics})
    return {"run_id": run_id, "output_dir": str(output), "metrics": metrics,
            "topk": [{k: v for k, v in r.items() if k != "summary"} for r in results],
            "report_preview": report[:3000]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrent", action="store_true")
    parser.add_argument("--ranking-method",
                        choices=("basic", "bm25", "dense_linear", "hybrid_rrf"))
    args = parser.parse_args()
    config.update(concurrent=args.concurrent, ranking_method=args.ranking_method)
    if args.ranking_method:
        os.environ["PAPER_RANKING_METHOD"] = args.ranking_method
    mcp.run(transport="stdio")
