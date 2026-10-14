"""Compare serial/concurrent processing with the SAME already-selected paper list."""
import argparse
import asyncio
import json
import os
import time
from pathlib import Path
from dotenv import load_dotenv
from research.pipeline import process_papers, save_json
from research.task_agent import TaskAgent


async def run(args):
    load_dotenv()
    papers = json.loads(Path(args.selection).read_text(encoding="utf-8"))
    if not papers:
        raise ValueError("selection.json 必须包含已筛选的论文")
    root = Path("outputs") / ("benchmark-" + str(time.time_ns()))
    rows = []
    for repeat in range(args.repeats):
        # Alternate order to reduce the effect of server-side warming.
        for concurrent in ((False, True) if repeat % 2 == 0 else (True, False)):
            out = root / f'{repeat}-{"concurrent" if concurrent else "serial"}'
            # Normal demos reuse downloaded/converted full text, but a benchmark must
            # not let later runs inherit earlier runs' cache entries.
            previous_cache = os.environ.get("PAPER_CACHE_DIR")
            os.environ["PAPER_CACHE_DIR"] = str(out / "_isolated_cache")
            try:
                results, metrics = await process_papers(papers, out, TaskAgent(), concurrent)
            finally:
                if previous_cache is None:
                    os.environ.pop("PAPER_CACHE_DIR", None)
                else:
                    os.environ["PAPER_CACHE_DIR"] = previous_cache
            metrics["repeat"] = repeat
            rows.append(metrics)
            save_json(out / "results.json", results)
            print(json.dumps(metrics, ensure_ascii=False), flush=True)
    save_json(root / "comparison.json", {"rows": rows, "same_ids": [p["id"] for p in papers],
        "note": "Measures PDF+Markdown+single-paper summaries only. No local cache reused. Not full-system latency."})
    print("对比文件：" + str(root / "comparison.json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", required=True)
    parser.add_argument("--repeats", type=int, choices=range(1, 6), default=3)
    asyncio.run(run(parser.parse_args()))
