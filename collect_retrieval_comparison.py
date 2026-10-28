"""预览查询；传入 --run 后采集 arXiv 原查询/概念组合查询的结果。"""
import argparse
import asyncio
import json
import os
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
API = "https://export.arxiv.org/api/query"
NS = {"a": "http://www.w3.org/2005/Atom",
      "o": "http://a9.com/-/spec/opensearch/1.1/"}


def phrase(text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("查询词必须为非空字符串")
    return 'all:"' + text.replace('"', ' ').strip() + '"'


def build_queries(item):
    groups = item["concept_groups"]
    if not groups or any(not isinstance(g, list) or not g for g in groups):
        raise ValueError("concept_groups 必须包含非空概念组")
    return {
        "baseline": phrase(item["baseline_query"]),
        "concept": " AND ".join(
            "(" + " OR ".join(phrase(term) for term in group) + ")"
            for group in groups
        ),
    }


def parse_feed(content):
    feed = ET.fromstring(content)
    if feed.tag != "{" + NS["a"] + "}feed":
        raise ValueError("返回内容不是预期的 Atom feed")
    papers = []
    for entry in feed.findall("a:entry", NS):
        url = entry.findtext("a:id", "", NS).replace("http:", "https:")
        if "/abs/" not in url:
            raise ValueError("arXiv 返回错误条目：" + entry.findtext("a:summary", "", NS))
        aid = url.split("/abs/")[-1]
        papers.append({
            "stable_id": "arxiv:" + re.sub(r"v\d+$", "", aid),
            "arxiv_id": aid,
            "title": " ".join(entry.findtext("a:title", "", NS).split()),
            "abstract": " ".join(entry.findtext("a:summary", "", NS).split()),
            "authors": [a.findtext("a:name", "", NS)
                        for a in entry.findall("a:author", NS)],
            "url": url,
        })
    total = feed.findtext("o:totalResults", None, NS)
    return papers, int(total) if total is not None else None


async def main(args):
    load_dotenv(ROOT / ".env")
    items = json.loads((ROOT / "evaluation/retrieval_queries.json").read_text("utf-8"))
    plans = [(item, build_queries(item)) for item in items]
    for item, queries in plans:
        print(f"\n{item['id']}：{item['idea']}", flush=True)
        for method, query in queries.items():
            print(f"{method}: {query}", flush=True)
    if not args.run:
        print("\n仅预览，未联网。确认后加 --run 开始采集。")
        return

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = ROOT / "outputs/retrieval" / stamp
    output.mkdir(parents=True)
    rows = []
    proxy = os.getenv("PAPER_PROXY_URL", "").strip() or None
    timeout = max(1, float(os.getenv("PAPER_SOURCE_TIMEOUT", "20")))
    async with httpx.AsyncClient(
        timeout=timeout, follow_redirects=True, proxy=proxy,
        trust_env=not proxy and os.getenv("PAPER_TRUST_ENV", "0") == "1",
    ) as client:
        for index, (item, queries) in enumerate(plans):
            # 在主题间交替顺序，减少固定先后次序的影响；不是多次重复实验。
            order = ("baseline", "concept") if index % 2 == 0 else ("concept", "baseline")
            for method in order:
                params = {"search_query": queries[method], "start": 0,
                          "max_results": args.limit, "sortBy": "relevance",
                          "sortOrder": "descending"}
                row = {"query_id": item["id"], "idea": item["idea"],
                       "method": method, "params": params, "request_count": 1,
                       "started_at": datetime.now(timezone.utc).isoformat(),
                       "status": "failed", "returned_count": None, "papers": []}
                started = time.perf_counter()
                print(f"\n采集 {item['id']} / {method} …", flush=True)
                try:
                    response = await client.get(API, params=params)
                    row["http_status"] = response.status_code
                    row["request_url"] = str(response.request.url)
                    # 保存原始响应，方便复查；无自动重试，失败不冒充零结果。
                    (output / f"{item['id']}_{method}.xml").write_bytes(response.content)
                    response.raise_for_status()
                    papers, total = parse_feed(response.content)
                    row.update(status="ok", returned_count=len(papers),
                               total_results=total, papers=papers)
                except (httpx.HTTPError, ET.ParseError, ValueError) as exc:
                    row["error"] = str(exc)
                row["request_seconds"] = round(time.perf_counter() - started, 3)
                rows.append(row)
                (output / "results.json").write_text(json.dumps(
                    {"limit": args.limit, "automatic_retries": 0,
                     "timing_scope": "HTTP request plus response parsing; excludes inter-request wait",
                     "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
                print(f"状态={row['status']}，返回={row['returned_count']}，"
                      f"耗时={row['request_seconds']} 秒", flush=True)
                if row["status"] == "failed":
                    print("错误：" + row["error"], flush=True)
                # 请求结束后等待，失败请求也限速，不并发访问 arXiv。
                await asyncio.sleep(3.1)
    print(f"\n结果目录：{output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="实际联网采集")
    parser.add_argument("--limit", type=int, choices=range(1, 21), default=10)
    asyncio.run(main(parser.parse_args()))
