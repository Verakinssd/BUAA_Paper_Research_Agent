"""Two real public metadata APIs, normalized to one paper schema."""
import asyncio
import os
import re
import time
import xml.etree.ElementTree as ET
import httpx

_arxiv_lock = asyncio.Lock()
_last_arxiv = 0.0


async def request(client, url, **kwargs):
    """Fetch metadata with short, status-aware retries.

    The main agent already broadens and retries an unsuccessful search. Blindly
    retrying a rate-limited source three times here only blocks the other source
    and previously turned one arXiv 429 into a 41-second delay.
    """
    retry_after_cap = max(0.0, float(os.getenv("SOURCE_RETRY_AFTER_CAP", "2")))
    retry_base = max(0.0, float(os.getenv("SOURCE_RETRY_BASE_SECONDS", "1")))
    max_attempts = 2
    for attempt in range(max_attempts):
        try:
            response = await client.get(url, **kwargs)
        except httpx.TransportError:
            if attempt + 1 == max_attempts:
                raise RuntimeError("论文来源网络请求失败") from None
            await asyncio.sleep(retry_base)
            continue
        if response.status_code == 429:
            raw_retry_after = response.headers.get("retry-after", "").strip()
            try:
                retry_after = float(raw_retry_after)
            except ValueError:
                retry_after = None
            if (attempt == 0 and retry_after is not None
                    and 0 <= retry_after <= retry_after_cap):
                await asyncio.sleep(retry_after)
                continue
            raise RuntimeError("论文来源 HTTP 429（限流，已快速停止重试）")
        if response.status_code in (500, 502, 503, 504) and attempt == 0:
            await asyncio.sleep(retry_base)
            continue
        if response.is_error:
            raise RuntimeError(f"论文来源 HTTP {response.status_code}")
        return response
    raise RuntimeError("论文来源请求未完成")


async def arxiv_search(query, limit):
    global _last_arxiv
    # One connection/request at a time, with >=3 seconds between requests.
    async with _arxiv_lock:
        await asyncio.sleep(max(0, 3.1 - (time.monotonic() - _last_arxiv)))
        proxy = os.getenv("PAPER_PROXY_URL", "").strip() or None
        timeout = max(1.0, float(os.getenv("PAPER_SOURCE_TIMEOUT", "20")))
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True, proxy=proxy,
                                    trust_env=(not proxy and os.getenv("PAPER_TRUST_ENV", "1") == "1")) as client:
            response = await request(client, "https://export.arxiv.org/api/query", params={
                "search_query": 'all:"' + query.replace('"', " ") + '"',
                "max_results": limit, "sortBy": "relevance",
            })
        _last_arxiv = time.monotonic()
    ns = {"a": "http://www.w3.org/2005/Atom", "x": "http://arxiv.org/schemas/atom"}
    papers = []
    for e in ET.fromstring(response.text).findall("a:entry", ns):
        url = e.findtext("a:id", "", ns).replace("http:", "https:")
        aid = url.split("/abs/")[-1]
        if "/abs/" not in url:
            raise ValueError("arXiv API 未返回有效论文")
        papers.append({
            "title": " ".join(e.findtext("a:title", "", ns).split()),
            "abstract": " ".join(e.findtext("a:summary", "", ns).split()),
            "authors": [a.findtext("a:name", "", ns) for a in e.findall("a:author", ns)],
            "year": e.findtext("a:published", "", ns)[:4],
            "doi": e.findtext("x:doi", "", ns), "arxiv_id": aid,
            "url": url, "pdf_urls": ["https://arxiv.org/pdf/" + aid],
            "sources": ["arxiv"],
        })
    return papers


def restore_abstract(index):
    pairs = [(pos, word) for word, positions in (index or {}).items() for pos in positions]
    return " ".join(word for _, word in sorted(pairs))


def cvf_pdf_url(work):
    """Derive the official CVF open-access PDF for CVPR/ICCV/WACV DOI records.

    OpenAlex often exposes only the arXiv copy even though the accepted paper is
    available from the Computer Vision Foundation. The generated URL is merely
    an extra candidate: normal PDF validation and fallback still apply.
    """
    doi = work.get("doi") or ""
    match = re.search(r"10\.1109/(cvpr|iccv|wacv)[^.]*\.(20\d{2})\.", doi, re.I)
    authorships = work.get("authorships") or []
    title = work.get("display_name") or ""
    if not match or not authorships or not title:
        return None
    author = authorships[0].get("author", {}).get("display_name", "").strip()
    surname = author.split()[-1] if author else ""
    surname = re.sub(r"[^A-Za-z0-9-]", "", surname)
    title_slug = re.sub(r"[^A-Za-z0-9-]+", "_", title).strip("_")
    if not surname or not title_slug:
        return None
    venue, year = match.group(1).upper(), match.group(2)
    return (f"https://openaccess.thecvf.com/content/{venue}{year}/papers/"
            f"{surname}_{title_slug}_{venue}_{year}_paper.pdf")


async def openalex_search(query, limit):
    headers = {}
    key = os.getenv("OPENALEX_API_KEY", "")
    if key:
        headers["Authorization"] = "Bearer " + key
    proxy = os.getenv("PAPER_PROXY_URL", "").strip() or None
    timeout = max(1.0, float(os.getenv("PAPER_SOURCE_TIMEOUT", "20")))
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True, proxy=proxy,
                                trust_env=(not proxy and os.getenv("PAPER_TRUST_ENV", "1") == "1")) as client:
        response = await request(client, "https://api.openalex.org/works",
                                 params={"search": query, "per_page": limit}, headers=headers)
    papers = []
    for w in response.json().get("results", []):
        urls = [cvf_pdf_url(w)]
        urls.extend(
            loc["pdf_url"] for loc in w.get("locations", [])
            if loc.get("is_oa") and loc.get("pdf_url")
        )
        urls = list(dict.fromkeys(url for url in urls if url))
        papers.append({
            "title": w.get("display_name") or "", "abstract": restore_abstract(w.get("abstract_inverted_index")),
            "authors": [a.get("author", {}).get("display_name", "") for a in w.get("authorships", [])],
            "year": w.get("publication_year"), "doi": w.get("doi") or "", "arxiv_id": "",
            "url": w.get("doi") or w["id"], "pdf_urls": urls, "sources": ["openalex"],
        })
    return papers


def paper_keys(p):
    keys = set()
    doi = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", p.get("doi", "").lower().strip())
    if doi:
        keys.add("doi:" + doi)
    aid = p.get("arxiv_id", "")
    for url in [p.get("url", "")] + p.get("pdf_urls", []):
        match = re.search(r"arxiv\.org/(?:abs|pdf)/(.+?)(?:\.pdf)?$", url)
        if match:
            aid = match[1]
    if aid:
        keys.add("arxiv:" + re.sub(r"v\d+$", "", aid.lower()))
    title = re.sub(r"[^\w]", "", p.get("title", "").lower())
    if title:
        keys.add("title:" + title)
    return keys


def merge_dedup(papers):
    groups = []
    for paper in papers:
        keys = paper_keys(paper)
        matching = [i for i, (k, _) in enumerate(groups) if k & keys]
        merged = dict(paper)
        merged["sources"] = list(paper["sources"])
        merged["pdf_urls"] = list(paper.get("pdf_urls", []))
        for i in reversed(matching):
            old_keys, old = groups.pop(i)
            keys |= old_keys
            merged["sources"] = sorted(set(merged["sources"] + old["sources"]))
            merged["pdf_urls"] = list(dict.fromkeys(merged["pdf_urls"] + old.get("pdf_urls", [])))
            for field in ("abstract", "doi", "arxiv_id"):
                if not merged.get(field) and old.get(field):
                    merged[field] = old[field]
        groups.append((keys, merged))
    return [p for _, p in groups]


def basic_rank(query, papers):
    tokens = set(re.findall(r"[a-z0-9]+", query.lower()))
    for p in papers:
        title = set(re.findall(r"[a-z0-9]+", p["title"].lower()))
        abstract = set(re.findall(r"[a-z0-9]+", p.get("abstract", "").lower()))
        p["basic_score"] = round((2 * len(tokens & title) + len(tokens & abstract)) / max(1, 3 * len(tokens)), 4)
    ranked = sorted(papers, key=lambda p: p["basic_score"], reverse=True)
    for i, p in enumerate(ranked, 1):
        p["id"] = f"P{i}"
    return ranked


async def retrieve(query, limit=8, concurrent=False):
    stats = {}
    async def one(name, func):
        started = time.perf_counter()
        try:
            data = await func(query, limit)
            stats[name] = {"status": "ok", "count": len(data)}
        except Exception as exc:
            data = []
            stats[name] = {"status": "failed", "count": 0, "error": str(exc)}
        stats[name]["seconds"] = round(time.perf_counter() - started, 3)
        return data
    jobs = [("arxiv", arxiv_search), ("openalex", openalex_search)]
    if concurrent:
        parts = await asyncio.gather(*(one(name, fn) for name, fn in jobs))
    else:
        parts = [await one(name, fn) for name, fn in jobs]
    papers = [p for part in parts for p in part]
    return basic_rank(query, merge_dedup(papers)), stats
