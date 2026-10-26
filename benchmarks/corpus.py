"""Frozen benchmark corpus: candidates, reference selections and full-text cache.

Layout (benchmarks/corpus/ is git-ignored):
    candidates/<key>.json   ranked candidate papers per search_query
    cache/                  PAPER_CACHE_DIR content (PDF + Markdown)
    manifest.json           dataset hash and the reference Top-5 per idea
"""
import contextlib
import hashlib
import json
import os
from pathlib import Path
from research.pipeline import _cache_paths

CORPUS_DIR = Path(__file__).with_name("corpus")


def query_key(search_query):
    return hashlib.sha256(search_query.encode("utf-8")).hexdigest()[:12]


class Corpus:
    def __init__(self, root=CORPUS_DIR):
        self.root = Path(root)
        self.cache_dir = self.root / "cache"
        self.manifest_path = self.root / "manifest.json"
        self._manifest = None

    def candidates_path(self, search_query):
        return self.root / "candidates" / f"{query_key(search_query)}.json"

    def save_candidates(self, search_query, papers):
        path = self.candidates_path(search_query)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(papers, ensure_ascii=False, indent=2), encoding="utf-8")

    def candidates(self, query):
        path = self.candidates_path(query["search_query"])
        if not path.is_file():
            raise FileNotFoundError(f"缺少候选文件 {path}，请先运行 python -m benchmarks.freeze_corpus")
        return json.loads(path.read_text(encoding="utf-8"))

    def manifest(self):
        if self._manifest is None:
            if not self.manifest_path.is_file():
                raise FileNotFoundError(f"缺少 {self.manifest_path}，请先运行 python -m benchmarks.freeze_corpus")
            self._manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        return self._manifest

    def save_manifest(self, manifest):
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        self._manifest = manifest

    def reference_ids(self, query):
        """Reference Top-k is a prefix of the idea's frozen ranked Top-5."""
        entry = self.manifest()["ideas"].get(query_key(query["search_query"]))
        if not entry or len(entry["ranked_ids"]) < query["top_k"]:
            raise KeyError(f"{query['qid']} 没有冻结的参考选择，请重新运行 freeze_corpus")
        return entry["ranked_ids"][:query["top_k"]]

    def reference_papers(self, query):
        by_id = {p["id"]: p for p in self.candidates(query)}
        return [dict(by_id[pid], selection_reason="frozen reference selection",
                     research_idea=query["idea"]) for pid in self.reference_ids(query)]

    @contextlib.contextmanager
    def cache_env(self):
        """Point PAPER_CACHE_DIR at the frozen cache (pipeline reads it per call)."""
        previous = os.environ.get("PAPER_CACHE_DIR")
        os.environ["PAPER_CACHE_DIR"] = str(self.cache_dir)
        try:
            yield
        finally:
            if previous is None:
                os.environ.pop("PAPER_CACHE_DIR", None)
            else:
                os.environ["PAPER_CACHE_DIR"] = previous

    def paper_cached(self, paper):
        """True if the frozen cache holds this paper's full text (any of its URLs)."""
        with self.cache_env():
            return self._paper_cached(paper)

    def _paper_cached(self, paper):
        for url in paper.get("pdf_urls", [])[:3]:
            pdf, md = _cache_paths(url)
            if pdf.is_file() and md.is_file():
                return True
        return False

    def check_ready(self, queries, dataset_hash):
        """Raise unless every query can run without touching the network."""
        manifest = self.manifest()
        if manifest.get("dataset_hash") != dataset_hash:
            raise RuntimeError("冻结语料与当前 queries.jsonl 版本不一致，请重新运行 freeze_corpus")
        missing = sorted({p["id"] + "@" + q["search_query"] for q in queries
                          for p in self.reference_papers(q) if not self.paper_cached(p)})
        if missing:
            raise RuntimeError("以下论文全文不在冻结缓存中：" + "; ".join(missing[:5]))
