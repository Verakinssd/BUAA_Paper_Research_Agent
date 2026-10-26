"""Pack / unpack the frozen corpus so a fresh clone can run the benchmark offline.

    python -m benchmarks.prepare_corpus unpack   # normal use: corpus.tar.gz -> benchmarks/corpus/
    python -m benchmarks.prepare_corpus pack     # maintainers, after freeze_corpus

Only the Markdown full text is archived (~3 MB compressed); the ~300 MB of PDFs are not
needed because the benchmark reads the Markdown from the cache. unpack writes a tiny
placeholder PDF next to each Markdown so the unchanged pipeline cache check, which
wants both files, still passes. The archive is byte-for-byte reproducible.
"""
import argparse
import gzip
import hashlib
import io
import json
import re
import tarfile
from pathlib import Path
from benchmarks.build_dataset import QUERIES_PATH, load_queries
from benchmarks.corpus import CORPUS_DIR, Corpus

ARCHIVE_PATH = Path(__file__).with_name("corpus.tar.gz")
PLACEHOLDER_PDF = b"%PDF-1.4\n% placeholder: the full text is in the sibling .md file\n"
ALLOWED_MEMBERS = (re.compile(r"manifest\.json"),
                   re.compile(r"candidates/[0-9a-f]{12}\.json"),
                   re.compile(r"fulltext/[0-9a-f]{64}\.md"))


def _cache_key(url):
    return hashlib.sha256(url.encode("utf-8")).hexdigest()


def _collect(corpus):
    """Return {archive name: bytes} for exactly what the manifest references."""
    manifest = corpus.manifest()
    files = {"manifest.json": corpus.manifest_path.read_bytes()}
    for key, entry in sorted(manifest["ideas"].items()):
        candidates_path = corpus.root / "candidates" / f"{key}.json"
        files[f"candidates/{key}.json"] = candidates_path.read_bytes()
        by_id = {p["id"]: p for p in json.loads(candidates_path.read_text(encoding="utf-8"))}
        for paper_id in entry["ranked_ids"]:
            for url in by_id[paper_id].get("pdf_urls", [])[:3]:
                md = corpus.cache_dir / f"{_cache_key(url)}.md"
                if md.is_file():
                    files[f"fulltext/{md.name}"] = md.read_bytes()
                    break
            else:
                raise RuntimeError(f"缓存中没有 {paper_id}（{entry['search_query']}）的全文，请先运行 freeze_corpus")
    return files


def pack(corpus=None, archive=ARCHIVE_PATH):
    files = _collect(corpus or Corpus())
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", mtime=0, compresslevel=9) as gz:
        with tarfile.open(fileobj=gz, mode="w") as tar:
            for name in sorted(files):
                info = tarfile.TarInfo(name)
                info.size, info.mtime, info.mode = len(files[name]), 0, 0o644
                tar.addfile(info, io.BytesIO(files[name]))
    Path(archive).write_bytes(buffer.getvalue())
    return len(files), len(buffer.getvalue())


def unpack(archive=ARCHIVE_PATH, root=CORPUS_DIR):
    """Restore the corpus under root. Only whitelisted regular members are accepted."""
    root = Path(root)
    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        for member in members:
            if not member.isreg() or not any(p.fullmatch(member.name) for p in ALLOWED_MEMBERS):
                raise ValueError(f"归档里有不允许的条目：{member.name!r}")
        for member in members:
            data = tar.extractfile(member).read()
            if member.name.startswith("fulltext/"):
                md = root / "cache" / Path(member.name).name
                md.parent.mkdir(parents=True, exist_ok=True)
                md.write_bytes(data)
                pdf = md.with_suffix(".pdf")
                if not pdf.exists():  # never clobber a real PDF from freeze_corpus
                    pdf.write_bytes(PLACEHOLDER_PDF)
            else:
                target = root / member.name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
    return len(members)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=("unpack", "pack"), nargs="?", default="unpack")
    parser.add_argument("--archive", default=str(ARCHIVE_PATH))
    parser.add_argument("--corpus-dir", default=str(CORPUS_DIR))
    parser.add_argument("--queries", default=str(QUERIES_PATH))
    args = parser.parse_args()
    if args.action == "pack":
        count, size = pack(Corpus(args.corpus_dir), args.archive)
        print(f"已打包 {count} 个文件，{size / 1e6:.1f} MB -> {args.archive}")
        return
    count = unpack(args.archive, args.corpus_dir)
    queries, dataset_hash = load_queries(args.queries)
    Corpus(args.corpus_dir).check_ready(queries, dataset_hash)
    print(f"语料就绪：解压 {count} 个文件到 {args.corpus_dir}；{len(queries)} 条查询均可离线运行；数据集哈希 {dataset_hash}")


if __name__ == "__main__":
    main()
