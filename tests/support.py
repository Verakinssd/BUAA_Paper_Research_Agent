"""Test-only substitutes; production code never imports this module."""
from pathlib import Path


def write_test_pdf(path, text="Synthetic test document. " * 20):
    import pymupdf
    with pymupdf.open() as document:
        page = document.new_page()
        page.insert_textbox(pymupdf.Rect(40, 40, 560, 800), text, fontsize=11)
        document.save(str(path))


def sample_papers(count=3):
    return [dict(id=f"P{i}", title=f"TEST paper {i}", abstract="Test metadata only.",
                 url=f"https://example.test/paper-{i}",
                 pdf_urls=[f"https://example.test/paper-{i}.pdf"],
                 sources=["arxiv" if i % 2 else "openalex"], rank_score=1)
            for i in range(1, count + 1)]


async def fake_download(url, path, progress=None):
    write_test_pdf(path)
    if progress:
        size = Path(path).stat().st_size
        progress(size, size)


class FakeTaskAgent:
    async def select(self, idea, papers, k):
        return [dict(paper, selection_reason="Test selection.") for paper in papers[:k]]

    async def summarize(self, paper, markdown, progress=None):
        return "# TEST summary\n\n" + markdown

    async def report(self, idea, results):
        return "# TEST report\n\n" + idea + "\n\n" + "\n".join(p["title"] for p in results)