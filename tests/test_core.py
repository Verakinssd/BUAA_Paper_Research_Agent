import asyncio
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch
import httpx
from research.sources import cvf_pdf_url, merge_dedup, paper_keys, restore_abstract
from research.sources import request as source_request
from research.task_agent import TaskAgent, validate_selection
from research.pipeline import convert_pdf, process_papers, restore_cached_fulltext, store_cached_fulltext


def paper(title, doi="", aid="", source="arxiv"):
    return {"id": "P1", "title": title, "doi": doi, "arxiv_id": aid,
            "sources": [source], "pdf_urls": [], "abstract": "", "url": ""}


class CoreTests(unittest.TestCase):
    def test_doi_normalization_preserves_both_sources(self):
        merged = merge_dedup([paper("Title", "10.1/example"),
                              paper("Other title", "https://doi.org/10.1/example", source="openalex")])
        self.assertEqual(len(merged), 1)
        self.assertEqual(set(merged[0]["sources"]), {"arxiv", "openalex"})

    def test_arxiv_versions_deduplicate(self):
        merged = merge_dedup([paper("A", aid="2501.12345v1"), paper("B", aid="2501.12345v2")])
        self.assertEqual(len(merged), 1)
        self.assertIn("arxiv:2501.12345", paper_keys(paper("A", aid="2501.12345v2")))

    def test_transitive_identity_bridge(self):
        merged = merge_dedup([paper("A", "10.1/a"), paper("B", aid="2501.12345"),
                              paper("Bridge", "10.1/a", "2501.12345")])
        self.assertEqual(len(merged), 1)

    def test_abstract_word_positions(self):
        self.assertEqual(restore_abstract({"world": [1, 3], "hello": [0, 2]}), "hello world hello world")

    def test_cvf_open_access_pdf_is_derived_from_proceedings_doi(self):
        work = {
            "doi": "https://doi.org/10.1109/iccv48922.2021.00370",
            "display_name": "Free-form Description Guided 3D Visual Graph Network",
            "authorships": [{"author": {"display_name": "Mingtao Feng"}}],
        }
        self.assertEqual(
            cvf_pdf_url(work),
            "https://openaccess.thecvf.com/content/ICCV2021/papers/"
            "Feng_Free-form_Description_Guided_3D_Visual_Graph_Network_ICCV_2021_paper.pdf",
        )

    def test_non_cvf_work_has_no_derived_pdf(self):
        self.assertIsNone(cvf_pdf_url({"doi": "https://doi.org/10.1000/example"}))

    def test_unknown_llm_id_rejected(self):
        with self.assertRaises(ValueError):
            validate_selection({"selected": [{"id": "hallucinated", "reason": "related"}]}, [paper("A")], 1)

    def test_duplicate_llm_id_rejected(self):
        p1, p2 = paper("A"), dict(paper("B"), id="P2")
        with self.assertRaises(ValueError):
            validate_selection({"selected": [{"id": "P1", "reason": "x"}, {"id": "P1", "reason": "y"}]}, [p1, p2], 2)

    def test_no_pdf_does_not_become_fulltext_summary(self):
        agent = TaskAgent(mock=True)
        agent.mock = False  # No model required: missing-PDF branch must exit before inference.
        with tempfile.TemporaryDirectory() as folder:
            results, metrics = asyncio.run(process_papers([paper("A")], Path(folder), agent))
        self.assertEqual(results[0]["status"], "failed")
        self.assertNotIn("summary", results[0])
        self.assertEqual(metrics["paper_success_rate"], 0)

    def test_fulltext_cache_restores_valid_pdf_and_markdown(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "source.pdf"
            target = root / "target.pdf"
            source.write_bytes(b"%PDF- cached fixture")
            markdown = "[Page 1]\n" + "cache text " * 20
            with patch.dict(os.environ, {"PAPER_CACHE_DIR": str(root / "cache")}):
                store_cached_fulltext("https://example.test/paper.pdf", source, markdown)
                restored = restore_cached_fulltext("https://example.test/paper.pdf", target)
            self.assertEqual(restored, markdown)
            self.assertEqual(target.read_bytes(), source.read_bytes())

    def test_all_equivalent_urls_are_checked_before_any_download(self):
        class Agent:
            mock = False

            async def summarize(self, paper, markdown, progress=None):
                return "cached summary"

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "source.pdf"
            source.write_bytes(b"%PDF- cached fixture")
            markdown = "[Page 1]\n" + "cached paper text " * 20
            cached_url = "https://arxiv.org/pdf/1234.56789"
            item = dict(paper("A"), pdf_urls=["https://publisher.test/a.pdf", cached_url])
            with patch.dict(os.environ, {"PAPER_CACHE_DIR": str(root / "cache")}), \
                    patch("research.pipeline.download_pdf", new_callable=AsyncMock) as download:
                store_cached_fulltext(cached_url, source, markdown)
                results, _ = asyncio.run(process_papers([item], root / "run", Agent()))
            self.assertEqual(results[0]["status"], "ok")
            download.assert_not_awaited()

    def test_legacy_pdf_conversion_uses_direct_page_text(self):
        with tempfile.TemporaryDirectory() as folder, \
                patch.dict(os.environ, {"PDF_LAYOUT_MODE": "legacy"}):
            path = Path(folder) / "fixture.pdf"
            text = convert_pdf(path, "direct extraction " * 20)
        self.assertIn("[Page 1]", text)
        self.assertIn("direct extraction", text)


class SourceRetryTests(unittest.IsolatedAsyncioTestCase):
    async def call(self, responses):
        calls = 0
        async def handler(request):
            nonlocal calls
            response = responses[calls]
            calls += 1
            return response
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with patch("research.sources.asyncio.sleep", new_callable=AsyncMock) as sleep:
                result = await source_request(client, "https://example.test/api")
        return result, calls, sleep

    async def test_429_without_short_retry_after_fails_fast(self):
        async def handler(request):
            return httpx.Response(429)
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with patch("research.sources.asyncio.sleep", new_callable=AsyncMock) as sleep:
                with self.assertRaisesRegex(RuntimeError, "快速停止"):
                    await source_request(client, "https://example.test/api")
        sleep.assert_not_awaited()

    async def test_short_retry_after_is_honored_once(self):
        response, calls, sleep = await self.call([
            httpx.Response(429, headers={"Retry-After": "1"}),
            httpx.Response(200, text="ok"),
        ])
        self.assertEqual(response.text, "ok")
        self.assertEqual(calls, 2)
        sleep.assert_awaited_once_with(1.0)

    async def test_long_retry_after_fails_fast(self):
        async def handler(request):
            return httpx.Response(429, headers={"Retry-After": "60"})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with patch("research.sources.asyncio.sleep", new_callable=AsyncMock) as sleep:
                with self.assertRaisesRegex(RuntimeError, "快速停止"):
                    await source_request(client, "https://example.test/api")
        sleep.assert_not_awaited()

    async def test_503_retries_once_with_short_backoff(self):
        response, calls, sleep = await self.call([
            httpx.Response(503), httpx.Response(200, text="ok")
        ])
        self.assertEqual(response.text, "ok")
        self.assertEqual(calls, 2)
        sleep.assert_awaited_once_with(1.0)


if __name__ == "__main__":
    unittest.main()
