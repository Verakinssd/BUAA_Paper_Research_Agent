import asyncio
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from research.pipeline import pipeline_settings, process_papers
from research.reranking import rerank
from research.task_agent import validate_selection
from support import sample_papers


class AsyncTracker:
    def __init__(self):
        self.active = 0
        self.maximum = 0

    async def pause(self, seconds=0.025):
        self.active += 1
        self.maximum = max(self.maximum, self.active)
        try:
            await asyncio.sleep(seconds)
        finally:
            self.active -= 1


class ThreadTracker:
    def __init__(self):
        self.active = 0
        self.maximum = 0
        self.lock = threading.Lock()

    def pause(self, seconds=0.025):
        with self.lock:
            self.active += 1
            self.maximum = max(self.maximum, self.active)
        try:
            time.sleep(seconds)
        finally:
            with self.lock:
                self.active -= 1


class PipelineConcurrencyTests(unittest.IsolatedAsyncioTestCase):
    async def test_reranking_metadata_survives_selection_and_both_pipeline_modes(self):
        async def download(url, path, progress=None):
            path.write_bytes(b"%PDF-test")

        class Agent:
            async def summarize(self, paper, markdown, progress=None):
                if paper["id"] == "P3":
                    raise RuntimeError("test summary failure")
                return "summary"

        environment = {
            "PAPER_RERANKING_METHOD": "local", "RERANKER_TOP_N": "3",
            "RERANKER_FALLBACK": "0", "PIPELINE_MAX_IN_FLIGHT": "2",
        }
        fields = ("base_rank", "base_score", "rerank_score", "rerank_rank", "final_rank")
        for mode in ("legacy", "staged"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as folder, \
                    patch.dict(os.environ, environment), \
                    patch("research.reranking._local_scores", new=AsyncMock(return_value=[0.2, 0.9, 0.5])), \
                    patch("research.pipeline.restore_cached_fulltext", return_value=None), \
                    patch("research.pipeline.store_cached_fulltext"), \
                    patch("research.pipeline.download_pdf", side_effect=download), \
                    patch("research.pipeline.convert_pdf", return_value="[Page 1]\n" + "text " * 30):
                ranked, stats = await rerank("paper retrieval", sample_papers(3))
                self.assertFalse(stats["fallback"])
                self.assertEqual([paper["id"] for paper in ranked], ["P2", "P3", "P1"])
                selected = validate_selection({"selected": [
                    {"id": paper["id"], "reason": "Relevant test evidence"} for paper in ranked[:2]
                ]}, ranked, 2)
                events = []
                results, metrics = await process_papers(
                    selected, Path(folder), Agent(), concurrent=True, mode=mode,
                    trace=lambda event, data: events.append((event, data)),
                )

            self.assertEqual([paper["status"] for paper in results], ["ok", "failed"])
            self.assertEqual(metrics["pipeline_mode"], mode)
            processed = {data["id"]: data for event, data in events if event == "paper_processed"}
            for source, result in zip(selected, results):
                self.assertEqual(result["id"], source["id"])
                for field in fields:
                    self.assertEqual(result[field], source[field])
                    self.assertEqual(processed[source["id"]][field], source[field])

    async def test_staged_limits_each_stage_and_inflight_tasks(self):
        download_tracker = AsyncTracker()
        convert_tracker = ThreadTracker()
        summary_tracker = AsyncTracker()

        async def download(url, path, progress=None):
            await download_tracker.pause()
            path.write_bytes(b"%PDF-test")

        def convert(path):
            convert_tracker.pause()
            return "[Page 1]\n" + "full text " * 20

        class Agent:
            async def summarize(self, paper, markdown, progress=None):
                await summary_tracker.pause()
                return "summary"

        environment = {
            "DOWNLOAD_CONCURRENCY": "3", "CONVERT_CONCURRENCY": "2",
            "SUMMARY_CONCURRENCY": "2", "PIPELINE_MAX_IN_FLIGHT": "4",
        }
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, environment), \
                patch("research.pipeline.restore_cached_fulltext", return_value=None), \
                patch("research.pipeline.store_cached_fulltext"), \
                patch("research.pipeline.download_pdf", side_effect=download), \
                patch("research.pipeline.convert_pdf", side_effect=convert):
            results, metrics = await process_papers(
                sample_papers(8), Path(folder), Agent(), concurrent=True, mode="staged"
            )

        self.assertTrue(all(result["status"] == "ok" for result in results))
        self.assertEqual(metrics["pipeline_mode"], "staged")
        self.assertLessEqual(download_tracker.maximum, 3)
        self.assertLessEqual(convert_tracker.maximum, 2)
        self.assertLessEqual(summary_tracker.maximum, 2)
        self.assertGreater(download_tracker.maximum, 1)
        self.assertGreater(convert_tracker.maximum, 1)
        self.assertGreater(summary_tracker.maximum, 1)

    async def test_legacy_holds_two_whole_paper_slots(self):
        active_papers = 0
        maximum_papers = 0
        completed_summaries = 0
        third_download_saw_completed_summary = None
        started_downloads = 0

        async def download(url, path, progress=None):
            nonlocal active_papers, maximum_papers, third_download_saw_completed_summary, started_downloads
            started_downloads += 1
            if started_downloads == 3:
                third_download_saw_completed_summary = completed_summaries
            active_papers += 1
            maximum_papers = max(maximum_papers, active_papers)
            await asyncio.sleep(0.02)
            path.write_bytes(b"%PDF-test")

        class Agent:
            async def summarize(self, paper, markdown, progress=None):
                nonlocal active_papers, completed_summaries
                await asyncio.sleep(0.02)
                completed_summaries += 1
                active_papers -= 1
                return "summary"

        with tempfile.TemporaryDirectory() as folder, \
                patch.dict(os.environ, {"PIPELINE_MAX_IN_FLIGHT": "6"}, clear=False), \
                patch("research.pipeline.restore_cached_fulltext", return_value=None), \
                patch("research.pipeline.store_cached_fulltext"), \
                patch("research.pipeline.download_pdf", side_effect=download), \
                patch("research.pipeline.convert_pdf", return_value="[Page 1]\n" + "text " * 30):
            results, metrics = await process_papers(
                sample_papers(5), Path(folder), Agent(), concurrent=True, mode="legacy"
            )

        self.assertTrue(all(result["status"] == "ok" for result in results))
        self.assertEqual(metrics["pipeline_mode"], "legacy")
        self.assertEqual(metrics["concurrency"]["legacy_concurrency"], 2)
        self.assertEqual(maximum_papers, 2)
        self.assertGreaterEqual(third_download_saw_completed_summary, 1)

    async def test_default_without_mode_remains_serial(self):
        tracker = AsyncTracker()

        async def download(url, path, progress=None):
            await tracker.pause(0.015)
            path.write_bytes(b"%PDF-test")

        class Agent:
            async def summarize(self, paper, markdown, progress=None):
                return "summary"

        environment = {"PIPELINE_MODE": "", "PIPELINE_MAX_IN_FLIGHT": "4"}
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, environment), \
                patch("research.pipeline.restore_cached_fulltext", return_value=None), \
                patch("research.pipeline.store_cached_fulltext"), \
                patch("research.pipeline.download_pdf", side_effect=download), \
                patch("research.pipeline.convert_pdf", return_value="[Page 1]\n" + "text " * 30):
            _, metrics = await process_papers(sample_papers(3), Path(folder), Agent())

        self.assertEqual(tracker.maximum, 1)
        self.assertFalse(metrics["concurrent"])
        self.assertEqual(metrics["concurrency"]["legacy_concurrency"], 1)

    async def test_recovered_attempt_is_not_a_final_failure(self):
        papers = sample_papers(2)
        papers[0]["pdf_urls"] = ["https://example.test/bad.pdf", "https://example.test/good.pdf"]

        async def download(url, path, progress=None):
            if "bad" in url:
                raise RuntimeError("temporary download failure")
            path.write_bytes(b"%PDF-test")

        class Agent:
            async def summarize(self, paper, markdown, progress=None):
                if paper["id"] == "P2":
                    raise RuntimeError("summary failed")
                return "summary"

        with tempfile.TemporaryDirectory() as folder, \
                patch.dict(os.environ, {"PIPELINE_MAX_IN_FLIGHT": "4"}), \
                patch("research.pipeline.restore_cached_fulltext", return_value=None), \
                patch("research.pipeline.store_cached_fulltext"), \
                patch("research.pipeline.download_pdf", side_effect=download), \
                patch("research.pipeline.convert_pdf", return_value="[Page 1]\n" + "text " * 30):
            results, metrics = await process_papers(
                papers, Path(folder), Agent(), concurrent=True, mode="staged"
            )

        self.assertEqual([result["status"] for result in results], ["ok", "failed"])
        self.assertEqual(metrics["attempted_count"], 2)
        self.assertEqual(metrics["completed_count"], 1)
        self.assertEqual(metrics["failed_count"], 1)
        self.assertEqual(metrics["failure_rate"], 0.5)
        self.assertEqual(metrics["stage_metrics"]["download"]["failures"], 1)
        self.assertEqual(metrics["stage_metrics"]["summary"]["failures"], 1)

    def test_invalid_mode_and_concurrency_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "PIPELINE_MODE"):
            pipeline_settings(True, "unknown")
        with patch.dict(os.environ, {"DOWNLOAD_CONCURRENCY": "0"}):
            with self.assertRaisesRegex(ValueError, "DOWNLOAD_CONCURRENCY"):
                pipeline_settings(True, "staged")


if __name__ == "__main__":
    unittest.main()
