import os
import unittest
from unittest.mock import AsyncMock, patch

from research.reranking import rerank


PAPERS = [
    {"id": "P1", "title": "first", "abstract": "one", "rank_score": 0.9},
    {"id": "P2", "title": "second", "abstract": "two", "rank_score": 0.8},
    {"id": "P3", "title": "third", "abstract": "three", "rank_score": 0.7},
]


class RerankingTests(unittest.IsolatedAsyncioTestCase):
    async def test_none_preserves_order_and_records_base_rank(self):
        with self.assertLogs("research.reranking", level="INFO") as logs, patch.dict(
            os.environ, {"PAPER_RERANKING_METHOD": "none"}, clear=False
        ):
            result, stats = await rerank("query", PAPERS)
        self.assertEqual([paper["id"] for paper in result], ["P1", "P2", "P3"])
        self.assertEqual([paper["base_rank"] for paper in result], [1, 2, 3])
        self.assertEqual([paper["final_rank"] for paper in result], [1, 2, 3])
        self.assertEqual(stats["status"], "ok")
        self.assertFalse(stats["fallback"])
        self.assertTrue(any("reranking skipped" in line for line in logs.output))

    async def test_local_scores_reorder_only_top_n(self):
        with self.assertLogs("research.reranking", level="INFO") as logs, patch.dict(os.environ, {
            "PAPER_RERANKING_METHOD": "local",
            "RERANKER_TOP_N": "2",
            "RERANKER_FALLBACK": "1",
        }, clear=False), patch(
            "research.reranking._local_scores",
            new=AsyncMock(return_value=[0.1, 0.9]),
        ):
            result, stats = await rerank("query", PAPERS)
        self.assertEqual([paper["id"] for paper in result], ["P2", "P1", "P3"])
        self.assertEqual(result[0]["rerank_score"], 0.9)
        self.assertNotIn("rerank_score", result[2])
        self.assertEqual(stats["processed_count"], 2)
        self.assertTrue(any("reranking started" in line for line in logs.output))
        self.assertTrue(any("reranking completed" in line for line in logs.output))

    async def test_http_failure_falls_back_to_first_stage(self):
        with self.assertLogs("research.reranking", level="WARNING") as logs, patch.dict(os.environ, {
            "PAPER_RERANKING_METHOD": "http",
            "RERANKER_FALLBACK": "1",
        }, clear=False), patch(
            "research.reranking._http_scores",
            new=AsyncMock(side_effect=RuntimeError("service unavailable")),
        ):
            result, stats = await rerank("query", PAPERS)
        self.assertEqual([paper["id"] for paper in result], ["P1", "P2", "P3"])
        self.assertEqual(stats["status"], "fallback")
        self.assertTrue(stats["fallback"])
        self.assertTrue(any("reranking fallback" in line for line in logs.output))

    async def test_invalid_scores_fall_back(self):
        with patch.dict(os.environ, {
            "PAPER_RERANKING_METHOD": "local",
            "RERANKER_FALLBACK": "1",
        }, clear=False), patch(
            "research.reranking._local_scores",
            new=AsyncMock(return_value=[0.5]),
        ):
            result, stats = await rerank("query", PAPERS)
        self.assertEqual([paper["id"] for paper in result], ["P1", "P2", "P3"])
        self.assertTrue(stats["fallback"])


if __name__ == "__main__":
    unittest.main()
