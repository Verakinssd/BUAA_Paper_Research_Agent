import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch
from support import FakeTaskAgent, fake_download, sample_papers
from benchmarks import build_dataset, compare, freeze_corpus, run_benchmark, summary
from benchmarks.corpus import Corpus, query_key
from research.pipeline import process_papers


class DatasetTests(unittest.TestCase):
    def test_committed_dataset_is_the_fixed_100_queries(self):
        rows, digest = build_dataset.load_queries()
        self.assertEqual(len(rows), 100)
        self.assertEqual(len({r["qid"] for r in rows}), 100)
        for k in range(1, 6):
            self.assertEqual(sum(r["top_k"] == k for r in rows), 20)
        self.assertEqual(sum(r["lang"] == "zh" for r in rows), 75)
        self.assertEqual(sum(r["lang"] == "en" for r in rows), 25)
        self.assertEqual(len(digest), 16)

    def test_committed_file_matches_generator(self):
        generated = build_dataset.dumps(build_dataset.build_queries())
        self.assertEqual(build_dataset.QUERIES_PATH.read_text(encoding="utf-8"), generated)

    def test_validate_rejects_wrong_shape(self):
        rows = build_dataset.build_queries()
        with self.assertRaisesRegex(ValueError, "100"):
            build_dataset.validate(rows[:-1])
        broken = [dict(r) for r in rows]
        broken[0]["top_k"] = 5
        with self.assertRaisesRegex(ValueError, "top_k"):
            build_dataset.validate(broken)


class SummaryTests(unittest.TestCase):
    def test_percentile_stats_and_fit(self):
        self.assertEqual(summary.percentile([1, 2, 3, 4, 5], 0.5), 3)
        self.assertAlmostEqual(summary.percentile(list(range(1, 101)), 0.95), 95.05)
        self.assertEqual(summary.stats([]), {"n": 0})
        fit = summary.fit_line([1, 2, 3, 4], [12, 22, 32, 42])
        self.assertEqual((fit["slope_seconds_per_paper"], fit["intercept_seconds"], fit["r2"]), (10.0, 2.0, 1.0))
        self.assertIsNone(summary.fit_line([3, 3], [1, 2]))

    def test_summarize_groups_and_excludes_warmup(self):
        llm = {"summarize": {"calls": 2, "seconds": 4.0, "prompt_tokens": 100, "completion_tokens": 40,
                             "truncated": 0, "prompt_n": 100, "prompt_ms": 500.0,
                             "predicted_n": 40, "predicted_ms": 2000.0}}
        rows = [
            {"qid": "a", "top_k": 1, "lang": "zh", "repeat": 0, "query_seconds": 10.0, "task_success": True,
             "stages": {"select": 1.0, "summarize": 8.0, "report": 1.0}, "llm": llm, "select_overlap": 1.0},
            {"qid": "b", "top_k": 2, "lang": "en", "repeat": 0, "query_seconds": 30.0, "task_success": False,
             "stages": {"select": 2.0, "summarize": 26.0, "report": 2.0}, "llm": llm, "select_overlap": 0.5},
            {"qid": "w", "top_k": 5, "lang": "zh", "repeat": 0, "query_seconds": 999.0, "warmup": True,
             "task_success": True, "stages": {}},
        ]
        result = summary.summarize(rows, {"label": "t"})
        self.assertEqual(result["n_queries"], 2)
        self.assertEqual(result["success_rate"], 0.5)
        self.assertEqual(result["overall"]["mean"], 20.0)
        self.assertEqual(result["by_top_k"]["1"]["mean"], 10.0)
        self.assertEqual(result["by_lang"]["en"]["n"], 1)
        self.assertEqual(result["stages"]["summarize"]["mean_seconds"], 17.0)
        self.assertEqual(result["select_overlap_mean"], 0.75)
        total = result["llm"]["total"]
        self.assertEqual(total["calls"], 4)
        self.assertEqual(total["prefill_tps_server"], 200.0)
        self.assertEqual(total["decode_tps_server"], 20.0)
        self.assertEqual(total["completion_tps_overall"], 10.0)
        self.assertIn("top_k=2", summary.render_markdown(result))


class CompareTests(unittest.TestCase):
    def make(self, label, mean, **meta):
        base = {"label": label, "dataset_hash": "h", "mode": "stage", "workload": "frozen",
                "full_dataset": True, "concurrent": False, "repeats": 1}
        base.update(meta)
        return {"meta": base, "success_rate": 1.0, "overall": {"mean": mean, "median": mean, "p95": mean},
                "by_top_k": {"1": {"mean": mean}}, "by_lang": {}, "stages": {},
                "llm": {"total": {}}, "select_overlap_mean": None}

    def test_compare_ratio(self):
        text = compare.compare(self.make("q4", 10.0), self.make("q5", 15.0))
        self.assertIn("| 查询时间 mean(s) | 10.0 | 15.0 | 1.5 |", text)

    def test_compare_refuses_incomparable_runs(self):
        with self.assertRaisesRegex(ValueError, "dataset_hash"):
            compare.compare(self.make("a", 1.0), self.make("b", 1.0, dataset_hash="other"))
        with self.assertRaisesRegex(ValueError, "完整"):
            compare.compare(self.make("a", 1.0), self.make("b", 1.0, full_dataset=False))
        with self.assertRaisesRegex(ValueError, "mode"):
            compare.compare(self.make("a", 1.0), self.make("b", 1.0, mode="e2e"))


class OrderTests(unittest.TestCase):
    def test_stratified_prefix_covers_every_top_k_and_is_deterministic(self):
        rows, _ = build_dataset.load_queries()
        order = run_benchmark.stratified(rows, 0)
        self.assertEqual(sorted(r["top_k"] for r in order[:5]), [1, 2, 3, 4, 5])
        self.assertEqual(len(order), 100)
        self.assertEqual([r["qid"] for r in order], [r["qid"] for r in run_benchmark.stratified(rows, 0)])
        self.assertNotEqual([r["qid"] for r in run_benchmark.shuffled(rows, 0)],
                            [r["qid"] for r in run_benchmark.shuffled(rows, 1)])

    def test_collector_tags_calls_with_stage(self):
        collector = run_benchmark.Collector()
        run_benchmark.stage_var.set("select")
        collector(1.5, {"prompt_tokens": 10, "completion_tokens": 4}, {"predicted_n": 4, "predicted_ms": 100.0}, "stop")
        run_benchmark.stage_var.set("report")
        collector(2.0, None, None, "length")
        stats = collector.drain()
        self.assertEqual(stats["select"]["completion_tokens"], 4)
        self.assertEqual(stats["select"]["predicted_ms"], 100.0)
        self.assertEqual(stats["report"]["truncated"], 1)
        self.assertEqual(collector.drain(), {})


class RunnerTests(unittest.IsolatedAsyncioTestCase):
    """Frozen-workload runs against a tiny synthetic corpus; no model, no network."""

    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.rows, self.digest = build_dataset.load_queries()
        first = self.rows[0]
        self.corpus = Corpus(root / "corpus")
        papers = sample_papers(6)
        self.corpus.save_candidates(first["search_query"], papers)
        self.corpus.save_manifest({"dataset_hash": self.digest, "ideas": {
            query_key(first["search_query"]): {"search_query": first["search_query"],
                                              "ranked_ids": ["P1", "P2", "P3", "P4", "P5"]}}})
        self.env = patch.dict(os.environ, {"PAPER_CACHE_DIR": str(self.corpus.cache_dir),
                                           "PDF_LAYOUT_MODE": "legacy"})
        self.env.start()
        self.addCleanup(self.env.stop)
        # Populate the cache the way freeze_corpus does, then forbid any download.
        with patch("research.pipeline.download_pdf", new=fake_download):
            await process_papers(papers[:5], root / "freeze", run_benchmark.StubAgent(), False)

    async def test_run_query_uses_only_frozen_inputs(self):
        query = self.rows[2]  # q003: first idea, top_k=3
        work = Path(self.tmp.name) / "work"
        work.mkdir()
        with patch("research.pipeline.download_pdf", new_callable=AsyncMock) as download:
            row = await run_benchmark.run_query(query, FakeTaskAgent(), self.corpus,
                                                run_benchmark.Collector(), False, work)
        download.assert_not_called()
        self.assertTrue(row["task_success"])
        self.assertEqual(row["completed_count"], 3)
        self.assertEqual(row["select_overlap"], 1.0)
        self.assertAlmostEqual(row["query_seconds"], sum(row["stages"].values()), places=2)
        self.assertFalse((work / "fulltext").exists())

    async def test_check_ready_ignores_ambient_cache_dir(self):
        # Regression: the check once read PAPER_CACHE_DIR from .env instead of the corpus.
        with patch.dict(os.environ, {"PAPER_CACHE_DIR": str(Path(self.tmp.name) / "elsewhere")}):
            self.corpus.check_ready(self.rows[:2], self.digest)
            self.assertEqual(os.environ["PAPER_CACHE_DIR"], str(Path(self.tmp.name) / "elsewhere"))

    async def test_missing_fulltext_is_reported_before_running(self):
        for path in self.corpus.cache_dir.iterdir():
            path.unlink()
        with self.assertRaisesRegex(RuntimeError, "冻结缓存"):
            self.corpus.check_ready(self.rows[:2], self.digest)
        with self.assertRaisesRegex(RuntimeError, "版本不一致"):
            self.corpus.check_ready(self.rows[:2], "stale")

    async def test_run_writes_summary_and_resume_does_not_repeat_work(self):
        out = Path(self.tmp.name) / "out"
        argv = ["--label", "t", "--limit", "2", "--warmup", "1", "--out", str(out),
                "--corpus-dir", str(self.corpus.root)]
        quiet = patch("benchmarks.run_benchmark.collect_environment", return_value={})
        with quiet, patch("benchmarks.run_benchmark.load_dotenv"), \
                patch("research.pipeline.download_pdf", new_callable=AsyncMock):
            result = await run_benchmark.run(run_benchmark.parse_args(argv), agent_factory=FakeTaskAgent)
            first = (out / "results.jsonl").read_text(encoding="utf-8").splitlines()
            await run_benchmark.run(run_benchmark.parse_args(argv), agent_factory=FakeTaskAgent)
            second = (out / "results.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(first), 3)  # 1 warmup + 2 measured
        self.assertEqual(first, second)
        self.assertEqual(result["n_queries"], 2)
        self.assertFalse(result["meta"]["full_dataset"])
        self.assertEqual(json.loads((out / "summary.json").read_text(encoding="utf-8"))["n_queries"], 2)
        self.assertTrue((out / "summary.md").is_file())


class FreezeAndE2ETests(unittest.IsolatedAsyncioTestCase):
    async def test_freeze_drops_papers_without_fulltext_and_reselects(self):
        async def flaky_download(url, path, progress=None):
            if url.endswith("paper-2.pdf"):
                raise ValueError("no public pdf")
            await fake_download(url, path, progress)
        with tempfile.TemporaryDirectory() as tmp:
            env = {"PAPER_CACHE_DIR": str(Path(tmp) / "cache"), "PDF_LAYOUT_MODE": "legacy"}
            with patch.dict(os.environ, env), \
                    patch("research.pipeline.download_pdf", new=flaky_download), \
                    patch("benchmarks.freeze_corpus.TaskAgent", new=FakeTaskAgent):
                entry = await freeze_corpus.freeze_idea("idea", "query", sample_papers(6), Path(tmp) / "work")
                with self.assertRaisesRegex(RuntimeError, "无法"):
                    await freeze_corpus.freeze_idea("idea", "query", sample_papers(5), Path(tmp) / "work2")
        self.assertEqual(entry["ranked_ids"], ["P1", "P3", "P4", "P5", "P6"])
        self.assertEqual(entry["rounds"], 2)
        self.assertTrue(all(v["markdown_chars"] > 0 for v in entry["papers"].values()))

    async def test_e2e_row_reads_metrics_written_by_main_agent(self):
        fake = """import json, os, pathlib
run = pathlib.Path(os.environ['PAPER_OUTPUT_ROOT']) / 'run1'
run.mkdir(parents=True)
json.dump({'search_seconds': 1.5, 'processing_seconds': 2.5, 'task_success': True},
          open(run / 'metrics.json', 'w'))
"""
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "main_agent.py").write_text(fake, encoding="utf-8")
            query = {"qid": "q9", "topic": "t", "lang": "zh", "top_k": 2, "idea": "想法"}
            with patch.object(run_benchmark, "REPO", Path(tmp)):
                row = await run_benchmark.run_query_e2e(query, Path(tmp) / "out", False, 60)
        self.assertTrue(row["task_success"])
        self.assertEqual(row["exit_code"], 0)
        self.assertEqual(row["stages"]["search"], 1.5)
        self.assertEqual(row["stages"]["processing"], 2.5)
        self.assertGreaterEqual(row["stages"]["other"], 0)


if __name__ == "__main__":
    unittest.main()
