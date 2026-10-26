import asyncio
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch
from support import FakeTaskAgent, fake_download, sample_papers
import io
import tarfile
from benchmarks import build_dataset, compare, freeze_corpus, prepare_corpus, preflight, run_all, run_benchmark, summary
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

    def test_hash_is_independent_of_line_endings(self):
        import tempfile
        rows, digest = build_dataset.load_queries()
        text = build_dataset.QUERIES_PATH.read_bytes().replace(b"\r\n", b"\n")
        with tempfile.TemporaryDirectory() as tmp:
            crlf = Path(tmp) / "queries.jsonl"
            crlf.write_bytes(text.replace(b"\n", b"\r\n"))
            self.assertEqual(build_dataset.load_queries(crlf)[1], digest)

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


    def test_busy_seconds_merges_overlap_and_ignores_idle_gaps(self):
        rows = [{"started_at": 0, "ended_at": 10}, {"started_at": 5, "ended_at": 12},
                {"started_at": 100, "ended_at": 104}, {"query_seconds": 1}]
        self.assertEqual(summary.busy_seconds(rows), 16.0)
        self.assertEqual(summary.busy_seconds([]), 0.0)
        self.assertIsNone(summary.throughput([{"query_seconds": 1}], 10))
        self.assertEqual(summary.throughput(rows[:3], 160)["queries_per_hour"], 675.0)
        self.assertEqual(summary.throughput(rows[:3], 160)["completion_tps_aggregate"], 10.0)


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

    def test_compare_refuses_different_workers_and_treats_missing_as_one(self):
        with self.assertRaisesRegex(ValueError, "workers"):
            compare.compare(self.make("a", 1.0, workers=1), self.make("b", 1.0, workers=4))
        compare.compare(self.make("a", 1.0), self.make("b", 1.0, workers=1))  # old summary has no key


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


class SyntheticCorpusCase(unittest.IsolatedAsyncioTestCase):
    """Base: a tiny synthetic frozen corpus (no model, no network). Has no tests itself."""

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


class RunnerTests(SyntheticCorpusCase):
    """Frozen-workload runs against the synthetic corpus."""

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


class SlowAgent(FakeTaskAgent):
    """Sleeps like a real model and reports each call through llm.observer."""
    def __init__(self):
        self.llm = type("Llm", (), {"observer": None})()

    async def select(self, idea, papers, k):
        await asyncio.sleep(0.05)
        self.llm.observer(0.05, {"completion_tokens": k}, None, "stop")  # tagged with k
        return await super().select(idea, papers, k)


class WorkersTests(SyntheticCorpusCase):
    """--workers runs several queries at once; every call must land in its own query's row."""

    async def run_with(self, workers, extra=()):
        out = Path(self.tmp.name) / f"out{workers}"
        argv = ["--label", "t", "--limit", "5", "--warmup", "0", "--workers", str(workers),
                "--out", str(out), "--corpus-dir", str(self.corpus.root), *extra]
        with patch("benchmarks.run_benchmark.collect_environment", return_value={}),                 patch("benchmarks.run_benchmark.load_dotenv"),                 patch("research.pipeline.download_pdf", new_callable=AsyncMock):
            result = await run_benchmark.run(run_benchmark.parse_args(argv), agent_factory=SlowAgent)
        rows = [json.loads(line) for line in (out / "results.jsonl").read_text(encoding="utf-8").splitlines()]
        return out, result, rows

    async def test_concurrent_queries_keep_their_own_llm_counters(self):
        _, result, rows = await self.run_with(5)
        self.assertEqual(len(rows), 5)
        for row in rows:
            self.assertEqual(row["llm"]["select"]["completion_tokens"], row["top_k"])
            self.assertEqual(row["llm"]["select"]["calls"], 1)
        self.assertEqual(result["meta"]["workers"], 5)

    async def test_workers_overlap_in_time_and_serial_does_not(self):
        _, serial, serial_rows = await self.run_with(1)
        _, parallel, parallel_rows = await self.run_with(5)
        self.assertGreater(serial["throughput"]["busy_seconds"], 0)
        self.assertLess(parallel["throughput"]["busy_seconds"], sum(r["query_seconds"] for r in parallel_rows))
        self.assertAlmostEqual(serial["throughput"]["busy_seconds"],
                               sum(r["query_seconds"] for r in serial_rows), delta=0.5)
        self.assertIn("同时查询数：5", summary.render_markdown(parallel))

    async def test_resume_refuses_a_different_workers_value(self):
        out, _, _ = await self.run_with(2)
        argv = ["--label", "t", "--limit", "5", "--warmup", "0", "--workers", "3", "--out", str(out),
                "--corpus-dir", str(self.corpus.root)]
        with patch("benchmarks.run_benchmark.collect_environment", return_value={}),                 patch("benchmarks.run_benchmark.load_dotenv"):
            with self.assertRaisesRegex(SystemExit, "workers"):
                await run_benchmark.run(run_benchmark.parse_args(argv), agent_factory=SlowAgent)

    async def test_resume_of_an_old_run_without_workers_key_counts_as_one(self):
        out, _, _ = await self.run_with(1)
        meta_path = out / "run_meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        del meta["workers"]
        meta_path.write_text(json.dumps(meta), encoding="utf-8")
        _, again, rows = await self.run_with(1)  # same --out, so this resumes the old run
        self.assertEqual(again["meta"]["workers"], 1)
        self.assertEqual(len(rows), 5)  # nothing was repeated

    async def test_router_ignores_calls_outside_any_query(self):
        run_benchmark.Router()(1.0, None, None, "stop")  # no collector set: must not raise


class FakeLlamaServer:
    """Just enough of llama-server for the preflight: /health, /props and chat completions."""
    def __init__(self, health=200, n_ctx=8192, slots=2, chat=200):
        outer = self
        self.health, self.n_ctx, self.slots, self.chat = health, n_ctx, slots, chat

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def reply(self, status, payload):
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if self.path == "/health":
                    self.reply(outer.health, {"status": "ok"})
                elif self.path == "/props":
                    self.reply(200, {"default_generation_settings": {"n_ctx": outer.n_ctx},
                                     "total_slots": outer.slots, "model_path": "m/qwen.gguf"})
                else:
                    self.reply(404, {})

            def do_POST(self):
                self.rfile.read(int(self.headers.get("Content-Length", 0)))
                self.reply(outer.chat, {"choices": [{"message": {"content": "OK"}}]})

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.httpd.server_port}/v1"

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


class PreflightTests(unittest.TestCase):
    def serve(self, **kwargs):
        server = FakeLlamaServer(**kwargs)
        self.addCleanup(server.close)
        return server

    def env(self, server, **override):
        env = {"LLM_BASE_URL": server.url, "LLM_MODEL": "qwen-local", "LLM_TEMPERATURE": "0",
               "LLM_ENABLE_THINKING": "0", "LLM_CONTEXT_TOKENS": "8192"}
        env.update(override)
        return env

    def test_good_setup_passes_and_reports_server_facts(self):
        server = self.serve()
        report = preflight.check(self.env(server))
        self.assertTrue(report.ok, report.errors)
        self.assertEqual((report.info["n_ctx_per_slot"], report.info["total_slots"]), (8192, 2))
        self.assertIn("预检通过", preflight.render(report))

    def test_unreachable_server_is_an_error(self):
        server = self.serve()
        env = self.env(server)
        server.close()
        report = preflight.check(env)
        self.assertFalse(report.ok)
        self.assertIn("连不上", report.errors[0])

    def test_loading_server_and_failing_chat_are_errors(self):
        self.assertIn("加载", preflight.check(self.env(self.serve(health=503))).errors[0])
        self.assertIn("试调用失败", preflight.check(self.env(self.serve(chat=500))).errors[0])

    def test_sampling_must_be_pinned_for_full_runs_only(self):
        server = self.serve()
        for override in ({"LLM_TEMPERATURE": ""}, {"LLM_TEMPERATURE": "0.2"}, {"LLM_ENABLE_THINKING": ""}):
            env = self.env(server, **override)
            self.assertFalse(preflight.check(env, full=True).ok, override)
            debug = preflight.check(env, full=False)
            self.assertTrue(debug.ok and debug.warnings, override)

    def test_context_smaller_than_client_budget_is_an_error(self):
        report = preflight.check(self.env(self.serve(n_ctx=4096)))
        self.assertFalse(report.ok)
        self.assertIn("4096", report.errors[0])

    def test_too_few_slots_only_warns(self):
        server = self.serve(slots=2)
        report = preflight.check(self.env(server), workers=4)
        self.assertTrue(report.ok)
        self.assertIn("LLAMA_PARALLEL=4", report.warnings[0])
        self.assertIn("LLAMA_PARALLEL=8", preflight.check(self.env(server), workers=4, concurrent=True).warnings[0])
        self.assertFalse(preflight.check(self.env(server), workers=2).warnings)

    def test_missing_config_is_an_error(self):
        self.assertIn("LLM_BASE_URL", preflight.check({}).errors[0])


class ValidityTests(unittest.TestCase):
    def row(self, ok=True, calls=True):
        llm = {"select": {"calls": 1, "seconds": 1.0, "prompt_tokens": 1, "completion_tokens": 1,
                          "truncated": 0, "prompt_n": 0, "prompt_ms": 0.0, "predicted_n": 0,
                          "predicted_ms": 0.0}} if calls else {}
        return {"qid": "a", "top_k": 1, "lang": "zh", "repeat": 0, "query_seconds": 1.0,
                "task_success": ok, "stages": {}, "llm": llm}

    def test_all_failed_run_is_invalid(self):
        result = summary.summarize([self.row(ok=False)], {"label": "x", "mode": "stage"})
        self.assertFalse(result["validity"]["ok"])
        self.assertIn("成功率为 0", result["validity"]["reasons"][0])
        self.assertIn("结果无效", summary.render_markdown(result))

    def test_stage_run_without_llm_calls_is_invalid_but_e2e_is_not(self):
        stage = summary.summarize([self.row(calls=False)], {"label": "x", "mode": "stage"})
        self.assertIn("没有记录到任何 LLM 调用", stage["validity"]["reasons"][0])
        e2e = summary.summarize([self.row(calls=False)], {"label": "x", "mode": "e2e"})
        self.assertTrue(e2e["validity"]["ok"])

    def test_empty_run_is_invalid_and_partial_failure_only_warns(self):
        self.assertFalse(summary.summarize([], {"label": "x", "mode": "stage"})["validity"]["ok"])
        rows = [self.row()] + [self.row(ok=False)] * 3
        result = summary.summarize(rows, {"label": "x", "mode": "stage"})
        self.assertTrue(result["validity"]["ok"])
        self.assertTrue(result["validity"]["warnings"])

    def test_compare_refuses_an_invalid_summary(self):
        good = CompareTests().make("a", 1.0)
        bad = CompareTests().make("b", 1.0)
        bad["validity"] = {"ok": False, "reasons": ["任务成功率为 0"], "warnings": []}
        with self.assertRaisesRegex(ValueError, "B 的结果无效"):
            compare.compare(good, bad)


class GuardedRunTests(SyntheticCorpusCase):
    """main() exit codes and the one-command run_all, against the synthetic corpus."""

    async def asyncSetUp(self):
        await super().asyncSetUp()
        previous = os.getcwd()
        os.chdir(self.tmp.name)  # run_all writes to a relative outputs/
        self.addCleanup(os.chdir, previous)
        self.ok_report = preflight.Report()
        for target in (patch("benchmarks.preflight.check", return_value=self.ok_report),
                       patch("benchmarks.run_benchmark.load_dotenv"),
                       patch("benchmarks.run_all.load_dotenv"),
                       patch("benchmarks.run_all.ensure_corpus"),
                       patch("benchmarks.run_benchmark.collect_environment", return_value={})):
            target.start()
            self.addCleanup(target.stop)

    def argv(self, *extra):
        return ["--label", "t", "--limit", "3", "--warmup", "0", "--corpus-dir", str(self.corpus.root), *extra]

    async def test_main_exit_codes(self):
        out = Path(self.tmp.name) / "m"
        code = await asyncio.to_thread(run_benchmark.main, self.argv("--out", str(out)), SlowAgent)
        self.assertEqual(code, 0)
        out2 = Path(self.tmp.name) / "m2"  # FakeTaskAgent never reaches the LLM client -> nothing was measured
        code = await asyncio.to_thread(run_benchmark.main, self.argv("--out", str(out2)), FakeTaskAgent)
        self.assertEqual(code, 2)
        self.assertFalse(json.loads((out2 / "summary.json").read_text(encoding="utf-8"))["validity"]["ok"])

    async def test_main_stops_on_failed_preflight_before_creating_output(self):
        bad = preflight.Report()
        bad.errors.append("连不上模型服务")
        out = Path(self.tmp.name) / "never"
        with patch("benchmarks.preflight.check", return_value=bad):
            code = await asyncio.to_thread(run_benchmark.main, self.argv("--out", str(out)), SlowAgent)
        self.assertEqual(code, 1)
        self.assertFalse(out.exists())

    async def test_main_preflight_treats_filtered_runs_as_debug(self):
        with patch("benchmarks.preflight.check", return_value=self.ok_report) as check:
            await asyncio.to_thread(run_benchmark.main, self.argv("--out", str(Path(self.tmp.name) / "d")), SlowAgent)
        self.assertFalse(check.call_args.kwargs["full"])

    async def test_run_all_smokes_then_runs_and_resume_skips_smoke(self):
        out = Path(self.tmp.name) / "full"
        argv = ["--label", "t", "--limit", "3", "--workers", "2", "--out", str(out),
                "--corpus-dir", str(self.corpus.root)]
        self.assertEqual(await asyncio.to_thread(run_all.main, argv, SlowAgent), 0)
        smokes = list((Path(self.tmp.name) / "outputs").glob("bench-t-smoke-*"))
        self.assertEqual(len(smokes), 1)
        self.assertEqual(len((smokes[0] / "results.jsonl").read_text(encoding="utf-8").splitlines()), 3)
        self.assertEqual(json.loads((out / "run_meta.json").read_text(encoding="utf-8"))["workers"], 2)
        self.assertTrue((out / "summary.md").is_file())
        self.assertEqual(await asyncio.to_thread(run_all.main, argv, SlowAgent), 0)  # resume
        self.assertEqual(len(list((Path(self.tmp.name) / "outputs").glob("bench-t-smoke-*"))), 1)

    async def test_run_all_aborts_when_the_smoke_run_is_invalid(self):
        out = Path(self.tmp.name) / "full"
        argv = ["--label", "t", "--limit", "3", "--out", str(out), "--corpus-dir", str(self.corpus.root)]
        self.assertEqual(await asyncio.to_thread(run_all.main, argv, FakeTaskAgent), 2)
        self.assertFalse(out.exists())


class CorpusArchiveTests(SyntheticCorpusCase):
    """pack/unpack on the synthetic corpus."""

    async def test_roundtrip_is_reproducible_and_runs_with_placeholder_pdfs(self):
        tmp = Path(self.tmp.name)
        first, second = tmp / "a.tar.gz", tmp / "b.tar.gz"
        prepare_corpus.pack(self.corpus, first)
        prepare_corpus.pack(self.corpus, second)
        self.assertEqual(first.read_bytes(), second.read_bytes())

        restored = Corpus(tmp / "restored")
        prepare_corpus.unpack(first, restored.root)
        restored.check_ready(self.rows[:2], self.digest)
        pdfs = list(restored.cache_dir.glob("*.pdf"))
        self.assertEqual(len(pdfs), 5)
        self.assertTrue(all(p.read_bytes().startswith(b"%PDF-") for p in pdfs))

        # The unchanged pipeline must accept the placeholders without any download.
        work = tmp / "work"
        work.mkdir()
        with restored.cache_env(), patch("research.pipeline.download_pdf", new_callable=AsyncMock) as download:
            row = await run_benchmark.run_query(self.rows[2], FakeTaskAgent(), restored,
                                                run_benchmark.Collector(), False, work)
        download.assert_not_called()
        self.assertTrue(row["task_success"])

    async def test_unpack_never_clobbers_a_real_pdf(self):
        archive = Path(self.tmp.name) / "a.tar.gz"
        prepare_corpus.pack(self.corpus, archive)
        before = {p.name: p.stat().st_size for p in self.corpus.cache_dir.glob("*.pdf")}
        prepare_corpus.unpack(archive, self.corpus.root)  # same dir that holds real PDFs
        self.assertEqual(before, {p.name: p.stat().st_size for p in self.corpus.cache_dir.glob("*.pdf")})

    async def test_unpack_rejects_unexpected_members(self):
        archive = Path(self.tmp.name) / "evil.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            info = tarfile.TarInfo("../evil.txt")
            info.size = 1
            tar.addfile(info, io.BytesIO(b"x"))
        target = Path(self.tmp.name) / "target"
        with self.assertRaisesRegex(ValueError, "不允许"):
            prepare_corpus.unpack(archive, target)
        self.assertFalse((Path(self.tmp.name) / "evil.txt").exists())
        self.assertFalse(target.exists())


class CommittedArchiveTests(unittest.TestCase):
    def test_committed_archive_covers_all_100_queries(self):
        """Fails if queries.jsonl changes without re-running freeze_corpus + prepare_corpus pack."""
        rows, digest = build_dataset.load_queries()
        with tempfile.TemporaryDirectory() as tmp:
            prepare_corpus.unpack(prepare_corpus.ARCHIVE_PATH, tmp)
            Corpus(tmp).check_ready(rows, digest)


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
