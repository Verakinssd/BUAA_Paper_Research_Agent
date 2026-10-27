import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from benchmarks import ab_e2e, preflight, run_benchmark
from benchmarks.build_dataset import load_queries

FAKE_MAIN_AGENT = """import json, os, pathlib, sys, time
delay = float(os.environ.get("FAKE_DELAY", "0"))
time.sleep(delay)
if os.environ.get("FAKE_FAIL") == "1":
    sys.exit(3)
run = pathlib.Path(os.environ["PAPER_OUTPUT_ROOT"]) / "run1"
run.mkdir(parents=True)
(run / "cache_dir.txt").write_text(os.environ["PAPER_CACHE_DIR"])
json.dump({"search_seconds": 0.01, "processing_seconds": delay, "server_total_seconds": delay + 0.01,
           "task_success": True, "completed_count": 1}, open(run / "metrics.json", "w"))
"""


def make_repo(root, name):
    repo = Path(root) / name
    repo.mkdir()
    (repo / "main_agent.py").write_text(FAKE_MAIN_AGENT, encoding="utf-8")
    return str(repo)


class AbTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.slow, self.fast = make_repo(self.root, "slow"), make_repo(self.root, "fast")

    def variants(self, **overrides):
        data = [{"label": "base", "repo": self.slow, "env": {"FAKE_DELAY": "0.4"}},
                {"label": "base-control", "repo": self.slow, "env": {"FAKE_DELAY": "0.4"}, "control": True},
                {"label": "fast", "repo": self.fast, "env": {"FAKE_DELAY": "0.05"}}]
        data = overrides.get("data", data)
        path = self.root / "variants.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def args(self, *extra, out="out", variants=None):
        return ab_e2e.parse_args(["--variants", str(variants or self.variants()), "--out", str(self.root / out),
                                  "--limit", "3", "--repeats", "2", "--timeout", "60", *extra])

    def test_variants_are_validated(self):
        with self.assertRaisesRegex(ValueError, "至少"):
            ab_e2e.load_variants(self.variants(data=[{"label": "a", "repo": self.slow}]))
        with self.assertRaisesRegex(ValueError, "唯一"):
            ab_e2e.load_variants(self.variants(data=[{"label": "a", "repo": self.slow}, {"label": "a", "repo": self.slow}]))
        with self.assertRaisesRegex(ValueError, "main_agent.py"):
            ab_e2e.load_variants(self.variants(data=[{"label": "a", "repo": self.slow}, {"label": "b", "repo": str(self.root)}]))
        with self.assertRaisesRegex(ValueError, "基线"):
            ab_e2e.load_variants(self.variants(data=[{"label": "a", "repo": self.slow, "control": True},
                                                     {"label": "b", "repo": self.slow}]))

    def test_plan_pairs_every_variant_with_every_query_in_varying_order(self):
        variants = ab_e2e.load_variants(self.variants())
        queries, _ = load_queries()
        plan = ab_e2e.build_plan(variants, queries[:10], 2, 0)
        self.assertEqual(plan, ab_e2e.build_plan(variants, queries[:10], 2, 0))  # deterministic
        groups = {}
        for repeat, query, variant, position in plan:
            groups.setdefault((repeat, query["qid"]), []).append((position, variant["label"]))
        self.assertEqual(len(groups), 20)
        for members in groups.values():
            self.assertEqual(sorted(p for p, _ in members), [0, 1, 2])
            self.assertEqual(sorted(l for _, l in members), ["base", "base-control", "fast"])
        first = {sorted(m)[0][1] for m in groups.values()}
        self.assertGreater(len(first), 1)  # not always the same variant first

    async def test_run_measures_the_faster_variant_with_fresh_caches_and_a_control(self):
        summary = await ab_e2e.run(self.args())
        out = self.root / "out"
        rows = [json.loads(line) for line in (out / "results.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(rows), 18)
        caches = [(out / "e2e" / r["variant"] / f"r{r['repeat']}-{r['qid']}" / "run1" / "cache_dir.txt").read_text()
                  for r in rows]
        self.assertEqual(len(set(caches)), 18)  # every run had its own empty cache
        self.assertTrue(all("cache" in c for c in caches))
        fast = summary["vs_baseline"]["fast"]
        self.assertEqual(fast["pairs"], 6)
        self.assertLess(fast["mean_diff_seconds"], -0.1)
        self.assertLess(fast["ci95"][1], 0)  # the whole interval is below zero
        self.assertEqual(fast["faster"], 6)
        control = summary["vs_baseline"]["base-control"]
        self.assertLess(abs(control["mean_diff_seconds"]), abs(fast["mean_diff_seconds"]))
        self.assertIsNotNone(summary["noise_floor_seconds"])
        self.assertTrue(summary["validity"]["ok"])
        self.assertEqual(summary["controls"], ["base-control"])
        text = (out / "summary.md").read_text(encoding="utf-8")
        self.assertIn("fast", text)
        self.assertIn("全新 PDF 缓存", text)

    async def test_resume_does_not_repeat_runs_and_refuses_changed_settings(self):
        await ab_e2e.run(self.args())
        before = (self.root / "out" / "results.jsonl").read_text(encoding="utf-8")
        await ab_e2e.run(self.args())
        self.assertEqual(before, (self.root / "out" / "results.jsonl").read_text(encoding="utf-8"))
        changed = ab_e2e.parse_args(["--variants", str(self.variants()), "--out", str(self.root / "out"),
                                     "--limit", "3", "--repeats", "3"])
        with self.assertRaisesRegex(SystemExit, "repeats"):
            await ab_e2e.run(changed)

    async def test_a_failing_variant_is_reported_not_averaged_in(self):
        data = [{"label": "base", "repo": self.slow, "env": {"FAKE_DELAY": "0.05"}},
                {"label": "broken", "repo": self.fast, "env": {"FAKE_FAIL": "1"}}]
        summary = await ab_e2e.run(self.args(variants=self.variants(data=data)))
        self.assertEqual(summary["variants"]["broken"]["success_rate"], 0.0)
        self.assertFalse(summary["validity"]["ok"])
        self.assertEqual(summary["vs_baseline"]["broken"]["pairs"], 0)
        self.assertEqual(summary["vs_baseline"]["broken"]["dropped_pairs"], 6)

    def test_pairs_with_a_failed_run_are_dropped(self):
        rows = [{"variant": "a", "repeat": 0, "qid": "q1", "task_success": True, "query_seconds": 10.0},
                {"variant": "b", "repeat": 0, "qid": "q1", "task_success": True, "query_seconds": 7.0},
                {"variant": "a", "repeat": 0, "qid": "q2", "task_success": False, "query_seconds": 99.0},
                {"variant": "b", "repeat": 0, "qid": "q2", "task_success": True, "query_seconds": 5.0},
                {"variant": "b", "repeat": 0, "qid": "q3", "task_success": True, "query_seconds": 5.0}]
        diffs, ratios, dropped = ab_e2e.paired(rows, "a", "b")
        self.assertEqual((diffs, ratios, dropped), ([-3.0], [0.7], 2))

    def test_missing_control_and_tiny_samples_are_warned_about(self):
        variants = [{"label": "a"}, {"label": "b"}]
        rows = [{"variant": v, "repeat": 0, "qid": "q1", "task_success": True, "query_seconds": s,
                 "stages": {"processing": s}} for v, s in (("a", 5.0), ("b", 4.0))]
        summary = ab_e2e.summarize(rows, variants)
        warnings = ab_e2e.validity(summary, variants)["warnings"]
        self.assertTrue(any("control" in w for w in warnings))
        self.assertTrue(any("有效配对" in w for w in warnings))

    def test_main_stops_when_preflight_fails(self):
        bad = preflight.Report()
        bad.errors.append("连不上模型服务")
        with patch("benchmarks.preflight.check", return_value=bad) as check, patch("benchmarks.ab_e2e.load_dotenv"):
            code = ab_e2e.main(["--variants", str(self.variants()), "--out", str(self.root / "never")])
        self.assertEqual(code, 1)
        self.assertFalse((self.root / "never").exists())
        self.assertTrue(check.call_args.kwargs["full"])


class FreshCacheTests(unittest.IsolatedAsyncioTestCase):
    async def test_plain_e2e_runs_also_get_their_own_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "main_agent.py").write_text(FAKE_MAIN_AGENT, encoding="utf-8")
            query = {"qid": "q9", "topic": "t", "lang": "zh", "top_k": 2, "idea": "想法"}
            with patch.object(run_benchmark, "REPO", Path(tmp)):
                row = await run_benchmark.run_query_e2e(query, Path(tmp) / "out", False, 60)
            cache = (Path(tmp) / "out" / "e2e" / "q9" / "run1" / "cache_dir.txt").read_text()
        self.assertTrue(row["task_success"])
        self.assertTrue(cache.endswith("cache"))
        self.assertIn("q9", cache)
        self.assertEqual(row["metrics"]["completed_count"], 1)


if __name__ == "__main__":
    unittest.main()
