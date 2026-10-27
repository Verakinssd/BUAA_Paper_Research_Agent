import asyncio
import copy
import csv
import json
import math
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from support import FakeTaskAgent
from benchmarks import rank_eval, rank_labels, rank_pools
from benchmarks.build_dataset import load_queries
from benchmarks.rank_labels import label_id
from benchmarks.rank_pools import paper_key
from research import sources

QUERY = "parameter efficient tuning"


def paper(title, abstract="Some abstract.", **extra):
    return dict({"title": title, "abstract": abstract, "year": "2023", "url": "https://x/" + title[:6],
                 "pdf_urls": [], "sources": [], "authors": [], "doi": "", "arxiv_id": ""}, **extra)


def synthetic_pools():
    """Two ideas. In each, lexical overlap with the query orders the papers like the labels do."""
    ideas = {}
    for qkey, query in (("aaaaaaaaaaaa", QUERY), ("bbbbbbbbbbbb", "speculative decoding")):
        words = query.split()
        arxiv = [paper(f"Unrelated study {qkey} {i}", "Nothing about it.") for i in range(3)]
        arxiv += [paper(f"{' '.join(words)} survey {qkey}", f"We study {query}."),
                  paper(f"{words[0]} methods {qkey}", f"{query} is discussed.")]
        openalex = [paper(f"Another unrelated {qkey} {i}", "Different field.") for i in range(3)]
        openalex += [paper(f"{' '.join(words)} benchmark {qkey}", f"{query} {query}.")]
        ideas[qkey] = {"qkey": qkey, "search_query": query, "idea": "想法：" + query, "topic": "t", "lang": "zh",
                       "arxiv": arxiv, "openalex": openalex}
    return {"version": 1, "limit": 5, "dataset_hash": "d", "ideas": ideas}


def synthetic_scores(pools):
    scores = {}
    for qkey, idea in pools["ideas"].items():
        for name in ("arxiv", "openalex"):
            for p in idea[name]:
                title = p["title"].lower()
                scores[label_id(qkey, paper_key(p))] = (3 if "survey" in title or "benchmark" in title
                                                        else 2 if "methods" in title else 0)
    return scores


class PoolTests(unittest.IsolatedAsyncioTestCase):
    def test_paper_key_ignores_case_and_punctuation(self):
        self.assertEqual(paper_key({"title": "Hello, World!"}), paper_key({"title": "hello world"}))
        self.assertNotEqual(paper_key({"title": "Hello World"}), paper_key({"title": "Hello Word"}))

    def test_unique_ideas_are_the_twenty_search_queries(self):
        queries, _ = load_queries()
        ideas = rank_pools.unique_ideas(queries)
        self.assertEqual(len(ideas), 20)
        self.assertEqual(len({i["qkey"] for i in ideas}), 20)

    def test_save_load_is_reproducible_and_detects_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp) / "a.gz", Path(tmp) / "b.gz"
            pools = synthetic_pools()
            rank_pools.save_pools(copy.deepcopy(pools), a)
            rank_pools.save_pools(copy.deepcopy(pools), b)
            self.assertEqual(a.read_bytes(), b.read_bytes())
            self.assertEqual(rank_pools.load_pools(a)["ideas"].keys(), pools["ideas"].keys())
            import gzip
            data = json.loads(gzip.decompress(a.read_bytes()))
            data["ideas"]["aaaaaaaaaaaa"]["arxiv"][0]["title"] = "tampered"
            a.write_bytes(gzip.compress(json.dumps(data).encode()))
            with self.assertRaisesRegex(ValueError, "哈希"):
                rank_pools.load_pools(a)
            with self.assertRaises(FileNotFoundError):
                rank_pools.load_pools(Path(tmp) / "missing.gz")

    async def test_fetch_retries_until_both_sources_are_complete(self):
        calls = {"arxiv": 0}

        async def flaky_arxiv(query, limit):
            calls["arxiv"] += 1
            if calls["arxiv"] == 1:
                raise RuntimeError("HTTP 429")
            return [paper("a")]

        async def openalex(query, limit):
            return [paper("o")]
        with patch.object(sources, "arxiv_search", flaky_arxiv), patch.object(sources, "openalex_search", openalex):
            got = await rank_pools.fetch_idea("q", 5, attempts=3, pause=0)
            self.assertEqual(set(got), {"arxiv", "openalex"})
            self.assertEqual(calls["arxiv"], 2)

        async def empty(query, limit):
            return []
        with patch.object(sources, "arxiv_search", empty), patch.object(sources, "openalex_search", openalex):
            with self.assertRaisesRegex(RuntimeError, "完整"):
                await rank_pools.fetch_idea("q", 5, attempts=2, pause=0)

    async def test_replay_slices_per_source_and_hands_out_copies(self):
        idea = synthetic_pools()["ideas"]["aaaaaaaaaaaa"]
        with rank_pools.replay(idea, 2):
            ranked, stats = await sources.retrieve(QUERY, 8, ranking_method="bm25")
        self.assertEqual(len(ranked), 4)  # 2 per source, the limit asked for is 8 but the pool has 2
        self.assertEqual(stats["arxiv"]["count"], 2)
        ranked[0]["title"] = "mutated"
        self.assertNotIn("mutated", [p["title"] for p in idea["arxiv"] + idea["openalex"]])
        self.assertNotIn("rank_score", idea["arxiv"][0])  # ranking annotated copies only

    @unittest.skipUnless(rank_pools.POOLS_PATH.is_file(), "rank_pools.json.gz has not been frozen")
    def test_committed_pools_cover_the_dataset(self):
        pools = rank_pools.load_pools()
        queries, digest = load_queries()
        self.assertEqual(pools["dataset_hash"], digest)
        self.assertEqual(set(pools["ideas"]), {i["qkey"] for i in rank_pools.unique_ideas(queries)})
        for idea in pools["ideas"].values():
            self.assertTrue(idea["arxiv"] and idea["openalex"], idea["search_query"])


class FakeJudgeLLM:
    def __init__(self, replies=None):
        self.replies, self.calls = list(replies or []), 0

    async def text(self, system, user, max_tokens=None):
        self.calls += 1
        if self.replies:
            return self.replies.pop(0)
        return json.dumps({"score": 3 if "survey" in user.lower() else 0, "reason": "ok"})


class LabelTests(unittest.IsolatedAsyncioTestCase):
    def test_parse_judgement_validates_the_score(self):
        self.assertEqual(rank_labels.parse_judgement('{"score": 2, "reason": "x"}'), (2, "x"))
        self.assertEqual(rank_labels.parse_judgement('```json\n{"score": 0}\n```')[0], 0)
        for bad in ('{"score": 4}', '{"score": true}', '{"score": "2"}', '{"reason": "x"}', "not json"):
            with self.assertRaises((ValueError, KeyError, TypeError)):
                rank_labels.parse_judgement(bad)

    async def test_judge_retries_once_then_gives_up(self):
        llm = FakeJudgeLLM(["garbage", '{"score": 1, "reason": "r"}'])
        self.assertEqual((await rank_labels.judge_one(llm, "idea", paper("t")))[0], 1)
        self.assertEqual(llm.calls, 2)
        with self.assertRaisesRegex(ValueError, "无法解析"):
            await rank_labels.judge_one(FakeJudgeLLM(["x", "y"]), "idea", paper("t"))

    def test_prompt_marks_missing_abstract(self):
        self.assertIn("摘要缺失", rank_labels.judge_prompt("想法", paper("t", abstract="")))
        self.assertNotIn("摘要缺失", rank_labels.judge_prompt("想法", paper("t")))

    async def test_labeling_is_resumable_and_retries_failures(self):
        pools = synthetic_pools()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "labels.jsonl"
            first = await rank_labels.label_all(pools, out, FakeJudgeLLM(), "judge-x", concurrency=3,
                                                limit=5, progress=lambda *_: None)
            self.assertEqual(first["labeled"], 5)
            second = await rank_labels.label_all(pools, out, FakeJudgeLLM(), "judge-x", concurrency=3,
                                                 progress=lambda *_: None)
            total = sum(len(rank_labels.pool_papers(i)) for i in pools["ideas"].values())
            self.assertEqual(second["already"], 5)
            self.assertEqual(first["labeled"] + second["labeled"], total)
            rows = rank_labels.read_jsonl(out)
            self.assertEqual(len({(r["qkey"], r["pkey"]) for r in rows}), len(rows))  # no duplicates
            self.assertEqual(rank_labels.coverage(pools, rank_labels.merged_scores(out, Path(tmp) / "none"))[1], 0)
            # an unparseable judge leaves the pair unlabeled and reported, to be retried next time
            out2 = Path(tmp) / "labels2.jsonl"
            bad = await rank_labels.label_all(pools, out2, FakeJudgeLLM(["x"] * 100), "j", 1, limit=2,
                                              progress=lambda *_: None)
            self.assertEqual((bad["labeled"], len(bad["failed"])), (0, 2))
            self.assertFalse(out2.exists())

    def test_human_labels_override_judge_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            judge, human = Path(tmp) / "j.jsonl", Path(tmp) / "h.jsonl"
            judge.write_text(json.dumps({"qkey": "q", "pkey": "p", "score": 0}) + "\n"
                             + json.dumps({"qkey": "q", "pkey": "r", "score": 1}) + "\n", encoding="utf-8")
            human.write_text(json.dumps({"qkey": "q", "pkey": "p", "score": 3}) + "\n", encoding="utf-8")
            self.assertEqual(rank_labels.merged_scores(judge, human), {"q:p": 3, "q:r": 1})

    def test_sample_export_is_stratified_and_agreement_roundtrips(self):
        pools = synthetic_pools()
        scores = {}
        for idea in pools["ideas"].values():
            for pkey, p in rank_labels.pool_papers(idea).items():
                scores[label_id(idea["qkey"], pkey)] = {"qkey": idea["qkey"], "pkey": pkey,
                                                        "score": 3 if "survey" in p["title"] else 0}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "s.csv"
            self.assertEqual(rank_labels.export_sample(pools, scores, 6, 0, path), 6)
            rows = list(csv.DictReader(open(path, encoding="utf-8-sig", newline="")))
            judged = [int(r["judge_score(标注完再看)"]) for r in rows]
            self.assertIn(3, judged)
            self.assertIn(0, judged)  # both classes present although 0 dominates the pool
            for r in rows:  # a person agrees with the judge on all but one row
                r["human_score(0-3)"] = str(int(r["judge_score(标注完再看)"]))
            rows[0]["human_score(0-3)"] = "1" if rows[0]["human_score(0-3)"] != "1" else "2"
            with open(path, "w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
            human = rank_labels.read_human_csv(path)
            result = rank_labels.agreement(human, scores)
            self.assertEqual(result["n"], 6)
            self.assertAlmostEqual(result["exact"], 5 / 6, places=3)
            rows[1]["human_score(0-3)"] = "9"
            with open(path, "w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "0-3"):
                rank_labels.read_human_csv(path)

    def test_quadratic_kappa(self):
        self.assertEqual(rank_labels.quadratic_kappa([(0, 0), (1, 1), (2, 2), (3, 3)]), 1.0)
        self.assertLess(rank_labels.quadratic_kappa([(0, 3), (3, 0), (1, 2), (2, 1)]), 0)
        self.assertIsNone(rank_labels.quadratic_kappa([(2, 2), (2, 2)]))  # no variation: undefined
        self.assertIsNone(rank_labels.quadratic_kappa([]))

    def test_judge_can_use_a_different_endpoint(self):
        env = {"LLM_BASE_URL": "http://main/v1", "LLM_MODEL": "main", "JUDGE_BASE_URL": "http://judge/v1",
               "JUDGE_MODEL": "big-judge"}
        with patch.dict(os.environ, {"LLM_ALLOW_NO_KEY": "1"}):
            llm, name = rank_labels.make_judge_llm(env)
        self.assertEqual((llm.base, llm.model, name), ("http://judge/v1", "big-judge", "big-judge"))


class MetricTests(unittest.TestCase):
    def test_ndcg_precision_mrr_recall(self):
        pool = [3, 2, 0, 0]
        self.assertEqual(rank_eval.ndcg_at([3, 2, 0, 0], 3, pool), 1.0)
        worst = rank_eval.ndcg_at([0, 0, 2, 3], 4, pool)
        expected = (3 / math.log2(4) + 7 / math.log2(5)) / (7 + 3 / math.log2(3))  # gains 2^s-1: 0,0,3,7
        self.assertAlmostEqual(worst, expected)
        self.assertIsNone(rank_eval.ndcg_at([0, 0], 3, [0, 0]))
        self.assertEqual(rank_eval.precision_at([3, 0, 2], 3), 2 / 3)
        self.assertEqual(rank_eval.precision_at([3], 3), 1 / 3)  # a short list is not rewarded
        self.assertEqual(rank_eval.reciprocal_rank([0, 1, 2]), 1 / 3)
        self.assertEqual(rank_eval.reciprocal_rank([0, 1]), 0.0)
        self.assertEqual(rank_eval.recall_at([3, 0, 2], 1, [3, 0, 2]), 0.5)
        self.assertIsNone(rank_eval.recall_at([0], 1, [0]))

    def test_bootstrap_is_deterministic_and_brackets_the_mean(self):
        diffs = [0.1, 0.2, 0.15, 0.3, 0.05, 0.25]
        low, high = rank_eval.paired_bootstrap(diffs, 2000)
        self.assertEqual((low, high), rank_eval.paired_bootstrap(diffs, 2000))
        self.assertLess(low, sum(diffs) / len(diffs))
        self.assertGreater(high, sum(diffs) / len(diffs))
        self.assertGreater(low, 0)  # all positive: the interval excludes zero
        self.assertEqual(rank_eval.paired_bootstrap([0.0, 0.0, 0.0], 500), (0.0, 0.0))


class EvaluateTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.pools = synthetic_pools()
        self.scores = synthetic_scores(self.pools)

    async def evaluate(self, **kwargs):
        defaults = dict(limit=5, method="bm25", progress=lambda *_: None)
        return await rank_eval.evaluate(self.pools, self.scores, **dict(defaults, **kwargs))

    async def test_a_lexical_ranker_finds_the_labelled_relevant_papers(self):
        ideas, issues = await self.evaluate()
        self.assertEqual(issues, {"unlabeled": 0, "rerank_failed": [], "select_errors": 0})
        for idea in ideas:
            self.assertEqual(idea["n_pool"], 9)
            self.assertEqual(idea["n_relevant"], 3)
            self.assertEqual(idea["metrics"]["p@3"], 1.0)
            self.assertEqual(idea["metrics"]["mrr"], 1.0)
            self.assertAlmostEqual(idea["metrics"]["ndcg@3"], 1.0)

    async def test_worse_labels_lower_the_metrics(self):
        flipped = {k: (3 if v == 0 else 0) for k, v in self.scores.items()}
        good, _ = await self.evaluate()
        bad, _ = await rank_eval.evaluate(self.pools, flipped, limit=5, method="bm25", progress=lambda *_: None)
        self.assertGreater(good[0]["metrics"]["ndcg@5"], bad[0]["metrics"]["ndcg@5"])

    async def test_limit_shrinks_the_pool(self):
        ideas, _ = await self.evaluate(limit=2)
        self.assertEqual(ideas[0]["n_pool"], 4)

    async def test_select_scores_what_the_agent_picks(self):
        ideas, issues = await self.evaluate(select=True, agent=FakeTaskAgent())
        self.assertEqual(issues["select_errors"], 0)
        first = ideas[0]["select"]
        self.assertEqual(set(first), {"1", "2", "3", "4", "5"})
        self.assertEqual(first["1"]["precision"], 1.0)
        self.assertEqual(first["3"]["mean_rel"], (3 + 3 + 2) / 3)
        self.assertEqual(first["5"]["shown"], 9)
        self.assertEqual(len(first["2"]["picked"]), 2)

    async def test_select_failure_is_counted_not_hidden(self):
        class Broken:
            async def select(self, *args):
                raise ValueError("bad json")
        ideas, issues = await self.evaluate(select=True, agent=Broken())
        self.assertEqual(issues["select_errors"], 10)
        self.assertEqual(ideas[0]["select"]["1"], {"error": "bad json"})
        self.assertFalse(rank_eval.judge_validity(issues, False, 2)["ok"])

    async def test_unlabeled_papers_make_the_run_invalid(self):
        scores = dict(self.scores)
        scores.pop(next(iter(scores)))
        _, issues = await rank_eval.evaluate(self.pools, scores, limit=5, method="bm25", progress=lambda *_: None)
        self.assertEqual(issues["unlabeled"], 1)
        self.assertFalse(rank_eval.judge_validity(issues, False, 2)["ok"])
        self.assertTrue(rank_eval.judge_validity(issues, True, 2)["ok"])

    def test_agent_candidates_are_cut_only_after_a_successful_rerank(self):
        ranked = list(range(20))
        with patch.dict(os.environ, {"RERANKER_AGENT_TOP_M": "8"}):
            self.assertEqual(len(rank_eval.agent_candidates(ranked, {}, 3)), 20)
            self.assertEqual(len(rank_eval.agent_candidates(ranked, {"reranking": {"status": "ok", "method": "http"}}, 3)), 8)
            self.assertEqual(len(rank_eval.agent_candidates(ranked, {"reranking": {"status": "ok", "method": "http"}}, 12)), 12)
            self.assertEqual(len(rank_eval.agent_candidates(ranked, {"reranking": {"status": "failed", "method": "http"}}, 3)), 20)
            self.assertEqual(len(rank_eval.agent_candidates(ranked, {"reranking": {"status": "ok", "method": "none"}}, 3)), 20)

    def test_forced_agent_limit_applies_with_or_without_a_reranker(self):
        ranked = list(range(20))
        self.assertEqual(len(rank_eval.agent_candidates(ranked, {}, 3, forced_limit=8)), 8)
        self.assertEqual(len(rank_eval.agent_candidates(ranked, {}, 12, forced_limit=8)), 12)
        ok = {"reranking": {"status": "ok", "method": "http"}}
        with patch.dict(os.environ, {"RERANKER_AGENT_TOP_M": "8"}):
            self.assertEqual(len(rank_eval.agent_candidates(ranked, ok, 3, forced_limit=16)), 16)
            self.assertEqual(len(rank_eval.agent_candidates(ranked, ok, 3)), 8)

    async def test_select_sees_only_the_forced_number_of_candidates(self):
        ideas, _ = await self.evaluate(select=True, agent=FakeTaskAgent(), agent_limit=4)
        self.assertEqual(ideas[0]["select"]["2"]["shown"], 4)
        self.assertEqual(ideas[0]["select"]["5"]["shown"], 5)  # never fewer than k

    async def test_a_reranker_that_fell_back_silently_is_flagged(self):
        real = sources.retrieve

        async def retrieve_with_fallback(*args, **kwargs):
            ranked, stats = await real(*args, **kwargs)
            stats["reranking"] = {"status": "fallback", "method": "http"}
            return ranked, stats
        with patch.object(sources, "retrieve", retrieve_with_fallback):
            _, issues = await self.evaluate(rerank="http")
        self.assertEqual(len(issues["rerank_failed"]), 2)
        self.assertIn("精排未成功", rank_eval.judge_validity(issues, False, 2)["reasons"][0])

    def test_rerank_flag_is_refused_when_the_checkout_has_no_reranker(self):
        with patch("importlib.util.find_spec", return_value=None):
            with self.assertRaisesRegex(SystemExit, "没有 research.reranking"):
                rank_eval.configure_rerank("http")
            self.assertFalse(rank_eval.configure_rerank("none"))

    def test_dense_methods_need_the_embedding_service(self):
        args = rank_eval.parse_args(["run", "--label", "x", "--ranking-method", "hybrid_rrf"])
        env = {k: v for k, v in os.environ.items() if k != "BGE_M3_SERVICE_URL"}
        with patch.dict(os.environ, env, clear=True), \
                patch.object(rank_eval, "load_pools", return_value=self.pools):
            with self.assertRaisesRegex(SystemExit, "BGE"):
                asyncio.run(rank_eval.run(args))


class RunAndCompareTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.pools = synthetic_pools()
        rank_pools.save_pools(copy.deepcopy(self.pools), root / "pools.gz")
        self.labels = root / "labels.jsonl"
        self.labels.write_text("".join(
            json.dumps({"qkey": key.split(":")[0], "pkey": key.split(":")[1], "score": s, "judge": "fake"}) + "\n"
            for key, s in synthetic_scores(self.pools).items()), encoding="utf-8")
        self.human = root / "human.jsonl"

    def argv(self, label, *extra):
        return ["run", "--label", label, "--pools", str(Path(self.tmp.name) / "pools.gz"),
                "--labels", str(self.labels), "--human", str(self.human), "--out", str(Path(self.tmp.name) / label),
                "--pool-limit", "5", *extra]

    async def run_eval(self, label, *extra, agent=FakeTaskAgent):
        with patch("benchmarks.run_benchmark.collect_environment", return_value={"git_sha": "abc"}), \
                patch.dict(os.environ, {"PAPER_RANKING_METHOD": ""}):
            return await rank_eval.run(rank_eval.parse_args(self.argv(label, *extra)), agent_factory=agent)

    async def test_run_writes_results_and_records_what_was_measured(self):
        result = await self.run_eval("a", "--ranking-method", "bm25", "--select")
        out = Path(self.tmp.name) / "a"
        self.assertTrue((out / "results.json").is_file() and (out / "results.md").is_file())
        meta = result["meta"]
        self.assertEqual((meta["mode"], meta["ranking_method"], meta["rerank"], meta["pool_limit"]),
                         ("rank+select", "bm25", "none", 5))
        self.assertEqual(meta["judges"], ["fake"])
        self.assertEqual(meta["pools_hash"], rank_pools.load_pools(Path(self.tmp.name) / "pools.gz")["hash"])
        self.assertTrue(result["validity"]["ok"])
        self.assertIn("select", result["aggregate"])
        self.assertIn("recall@8", (out / "results.md").read_text(encoding="utf-8"))

    async def test_compare_reports_paired_differences(self):
        base = await self.run_eval("base", "--ranking-method", "basic")
        better = await self.run_eval("better", "--ranking-method", "bm25")
        text = rank_eval.compare(base, better, resamples=500)
        self.assertIn("`base`（A）vs `better`（B）", text)
        self.assertIn("ndcg@5", text)
        self.assertIn("95%", text)
        same = rank_eval.compare(base, base, resamples=500)
        self.assertIn("+0.0000", same)

    async def test_compare_refuses_runs_measured_differently(self):
        a = await self.run_eval("a", "--ranking-method", "bm25")
        b = await self.run_eval("b", "--ranking-method", "bm25", "--select")
        with self.assertRaisesRegex(ValueError, "mode"):
            rank_eval.compare(a, b)
        c = copy.deepcopy(a)
        c["meta"]["labels_hash"] = "other"
        with self.assertRaisesRegex(ValueError, "labels_hash"):
            rank_eval.compare(a, c)
        d = copy.deepcopy(a)
        d["meta"]["pool_limit"] = 25
        with self.assertRaisesRegex(ValueError, "pool_limit"):
            rank_eval.compare(a, d)
        e = copy.deepcopy(a)
        e["validity"] = {"ok": False, "reasons": ["精排未成功"], "warnings": []}
        with self.assertRaisesRegex(ValueError, "B 的结果无效"):
            rank_eval.compare(a, e)

    async def test_human_labels_change_the_labels_hash(self):
        a = await self.run_eval("a", "--ranking-method", "bm25")
        first = next(iter(synthetic_scores(self.pools)))
        self.human.write_text(json.dumps({"qkey": first.split(":")[0], "pkey": first.split(":")[1], "score": 1}) + "\n",
                              encoding="utf-8")
        b = await self.run_eval("b", "--ranking-method", "bm25")
        self.assertNotEqual(a["meta"]["labels_hash"], b["meta"]["labels_hash"])
        self.assertEqual(b["meta"]["human_labels"], 1)


if __name__ == "__main__":
    unittest.main()
