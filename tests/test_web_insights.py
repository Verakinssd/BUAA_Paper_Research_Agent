"""Tests for web/insights.js (run time + candidate ranking panel). Needs node; skipped without it."""
import json
import shutil
import subprocess
import unittest
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
NODE = shutil.which("node")

HARNESS = """
const fs = require('fs'), vm = require('vm');
const src = fs.readFileSync(process.argv[1], 'utf8');
const elements = {};
for (const id of ['insight-card', 'insight-body']) {
  elements[id] = { hidden: true, innerHTML: '', querySelector: (sel) => elements[sel.slice(1)] };
}
const ctx = { document: { getElementById: (id) => elements[id] || null } };
vm.createContext(ctx);
vm.runInContext(src + ';this.api = { insightStages, insightRanking, insightsHtml, renderInsights, insightSeconds };', ctx);
const cases = JSON.parse(fs.readFileSync(0, 'utf8'));
const api = ctx.api;
const out = {};
out.seconds = cases.seconds.map((v) => api.insightSeconds(v));
out.stages = cases.jobs.map((job) => api.insightStages(job));
out.ranking = cases.jobs.map((job) => api.insightRanking(job));
out.html = cases.jobs.map((job) => api.insightsHtml(job));
out.render = cases.jobs.map((job) => {
  elements['insight-card'].hidden = true; elements['insight-body'].innerHTML = '';
  api.renderInsights(job);
  return { hidden: elements['insight-card'].hidden, filled: elements['insight-body'].innerHTML.length > 0 };
});
process.stdout.write(JSON.stringify(out));
"""

EVENTS = [
    {"event": "search_started", "time": 1000.0},
    {"event": "sources_searched", "time": 1002.5},
    {"event": "selection_started", "time": 1003.0},
    {"event": "task_agent_selected", "time": 1013.0},
    {"event": "report_started", "time": 1093.0},
    {"event": "report_completed", "time": 1103.0},
]
CANDIDATES = [
    {"id": "P1", "title": "Low score paper", "year": "2021", "sources": ["arxiv"], "rank_score": 0.2},
    {"id": "P2", "title": "Best paper", "year": "2023", "sources": ["openalex"], "rank_score": 0.9},
    {"id": "P3", "title": "Middle paper", "year": "2022", "sources": ["arxiv", "openalex"], "rank_score": 0.5},
    {"id": "P4", "title": "No score paper", "year": "", "sources": [], "rank_score": None},
]
COMPLETE = {
    "top_k": 2, "concurrent": True, "model_id": "q4_k_m", "elapsed_seconds": 120,
    "metrics": {"search_seconds": 2.5, "processing_seconds": 80.0, "server_total_seconds": 100.0,
                "sources": {"arxiv": {"status": "ok"}, "ranking_method": "bm25"}},
    "events": EVENTS, "candidates": CANDIDATES, "papers": [{"id": "P2"}, {"id": "P3"}],
}
TRUNCATED = dict(COMPLETE, events=[{"event": "paper_started", "time": 1050.0}])
RUNNING = {"top_k": 3, "candidates": CANDIDATES, "papers": [], "metrics": {}, "events": []}
HOSTILE = {"top_k": 1, "metrics": {}, "events": [], "papers": [],
           "candidates": [{"id": 'P"1', "title": "<img src=x onerror=alert(1)>", "rank_score": 0.5}]}
EMPTY = {"top_k": 3, "candidates": [], "papers": [], "metrics": {}, "events": []}


@unittest.skipUnless(NODE, "node is not installed")
class InsightsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        payload = {"seconds": [None, 1.234, 42.4, 75, 3725],
                   "jobs": [COMPLETE, TRUNCATED, RUNNING, HOSTILE, EMPTY]}
        done = subprocess.run([NODE, "-e", HARNESS, str(PROJECT / "web" / "insights.js")],
                              input=json.dumps(payload), capture_output=True, text=True,
                              encoding="utf-8", timeout=30)
        if done.returncode:
            raise AssertionError(done.stderr)
        cls.out = json.loads(done.stdout)

    def test_seconds_formatting(self):
        self.assertEqual(self.out["seconds"], ["—", "1.2 秒", "42 秒", "1 分 15 秒", "1 时 2 分"])

    def test_stage_rows_use_metrics_and_event_times(self):
        stages = self.out["stages"][0]
        self.assertEqual([r["label"] for r in stages["rows"]], ["多源检索", "任务 Agent 筛选", "全文处理", "综合报告"])
        self.assertEqual([r["seconds"] for r in stages["rows"]], [2.5, 10.0, 80.0, 10.0])
        self.assertAlmostEqual(sum(r["share"] for r in stages["rows"]), 1.0)
        self.assertEqual(stages["total"], 100.0)

    def test_select_and_report_are_merged_when_trace_is_truncated(self):
        rows = self.out["stages"][1]["rows"]
        self.assertEqual([r["label"] for r in rows], ["多源检索", "全文处理", "筛选 + 报告"])
        self.assertEqual([r["seconds"] for r in rows], [2.5, 80.0, 17.5])

    def test_running_job_has_ranking_but_no_stage_times(self):
        self.assertFalse(self.out["stages"][2]["hasData"])
        self.assertIn("各阶段完成后显示耗时", self.out["html"][2])
        self.assertEqual(len(self.out["ranking"][2]["rows"]), 4)

    def test_ranking_order_scores_and_selection(self):
        ranking = self.out["ranking"][0]
        self.assertEqual([r["id"] for r in ranking["rows"]], ["P2", "P3", "P1", "P4"])
        self.assertEqual([r["rank"] for r in ranking["rows"]], [1, 2, 3, 4])
        self.assertEqual(ranking["rows"][3]["score"], None)  # missing score sorts last
        self.assertEqual(ranking["rows"][0]["width"], 100)
        self.assertEqual(ranking["pickedRanks"], [1, 2])
        self.assertEqual(ranking["methodLabel"], "BM25")
        self.assertIn("#1", self.out["html"][0])
        self.assertIn("入选", self.out["html"][0])
        self.assertIn("论文并发 2", self.out["html"][0])

    def test_untrusted_titles_are_escaped(self):
        html = self.out["html"][3]
        self.assertNotIn("<img", html)
        self.assertIn("&lt;img", html)
        self.assertNotIn('P"1', html)

    def test_card_visibility(self):
        self.assertEqual([r["hidden"] for r in self.out["render"]], [False, False, False, False, True])
        self.assertTrue(self.out["render"][0]["filled"])
        self.assertFalse(self.out["render"][4]["filled"])


class WiringTests(unittest.TestCase):
    def test_page_loads_insights_before_app_and_has_mount_points(self):
        html = (PROJECT / "web" / "index.html").read_text(encoding="utf-8")
        self.assertLess(html.index("/static/insights.js"), html.index("/static/app.js"))
        self.assertIn('id="insight-card"', html)
        self.assertIn('id="insight-body"', html)

    def test_render_job_calls_insights_defensively(self):
        app = (PROJECT / "web" / "app.js").read_text(encoding="utf-8")
        self.assertIn('typeof renderInsights === "function"', app)


if __name__ == "__main__":
    unittest.main()
