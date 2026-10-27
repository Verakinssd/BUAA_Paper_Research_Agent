import tempfile
import unittest
from pathlib import Path

from research.benchmarking import build_comparison, render_markdown, write_visualizations


def metrics(seconds, attempted=2, completed=2):
    failed = attempted - completed
    return {
        "processing_seconds": seconds,
        "attempted_count": attempted,
        "completed_count": completed,
        "failed_count": failed,
        "failure_rate": failed / attempted,
        "throughput_papers_per_second": attempted / seconds,
        "scheduler_wait_seconds": 0.5,
        "stage_metrics": {
            stage: {
                "seconds": 1.0,
                "wait_seconds": 0.25,
                "attempts": attempted,
                "failures": 0,
            }
            for stage in ("cache", "download", "convert", "summary")
        },
    }


def run(sequence, repeat, mode, seconds, completed=2):
    return {
        "sequence": sequence,
        "repeat": repeat,
        "mode": mode,
        "wall_seconds": seconds,
        "output_dir": f"/tmp/{mode}-{repeat}",
        "metrics": metrics(seconds, completed=completed),
        "papers": [
            {"id": "P1", "status": "ok", "seconds": seconds / 2},
            {
                "id": "P2",
                "status": "ok" if completed == 2 else "failed",
                "seconds": seconds / 2,
                "error": "" if completed == 2 else "failed",
            },
        ],
    }


def fixture_raw():
    return {
        "finished_at": "2026-01-01T00:00:00Z",
        "manifest_sha256": "abc",
        "manifest": {
            "name": "fixture",
            "papers": [
                {"id": "P1", "title": "Paper 1", "sources": ["arXiv"], "pdf_urls": ["https://example.com/1.pdf"]},
                {"id": "P2", "title": "Paper 2", "sources": ["PMLR"], "pdf_urls": ["https://example.com/2.pdf"]},
            ],
        },
        "pipeline_config": {
            "download_concurrency": 4,
            "convert_concurrency": 1,
            "summary_concurrency": 2,
            "max_in_flight": 8,
        },
        "model_config": {"LLM_MODEL": "fixture-model"},
        "runs": [
            run(1, 1, "legacy", 10),
            run(2, 1, "staged", 6, completed=1),
            run(3, 2, "staged", 8, completed=2),
            run(4, 2, "legacy", 12),
        ],
        "limitations": ["synthetic fixture"],
    }


class BenchmarkReportTests(unittest.TestCase):
    def test_report_uses_medians_formula_and_failure_delta(self):
        raw = fixture_raw()
        comparison = build_comparison(raw)
        self.assertAlmostEqual(comparison["modes"]["legacy"]["median_seconds"], 11)
        self.assertAlmostEqual(comparison["modes"]["staged"]["median_seconds"], 7)
        self.assertAlmostEqual(comparison["improvement_rate"], (11 - 7) / 11)
        self.assertGreater(comparison["failure_rate_delta"], 0)
        raw["comparison"] = comparison
        report = render_markdown(raw)
        self.assertIn("36.36%", report)
        self.assertIn("失败率上升", report)
        self.assertIn("阶段执行、等待与失败", report)
        self.assertIn("whole-paper scheduler", report)
        self.assertIn("中途发生阶段失败或内部重试", report)
        self.assertIn("01-round-duration.svg", report)
        self.assertIn("耗时改善率 =", report)

    def test_visualizations_are_valid_svg_files(self):
        raw = fixture_raw()
        raw["comparison"] = build_comparison(raw)
        with tempfile.TemporaryDirectory() as directory:
            names = write_visualizations(raw, directory)
            self.assertEqual(len(names), 4)
            for name in names:
                content = (Path(directory) / name).read_text(encoding="utf-8")
                self.assertTrue(content.startswith("<svg"))
                self.assertIn("</svg>", content)

    def test_revision_provenance_is_rendered(self):
        raw = fixture_raw()
        raw["comparison"] = build_comparison(raw)
        raw["revision"] = {
            "summary": "用成功补跑替换失败轮次。",
            "replaced_run": "/tmp/original.json",
            "replacement_run": "/tmp/replacement.json",
        }
        report = render_markdown(raw)
        self.assertIn("## 修订说明", report)
        self.assertIn("原始结果和失败样本均保留", report)


if __name__ == "__main__":
    unittest.main()
