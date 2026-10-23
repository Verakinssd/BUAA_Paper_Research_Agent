import tempfile
import unittest
from pathlib import Path

from research.simulated_benchmarking import (
    build_simulation_comparison,
    render_simulation_markdown,
    write_simulation_visualizations,
)


def metric(seconds):
    return {
        "attempted_count": 10,
        "failed_count": 0,
        "throughput_papers_per_second": 10 / seconds,
        "scheduler_wait_seconds": 2.0,
        "stage_metrics": {
            stage: {"seconds": 1.0, "wait_seconds": 0.25}
            for stage in ("cache", "download", "convert", "summary")
        },
    }


def fixture_raw():
    runs = []
    for repeat, legacy, staged in ((1, 10.0, 8.0), (2, 12.0, 9.0)):
        for mode, seconds in (("legacy", legacy), ("staged", staged)):
            runs.append({
                "scenario": "balanced",
                "repeat": repeat,
                "mode": mode,
                "wall_seconds": seconds,
                "metrics": metric(seconds),
                "observed_max_concurrency": {
                    "download": 2 if mode == "legacy" else 4,
                    "convert": 1,
                    "summary": 2,
                },
            })
    return {
        "finished_at": "2026-01-01T00:00:00Z",
        "manifest_path": "/tmp/manifest.json",
        "paper_count": 10,
        "repeats": 2,
        "scenarios": {
            "balanced": {
                "label": "均衡流水线",
                "download_seconds": 0.1,
                "convert_seconds": 0.04,
                "summary_seconds": 0.3,
                "description": "fixture",
            }
        },
        "pipeline_config": {
            "legacy_concurrency": 2,
            "download_concurrency": 4,
            "convert_concurrency": 1,
            "summary_concurrency": 2,
            "max_in_flight": 8,
        },
        "runs": runs,
    }


class SimulatedBenchmarkingTests(unittest.TestCase):
    def test_comparison_and_report(self):
        raw = fixture_raw()
        comparison = build_simulation_comparison(raw)
        self.assertEqual(comparison["balanced"]["modes"]["legacy"]["median_seconds"], 11)
        self.assertEqual(comparison["balanced"]["modes"]["staged"]["median_seconds"], 8.5)
        self.assertAlmostEqual(comparison["balanced"]["improvement_rate"], (11 - 8.5) / 11)
        raw["comparison"] = comparison
        report = render_simulation_markdown(raw)
        self.assertIn("完全离线", report)
        self.assertIn("22.73%", report)
        self.assertIn("不能替代真实网络环境测试", report)
        self.assertIn("## 三种实验设置说明", report)
        self.assertIn("这里的“均衡”不是指三个阶段耗时完全相等", report)
        self.assertIn("总结约占 93.0%", report)
        self.assertIn("这里故意不模拟 arXiv 的强制串行限速", report)

    def test_charts_are_written(self):
        raw = fixture_raw()
        raw["comparison"] = build_simulation_comparison(raw)
        with tempfile.TemporaryDirectory() as directory:
            names = write_simulation_visualizations(raw, directory)
            self.assertEqual(len(names), 3)
            for name in names:
                content = (Path(directory) / name).read_text(encoding="utf-8")
                self.assertTrue(content.startswith("<svg"))
                self.assertIn("</svg>", content)


if __name__ == "__main__":
    unittest.main()
