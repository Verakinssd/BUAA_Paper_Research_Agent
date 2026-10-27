"""Aggregation and reporting helpers for the deterministic pipeline simulation."""
from __future__ import annotations

import statistics
from pathlib import Path

from research.benchmarking import MODE_COLORS, STAGES, _bar_chart_svg


def _median(values):
    return statistics.median(values) if values else 0.0


def build_simulation_comparison(raw):
    comparisons = {}
    for scenario in raw["scenarios"]:
        scenario_runs = [row for row in raw["runs"] if row["scenario"] == scenario]
        modes = {}
        for mode in ("legacy", "staged"):
            selected = [row for row in scenario_runs if row["mode"] == mode]
            durations = [row["wall_seconds"] for row in selected]
            throughputs = [row["metrics"]["throughput_papers_per_second"] for row in selected]
            attempted = sum(row["metrics"]["attempted_count"] for row in selected)
            failed = sum(row["metrics"]["failed_count"] for row in selected)
            modes[mode] = {
                "rounds": len(selected),
                "median_seconds": _median(durations),
                "min_seconds": min(durations),
                "max_seconds": max(durations),
                "median_throughput": _median(throughputs),
                "attempted_count": attempted,
                "failed_count": failed,
                "failure_rate": failed / attempted if attempted else 0.0,
                "median_scheduler_wait_seconds": _median([
                    row["metrics"]["scheduler_wait_seconds"] for row in selected
                ]),
                "observed_max_concurrency": {
                    stage: max(
                        row["observed_max_concurrency"][stage] for row in selected
                    )
                    for stage in ("download", "convert", "summary")
                },
                "stage_medians": {
                    stage: {
                        "seconds": _median([
                            row["metrics"]["stage_metrics"][stage]["seconds"]
                            for row in selected
                        ]),
                        "wait_seconds": _median([
                            row["metrics"]["stage_metrics"][stage]["wait_seconds"]
                            for row in selected
                        ]),
                    }
                    for stage in STAGES
                },
            }
        legacy = modes["legacy"]["median_seconds"]
        staged = modes["staged"]["median_seconds"]
        comparisons[scenario] = {
            "modes": modes,
            "improvement_rate": (legacy - staged) / legacy if legacy else 0.0,
        }
    return comparisons


def write_simulation_visualizations(raw, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison = raw.get("comparison") or build_simulation_comparison(raw)
    labels, durations, colors, throughputs = [], [], [], []
    for scenario, config in raw["scenarios"].items():
        for mode in ("legacy", "staged"):
            labels.append(f"{config['label']}\n{mode}")
            durations.append(comparison[scenario]["modes"][mode]["median_seconds"])
            throughputs.append(
                comparison[scenario]["modes"][mode]["median_throughput"] * 60
            )
            colors.append(MODE_COLORS[mode])
    charts = {
        "01-simulation-medians.svg": _bar_chart_svg(
            "模拟实验中位耗时",
            f"每种场景交替执行 {raw['repeats']} 轮；所有阶段使用固定延迟，不访问网络",
            labels,
            durations,
            colors,
            "中位耗时（秒）",
            digits=3,
        ),
        "02-simulation-throughput.svg": _bar_chart_svg(
            "模拟实验中位吞吐量",
            "吞吐量按每轮 10 篇论文的实际墙钟耗时计算",
            labels,
            throughputs,
            colors,
            "论文数 / 分钟",
            digits=2,
        ),
    }
    stage_names = {"download": "下载", "convert": "转换", "summary": "总结"}
    for scenario, config in raw["scenarios"].items():
        stage_labels, waits, stage_colors = [], [], []
        for stage in ("download", "convert", "summary"):
            for mode in ("legacy", "staged"):
                stage_labels.append(f"{stage_names[stage]}\n{mode}")
                waits.append(
                    comparison[scenario]["modes"][mode]["stage_medians"][stage]["wait_seconds"]
                )
                stage_colors.append(MODE_COLORS[mode])
        filename = f"03-wait-{scenario}.svg"
        charts[filename] = _bar_chart_svg(
            f"{config['label']}：阶段等待时间",
            "每根柱为一轮中 10 篇论文的累计信号量等待时间中位数",
            stage_labels,
            waits,
            stage_colors,
            "累计等待（秒）",
            digits=3,
        )
    for filename, content in charts.items():
        (output_dir / filename).write_text(content, encoding="utf-8")
    return list(charts)


def render_simulation_markdown(raw):
    comparison = raw.get("comparison") or build_simulation_comparison(raw)
    lines = [
        "# 论文流水线分阶段并发：可控模拟实验报告",
        "",
        "> 本实验完全离线：不下载真实 PDF、不调用模型 API。它验证调度算法本身，不能替代真实网络环境测试。",
        "",
        "![模拟实验中位耗时](01-simulation-medians.svg)",
        "",
        f"- 生成时间：{raw['finished_at']}",
        f"- 论文清单：`{raw['manifest_path']}`（{raw['paper_count']} 篇，与真实 Benchmark 相同）",
        f"- 每个场景轮数：{raw['repeats']} 轮 legacy + {raw['repeats']} 轮 staged",
        "- 执行顺序：奇数轮 legacy→staged，偶数轮 staged→legacy",
        "- 计时：`time.perf_counter()` 单调时钟",
        "- 随机性：无随机延迟；每篇论文使用固定、可复现的耗时倍率",
        "",
        "## 模拟配置",
        "",
        "| 场景 | 下载/篇 | 转换/篇 | 总结/篇 | 目的 |",
        "|---|---:|---:|---:|---|",
    ]
    for scenario in raw["scenarios"].values():
        lines.append(
            f"| {scenario['label']} | {scenario['download_seconds']:.3f}s | "
            f"{scenario['convert_seconds']:.3f}s | {scenario['summary_seconds']:.3f}s | "
            f"{scenario['description']} |"
        )
    lines.extend([
        "",
        "## 三种实验设置说明",
        "",
        "三个场景使用完全相同的 10 篇论文、论文顺序、并发参数和调度代码，仅改变下载、转换、总结三个阶段的固定模拟耗时。每篇论文还使用 0.80～1.25 的固定耗时倍率，用来模拟论文大小和处理难度差异；同一篇论文在 legacy 与 staged 中的倍率完全相同，因此不会偏向任何模式。",
        "",
        "### 1. 均衡流水线",
        "",
        "- 基础耗时：下载 40ms、转换 15ms、总结 120ms。",
        "- 单篇三个阶段的基础总耗时为 175ms，其中下载约占 22.9%、转换约占 8.6%、总结约占 68.6%。",
        "- 这里的“均衡”不是指三个阶段耗时完全相等，而是指下载和转换的占比仍然足以影响总耗时，三个阶段之间存在可利用的流水线重叠空间。",
        "- 它模拟一般论文处理任务：API 总结最慢，但 PDF 获取和解析同样不是可以忽略的瞬时操作。",
        "- 预期现象：staged 可以在前两篇论文总结时继续下载和转换后续论文，从而减少 legacy 整篇占槽造成的队头阻塞。",
        "",
        "### 2. API 主导",
        "",
        "- 基础耗时：下载 10ms、转换 5ms、总结 200ms。",
        "- 单篇基础总耗时为 215ms，其中总结约占 93.0%，下载和转换合计仅占约 7.0%。",
        "- 它模拟真实系统中模型响应远慢于 PDF 下载和文本转换的情况，也是当前真实 Benchmark 最接近的抽象场景。",
        "- 两种模式的有效总结并发都为 2，因此 staged 无法提高最主要瓶颈的并发上限，只能让很短的下载和转换提前完成。",
        "- 预期现象：staged 仍可能因减少整篇调度等待而略有收益，但改善率应明显低于下载主导场景；如果外部 API 波动很大，这部分收益可能被噪声覆盖。",
        "",
        "### 3. 下载主导",
        "",
        "- 基础耗时：下载 120ms、转换 10ms、总结 30ms。",
        "- 单篇基础总耗时为 160ms，其中下载占 75.0%，转换占 6.25%，总结占 18.75%。",
        "- 它模拟 PDF 较大、论文站点响应较慢，但下载服务允许并行请求的情况。这里故意不模拟 arXiv 的强制串行限速，用于单独测量下载并发从 2 提升到 4 的调度上限。",
        "- legacy 最多只有两篇论文进入完整流程，因此同时下载数量不会超过 2；staged 可以使用 4 个下载槽，并让已下载论文继续进入转换和总结阶段。",
        "- 预期现象：这是最有利于 staged 的场景，用来验证下载并发和阶段重叠能够带来的理论收益，而不是代表当前 5 篇 arXiv 测试集的真实下载表现。",
        "",
        "### 三种设置的核心区别",
        "",
        "| 对比项 | 均衡流水线 | API 主导 | 下载主导 |",
        "|---|---|---|---|",
        "| 主要耗时来源 | 总结为主，下载/转换仍不可忽略 | 总结占绝对主导 | 下载占主导 |",
        "| staged 可利用的能力 | 三阶段重叠 | 主要是减少队头阻塞 | 下载并发 4 + 阶段重叠 |",
        "| 预期收益 | 中等 | 较小 | 最大 |",
        "| 对真实系统的含义 | 展示一般流水线收益 | 最接近当前 API 瓶颈 | 展示下载可并行时的上限 |",
        "",
        "因此，这三个场景不是三套不同算法，而是对同一套调度算法进行敏感性分析：通过改变瓶颈所在阶段，观察 staged 在什么条件下有效、收益何时受限。",
        "",
        "流水线并发参数：",
        "",
        "| 参数 | 值 |",
        "|---|---:|",
    ])
    for key, value in raw["pipeline_config"].items():
        lines.append(f"| {key} | {value} |")
    lines.extend([
        "",
        "## 总体结果",
        "",
        "| 场景 | Legacy 中位数 | Staged 中位数 | 改善率 | Legacy 范围 | Staged 范围 | 最终失败 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for scenario, config in raw["scenarios"].items():
        result = comparison[scenario]
        legacy = result["modes"]["legacy"]
        staged = result["modes"]["staged"]
        lines.append(
            f"| {config['label']} | {legacy['median_seconds']:.3f}s | "
            f"{staged['median_seconds']:.3f}s | {result['improvement_rate'] * 100:.2f}% | "
            f"{legacy['min_seconds']:.3f}–{legacy['max_seconds']:.3f}s | "
            f"{staged['min_seconds']:.3f}–{staged['max_seconds']:.3f}s | "
            f"{legacy['failed_count'] + staged['failed_count']} |"
        )
    lines.extend([
        "",
        "改善率 = `(Legacy 中位耗时 - Staged 中位耗时) / Legacy 中位耗时 × 100%`。",
        "",
        "![模拟实验吞吐量](02-simulation-throughput.svg)",
        "",
        "## 实际并发上限",
        "",
        "以下是全部轮次中观测到的最大同时执行数量：",
        "",
        "| 场景 | 模式 | 下载 | 转换 | 总结 | 整篇调度等待中位数 |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for scenario, config in raw["scenarios"].items():
        for mode in ("legacy", "staged"):
            result = comparison[scenario]["modes"][mode]
            observed = result["observed_max_concurrency"]
            lines.append(
                f"| {config['label']} | {mode} | {observed['download']} | "
                f"{observed['convert']} | {observed['summary']} | "
                f"{result['median_scheduler_wait_seconds']:.3f}s |"
            )
    lines.extend(["", "## 各场景阶段等待", ""])
    for scenario, config in raw["scenarios"].items():
        lines.extend([
            f"### {config['label']}",
            "",
            f"![{config['label']}阶段等待](03-wait-{scenario}.svg)",
            "",
        ])
    lines.extend([
        "## 各轮结果",
        "",
        "| 场景 | 轮次 | Legacy(s) | Staged(s) | 本轮改善率 |",
        "|---|---:|---:|---:|---:|",
    ])
    for scenario, config in raw["scenarios"].items():
        for repeat in range(1, raw["repeats"] + 1):
            legacy = next(
                row for row in raw["runs"]
                if row["scenario"] == scenario and row["repeat"] == repeat and row["mode"] == "legacy"
            )
            staged = next(
                row for row in raw["runs"]
                if row["scenario"] == scenario and row["repeat"] == repeat and row["mode"] == "staged"
            )
            improvement = (legacy["wall_seconds"] - staged["wall_seconds"]) / legacy["wall_seconds"]
            lines.append(
                f"| {config['label']} | {repeat} | {legacy['wall_seconds']:.3f} | "
                f"{staged['wall_seconds']:.3f} | {improvement * 100:.2f}% |"
            )
    lines.extend(["", "## 结论", ""])
    for scenario, config in raw["scenarios"].items():
        improvement = comparison[scenario]["improvement_rate"]
        if improvement > 0:
            conclusion = f"staged 中位耗时缩短 {improvement * 100:.2f}%"
        else:
            conclusion = f"staged 中位耗时增加 {abs(improvement) * 100:.2f}%"
        lines.append(f"- **{config['label']}**：{conclusion}。")
    lines.extend([
        "",
        "模拟实验用于隔离调度算法的影响：若 staged 在均衡或下载主导场景中加速，说明分阶段信号量能够减少整篇论文占槽造成的队头阻塞；若 API 主导场景收益很小，则说明总结并发度相同会限制最终收益。",
        "",
        "## 实验限制",
        "",
        "- `asyncio.sleep` 和 `time.sleep` 只能模拟耗时，不能复现真实网络协议、服务端限流和 PDF 解析负载。",
        "- Windows 线程调度和计时精度会产生少量波动，因此以 20 轮中位数而不是单轮结果判断。",
        "- 模拟实验只能证明调度机制在给定工作负载下的效果，真实端到端结论仍应以真实 Benchmark 为准。",
        "- 三种场景均在同一进程中顺序运行，不同时执行两种模式。",
    ])
    return "\n".join(lines) + "\n"
