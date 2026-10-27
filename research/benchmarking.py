"""Pure helpers for reproducible pipeline benchmark reports and SVG charts."""
from __future__ import annotations

import html
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path


STAGES = ("cache", "download", "convert", "summary")
MODE_COLORS = {"legacy": "#475569", "staged": "#0F766E"}


def _median(values):
    return statistics.median(values) if values else None


def _number(value, digits=3):
    return "—" if value is None else f"{value:.{digits}f}"


def _percent(value):
    return "—" if value is None else f"{value * 100:.2f}%"


def summarize_mode(runs, mode):
    selected = [row for row in runs if row["mode"] == mode]
    durations = [row["wall_seconds"] for row in selected]
    attempted = sum(row["metrics"]["attempted_count"] for row in selected)
    completed = sum(row["metrics"]["completed_count"] for row in selected)
    failed = sum(row["metrics"]["failed_count"] for row in selected)
    total_seconds = sum(durations)
    return {
        "mode": mode,
        "rounds": len(selected),
        "total_seconds": total_seconds,
        "median_seconds": _median(durations),
        "min_seconds": min(durations) if durations else None,
        "max_seconds": max(durations) if durations else None,
        "attempted_count": attempted,
        "completed_count": completed,
        "failed_count": failed,
        "failure_rate": failed / attempted if attempted else 0.0,
        "throughput_papers_per_second": attempted / total_seconds if total_seconds else 0.0,
    }


def build_comparison(raw):
    runs = raw["runs"]
    modes = {
        mode: summarize_mode(runs, mode)
        for mode in ("legacy", "staged")
        if any(row["mode"] == mode for row in runs)
    }
    legacy = modes.get("legacy")
    staged = modes.get("staged")
    improvement = None
    if legacy and staged and legacy["median_seconds"]:
        improvement = (
            (legacy["median_seconds"] - staged["median_seconds"])
            / legacy["median_seconds"]
        )
    failure_rate_delta = None
    if legacy and staged:
        failure_rate_delta = staged["failure_rate"] - legacy["failure_rate"]
    return {
        "modes": modes,
        "improvement_rate": improvement,
        "failure_rate_delta": failure_rate_delta,
    }


def _bar_chart_svg(title, subtitle, labels, values, colors, unit, digits=1):
    """Render one accessible, dependency-free SVG bar chart."""
    width, height = 1040, 560
    left, right, top, bottom = 92, 32, 100, 104
    plot_width = width - left - right
    plot_height = height - top - bottom
    maximum = max(values, default=0.0)
    axis_max = maximum * 1.18 if maximum else 1.0
    slot = plot_width / max(len(values), 1)
    bar_width = min(86, slot * 0.58)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" rx="18" fill="#F8FAFC"/>',
        f'<text x="{left}" y="42" font-size="25" font-weight="700" fill="#0F172A">{html.escape(title)}</text>',
        f'<text x="{left}" y="70" font-size="14" fill="#64748B">{html.escape(subtitle)}</text>',
    ]
    for tick in range(5):
        ratio = tick / 4
        y = top + plot_height * (1 - ratio)
        value = axis_max * ratio
        parts.extend([
            f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#CBD5E1" stroke-width="1"/>',
            f'<text x="{left-12}" y="{y+5:.1f}" text-anchor="end" font-size="12" fill="#64748B">{value:.{digits}f}</text>',
        ])
    for index, (label, value, color) in enumerate(zip(labels, values, colors)):
        x = left + slot * index + (slot - bar_width) / 2
        bar_height = plot_height * value / axis_max
        y = top + plot_height - bar_height
        label_lines = str(label).split("\n")
        parts.extend([
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" rx="7" fill="{color}"/>',
            f'<text x="{x+bar_width/2:.1f}" y="{max(y-10, top-4):.1f}" text-anchor="middle" font-size="13" font-weight="700" fill="#0F172A">{value:.{digits}f}</text>',
        ])
        for line_index, line in enumerate(label_lines):
            parts.append(
                f'<text x="{x+bar_width/2:.1f}" y="{top+plot_height+28+line_index*19:.1f}" text-anchor="middle" font-size="13" fill="#334155">{html.escape(line)}</text>'
            )
    parts.extend([
        f'<text x="22" y="{top+plot_height/2:.1f}" transform="rotate(-90 22 {top+plot_height/2:.1f})" text-anchor="middle" font-size="13" fill="#475569">{html.escape(unit)}</text>',
        '</svg>',
    ])
    return "\n".join(parts)


def _stage_median(raw, mode, stage, field):
    values = [
        row["metrics"]["stage_metrics"][stage].get(field, 0.0)
        for row in raw["runs"]
        if row["mode"] == mode
    ]
    return _median(values) or 0.0


def write_visualizations(raw, output_dir):
    """Write SVG charts next to the Markdown report and return their names."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = sorted(raw["runs"], key=lambda row: (row["repeat"], row["mode"]))
    chart_specs = []

    chart_specs.append((
        "01-round-duration.svg",
        _bar_chart_svg(
            "各轮总耗时",
            "同一轮使用完全相同的 10 篇论文；柱顶为实测墙钟时间",
            [f"第 {row['repeat']} 轮\n{row['mode']}" for row in runs],
            [row["wall_seconds"] for row in runs],
            [MODE_COLORS[row["mode"]] for row in runs],
            "总耗时（秒）",
        ),
    ))

    stage_labels = []
    stage_values = []
    stage_colors = []
    wait_labels = []
    wait_values = []
    wait_colors = []
    stage_names = {"cache": "缓存", "download": "下载", "convert": "转换", "summary": "总结"}
    for stage in STAGES:
        for mode in ("legacy", "staged"):
            if any(row["mode"] == mode for row in raw["runs"]):
                label = f"{stage_names[stage]}\n{mode}"
                stage_labels.append(label)
                stage_values.append(_stage_median(raw, mode, stage, "seconds"))
                stage_colors.append(MODE_COLORS[mode])
                wait_labels.append(label)
                wait_values.append(_stage_median(raw, mode, stage, "wait_seconds"))
                wait_colors.append(MODE_COLORS[mode])

    chart_specs.append((
        "02-stage-execution.svg",
        _bar_chart_svg(
            "分阶段累计执行时间（中位数）",
            "每根柱为一轮内所有论文该阶段耗时之和；阶段会重叠，不能相加为总耗时",
            stage_labels, stage_values, stage_colors, "累计执行时间（秒）",
        ),
    ))
    chart_specs.append((
        "03-stage-wait.svg",
        _bar_chart_svg(
            "分阶段信号量等待时间（中位数）",
            "等待时间揭示并发瓶颈；legacy 的整篇论文等待另见数据表",
            wait_labels, wait_values, wait_colors, "累计等待时间（秒）",
        ),
    ))

    comparison = raw.get("comparison") or build_comparison(raw)
    modes = [mode for mode in ("legacy", "staged") if mode in comparison["modes"]]
    throughput_values = [
        comparison["modes"][mode]["throughput_papers_per_second"] * 60
        for mode in modes
    ]
    chart_specs.append((
        "04-throughput.svg",
        _bar_chart_svg(
            "总体吞吐量",
            "按三轮累计尝试论文数 / 三轮累计耗时计算",
            modes,
            throughput_values,
            [MODE_COLORS[mode] for mode in modes],
            "论文数 / 分钟",
            digits=2,
        ),
    ))

    for filename, content in chart_specs:
        (output_dir / filename).write_text(content, encoding="utf-8")
    return [filename for filename, _ in chart_specs]


def render_markdown(raw):
    comparison = raw.get("comparison") or build_comparison(raw)
    modes = comparison["modes"]
    source_counts = {}
    for paper in raw["manifest"]["papers"]:
        source = ", ".join(paper.get("sources", [])) or "未标注"
        source_counts[source] = source_counts.get(source, 0) + 1
    source_summary = "，".join(f"{key} {value} 篇" for key, value in source_counts.items())
    lines = [
        "# 论文流水线并发性能对比报告",
        "",
        "> 本报告由真实 Benchmark 自动生成；所有数字均来自本次运行，未执行的实验不会生成结果。",
        "",
        "![各轮总耗时](01-round-duration.svg)",
        "",
        f"- 生成时间：{raw.get('finished_at') or datetime.now(timezone.utc).isoformat()}",
        f"- 测试集：{raw['manifest']['name']}",
        f"- 测试集哈希：`{raw['manifest_sha256']}`",
        f"- 论文数量：{len(raw['manifest']['papers'])}（{source_summary}）",
        "- 计时：`time.perf_counter()` 单调时钟",
        "- 执行顺序：两种模式严格串行；偶数轮反转模式顺序以降低先后顺序偏差",
    ]
    revision = raw.get("revision")
    if revision:
        lines.extend([
            "",
            "## 修订说明",
            "",
            f"- {revision['summary']}",
            f"- 被替换记录：`{revision['replaced_run']}`",
            f"- 补跑记录：`{revision['replacement_run']}`",
            "- 原始结果和失败样本均保留，未被覆盖或删除。",
        ])
    lines.extend([
        "",
        "## 测试集",
        "",
        "| ID | 论文 | 来源 | PDF |",
        "|---|---|---|---|",
    ])
    for paper in raw["manifest"]["papers"]:
        source = ", ".join(paper.get("sources", [])) or "未标注"
        lines.append(f"| {paper['id']} | {paper['title']} | {source} | [公开链接]({paper['pdf_urls'][0]}) |")
    lines.extend([
        "",
        "## 配置",
        "",
        "| 参数 | 值 |",
        "|---|---:|",
    ])
    for key, value in raw["pipeline_config"].items():
        lines.append(f"| {key} | {value} |")
    lines.extend([
        "",
        "模型、Prompt、重试与超时配置在所有轮次保持一致：",
        "",
        "```json",
        json.dumps(raw["model_config"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## 总体结果",
        "",
        "| 模式 | 轮数 | 总耗时(s) | 中位数(s) | 波动范围(s) | 吞吐量(篇/s) | 成功/尝试 | 失败率 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for mode in ("legacy", "staged"):
        if mode not in modes:
            continue
        row = modes[mode]
        spread = f"{_number(row['min_seconds'])}–{_number(row['max_seconds'])}"
        lines.append(
            f"| {mode} | {row['rounds']} | {_number(row['total_seconds'])} | "
            f"{_number(row['median_seconds'])} | {spread} | "
            f"{_number(row['throughput_papers_per_second'], 4)} | "
            f"{row['completed_count']}/{row['attempted_count']} | {_percent(row['failure_rate'])} |"
        )
    lines.extend([
        "",
        "![总体吞吐量](04-throughput.svg)",
        "",
        "计算公式：",
        "",
        "- 耗时改善率 = `(legacy 中位耗时 - staged 中位耗时) / legacy 中位耗时 × 100%`。",
        "- 失败率 = `最终失败论文数 / 尝试处理论文总数 × 100%`；重试后最终成功不计失败。",
        "",
        "## 各轮结果",
        "",
        "| 轮次 | 模式 | 总耗时(s) | 吞吐量(篇/s) | 成功 | 失败 | 失败率 | 输出目录 |",
        "|---:|---|---:|---:|---:|---:|---:|---|",
    ])
    for row in sorted(raw["runs"], key=lambda item: item["sequence"]):
        metrics = row["metrics"]
        lines.append(
            f"| {row['repeat']} | {row['mode']} | {_number(row['wall_seconds'])} | "
            f"{_number(metrics['throughput_papers_per_second'], 4)} | "
            f"{metrics['completed_count']} | {metrics['failed_count']} | "
            f"{_percent(metrics['failure_rate'])} | `{row['output_dir']}` |"
        )
    lines.extend([
        "",
        "## 单篇论文耗时",
        "",
        "| 轮次 | 模式 | 论文 | 状态 | 总耗时(s) | 错误 |",
        "|---:|---|---|---|---:|---|",
    ])
    for row in sorted(raw["runs"], key=lambda item: item["sequence"]):
        for paper in row["papers"]:
            error = str(paper.get("error", "")).replace("|", "\\|").replace("\n", " ")
            lines.append(
                f"| {row['repeat']} | {row['mode']} | {paper['id']} | {paper['status']} | "
                f"{_number(paper['seconds'])} | {error or '—'} |"
            )
    lines.extend([
        "",
        "## 阶段执行、等待与失败",
        "",
        "以下时间为每轮各论文阶段时间之和；阶段会重叠，因此不能相加得到总耗时。",
        "",
        "![分阶段累计执行时间](02-stage-execution.svg)",
        "",
        "![分阶段信号量等待时间](03-stage-wait.svg)",
        "",
        "| 轮次 | 模式 | 阶段 | 执行(s) | 等待(s) | 尝试 | 阶段失败 |",
        "|---:|---|---|---:|---:|---:|---:|",
    ])
    for row in sorted(raw["runs"], key=lambda item: item["sequence"]):
        scheduler_wait = row["metrics"].get("scheduler_wait_seconds", 0.0)
        lines.append(
            f"| {row['repeat']} | {row['mode']} | whole-paper scheduler | — | "
            f"{_number(scheduler_wait)} | {row['metrics']['attempted_count']} | 0 |"
        )
        for stage in STAGES:
            stage_row = row["metrics"]["stage_metrics"][stage]
            lines.append(
                f"| {row['repeat']} | {row['mode']} | {stage} | "
                f"{_number(stage_row['seconds'])} | {_number(stage_row['wait_seconds'])} | "
                f"{stage_row['attempts']} | {stage_row['failures']} |"
            )
    lines.extend(["", "## 结论", ""])
    improvement = comparison.get("improvement_rate")
    failure_delta = comparison.get("failure_rate_delta")
    if improvement is None:
        lines.append("尚未同时取得 legacy 与 staged 的实测结果，不能判断优化是否有效。")
    else:
        direction = "缩短" if improvement >= 0 else "增加"
        lines.append(
            f"按两种模式总耗时中位数计算，staged 相对 legacy 的耗时{direction} "
            f"**{abs(improvement) * 100:.2f}%**。"
        )
        if improvement > 0:
            lines.append("在本测试集与当前外部服务状态下，分阶段并发的耗时优化有效。")
        else:
            lines.append("在本测试集与当前外部服务状态下，分阶段并发未带来耗时改善。")
        if failure_delta and failure_delta > 0:
            lines.append(f"代价：staged 的最终失败率上升了 {failure_delta * 100:.2f} 个百分点。")
        elif failure_delta and failure_delta < 0:
            lines.append(f"staged 的最终失败率下降了 {abs(failure_delta) * 100:.2f} 个百分点。")
        else:
            lines.append("未观察到 staged 的最终失败率上升。")
    lines.extend([
        "",
        "最终失败仅按论文最终状态统计；中途发生阶段失败或内部重试、但最终成功的论文不会计入最终失败。",
        "",
        "## 实验限制",
        "",
    ])
    for limitation in raw["limitations"]:
        lines.append(f"- {limitation}")
    return "\n".join(lines) + "\n"
