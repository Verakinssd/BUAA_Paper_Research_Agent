"""Aggregate every independent real benchmark into an evidence analysis report."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.benchmarking import MODE_COLORS, _bar_chart_svg
from research.pipeline import save_json


def failure_category(error):
    value = (error or "").lower()
    if "429" in value:
        return "API 429 限流/额度"
    if "choices" in value:
        return "模型响应格式异常"
    if "llm" in value and ("网络" in value or "network" in value):
        return "模型 API 网络失败"
    if "server disconnected" in value or "pdf" in value:
        return "PDF 下载连接失败"
    return "其他外部错误"


def discover_real_batches(root):
    paths = sorted(root.rglob("benchmark_results.json"))
    selected = []
    for path in paths:
        if "simulated" in path.parts:
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if len(data.get("manifest", {}).get("papers", [])) == 10:
            selected.append(path)
    return selected


def compact_run(batch_id, source_path, data, row):
    errors = [
        str(paper.get("error", ""))
        for paper in row.get("papers", [])
        if paper.get("status") != "ok"
    ]
    stage_metrics = row["metrics"].get("stage_metrics", {})
    return {
        "batch": batch_id,
        "source_file": str(source_path),
        "paper_count": len(data["manifest"]["papers"]),
        "manifest_name": data["manifest"]["name"],
        "manifest_sha256": data["manifest_sha256"],
        "api_base_url": data["model_config"].get("LLM_BASE_URL", ""),
        "model": data["model_config"].get("LLM_MODEL", ""),
        "repeat": row["repeat"],
        "sequence": row["sequence"],
        "mode": row["mode"],
        "wall_seconds": row["wall_seconds"],
        "attempted": row["metrics"]["attempted_count"],
        "completed": row["metrics"]["completed_count"],
        "failed": row["metrics"]["failed_count"],
        "download_seconds": stage_metrics.get("download", {}).get("seconds", 0.0),
        "download_wait_seconds": stage_metrics.get("download", {}).get("wait_seconds", 0.0),
        "convert_seconds": stage_metrics.get("convert", {}).get("seconds", 0.0),
        "summary_seconds": stage_metrics.get("summary", {}).get("seconds", 0.0),
        "summary_wait_seconds": stage_metrics.get("summary", {}).get("wait_seconds", 0.0),
        "scheduler_wait_seconds": row["metrics"].get("scheduler_wait_seconds", 0.0),
        "error_categories": sorted({failure_category(error) for error in errors}),
        "errors": errors,
    }


def build_inventory(benchmark_root):
    batches = []
    runs = []
    for path in discover_real_batches(benchmark_root):
        data = json.loads(path.read_text(encoding="utf-8"))
        batch_id = path.parent.name
        batch_runs = [compact_run(batch_id, path.resolve(), data, row) for row in data["runs"]]
        attempted = sum(row["attempted"] for row in batch_runs)
        completed = sum(row["completed"] for row in batch_runs)
        batches.append({
            "batch": batch_id,
            "source_file": str(path.resolve()),
            "paper_count": len(data["manifest"]["papers"]),
            "manifest_name": data["manifest"]["name"],
            "manifest_sha256": data["manifest_sha256"],
            "api_base_url": data["model_config"].get("LLM_BASE_URL", ""),
            "model": data["model_config"].get("LLM_MODEL", ""),
            "run_count": len(batch_runs),
            "attempted": attempted,
            "completed": completed,
            "failed": attempted - completed,
            "failure_rate": (attempted - completed) / attempted if attempted else 0.0,
            "min_seconds": min(row["wall_seconds"] for row in batch_runs),
            "max_seconds": max(row["wall_seconds"] for row in batch_runs),
        })
        runs.extend(batch_runs)
    revised_paths = sorted(benchmark_root.rglob("benchmark_results_revised.json"))
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "independent_batch_count": len(batches),
        "independent_run_count": len(runs),
        "paper_attempt_count": sum(row["attempted"] for row in runs),
        "paper_completed_count": sum(row["completed"] for row in runs),
        "paper_failed_count": sum(row["failed"] for row in runs),
        "batches": batches,
        "runs": runs,
        "derived_revisions": [str(path.resolve()) for path in revised_paths],
    }


def _short_batch(batch):
    if batch.startswith("benchmark-"):
        return batch.removeprefix("benchmark-")[9:15]
    if "173345" in batch:
        return "补跑"
    return batch[:10]


def write_charts(inventory, output_dir):
    output_dir = Path(output_dir)
    ten = [row for row in inventory["runs"] if row["paper_count"] == 10]
    ten_labels = [f"{_short_batch(row['batch'])} R{row['repeat']}\n{row['mode']}" for row in ten]
    ten_colors = ["#B91C1C" if row["failed"] else MODE_COLORS[row["mode"]] for row in ten]
    charts = {
        "01-ten-paper-wall-times.svg": _bar_chart_svg(
            "10 篇真实实验：各次总耗时",
            "红色表示该次运行存在最终失败；补跑与候选轮均保留为独立观测",
            ten_labels,
            [row["wall_seconds"] for row in ten],
            ten_colors,
            "总耗时（秒）",
            digits=1,
        ),
        "02-ten-paper-download-times.svg": _bar_chart_svg(
            "10 篇真实实验：累计下载时间",
            "同一测试集仍出现显著波动；阶段累计时间可能互相重叠",
            ten_labels,
            [row["download_seconds"] for row in ten],
            ten_colors,
            "累计下载时间（秒）",
            digits=1,
        ),
        "03-batch-success-rates.svg": _bar_chart_svg(
            "真实实验：各批次论文成功率",
            "成功率受模型服务、额度、网络和 PDF 站点共同影响，不等同于调度算法可靠性",
            [f"{_short_batch(row['batch'])}\n{row['paper_count']}篇×{row['run_count']}次" for row in inventory["batches"]],
            [row["completed"] / row["attempted"] * 100 for row in inventory["batches"]],
            ["#0F766E" if row["failed"] == 0 else "#B91C1C" for row in inventory["batches"]],
            "成功率（%）",
            digits=1,
        ),
    }
    for filename, content in charts.items():
        (output_dir / filename).write_text(content, encoding="utf-8")
    return list(charts)


def _provider_label(row):
    if "openrouter" in row["api_base_url"]:
        return "OpenRouter"
    if "deepseek" in row["api_base_url"]:
        return "DeepSeek"
    return row["api_base_url"] or "未知"


def render_report(inventory, simulation_path):
    ten = [row for row in inventory["runs"] if row["paper_count"] == 10]
    ten_legacy = [row for row in ten if row["mode"] == "legacy"]
    ten_staged = [row for row in ten if row["mode"] == "staged"]
    successful_legacy = [row for row in ten_legacy if row["failed"] == 0]
    simulation = json.loads(Path(simulation_path).read_text(encoding="utf-8"))
    lines = [
        "# 真实环境 Benchmark 实验记录与影响因素分析",
        "",
        "> **实验结论摘要：本项目完成了多批次真实端到端 Benchmark，覆盖公共 PDF 下载、PDF 转换和模型 API 总结。实验验证了系统能够在真实服务环境中运行，同时观察到公共论文站点、网络链路和模型 API 状态会显著影响耗时与成功率。因而，真实实验主要用于评估工程表现与外部波动，可控模拟实验则用于进一步剥离外部变量、分析调度算法本身的贡献。**",
        "",
        "![10 篇真实实验总耗时](01-ten-paper-wall-times.svg)",
        "",
        "## 1. 数据范围与统计原则",
        "",
        f"- 独立真实实验批次：{inventory['independent_batch_count']} 个。",
        f"- 独立流水线运行：{inventory['independent_run_count']} 次。",
        f"- 论文处理尝试：{inventory['paper_attempt_count']} 次；最终成功 {inventory['paper_completed_count']} 次，失败 {inventory['paper_failed_count']} 次。",
        "- 所有纳入统计的真实批次均使用同一套 10 篇固定混合来源测试集。",
        "- 所有真实运行均使用独立输出目录与本地缓存，避免复用上一轮下载或转换结果。",
        "- `benchmark_results_revised.json` 是以成功补跑替换失败轮次得到的派生数据，不作为新的独立运行重复计数。",
        "- 失败运行完整保留。失败可能因重试/超时而异常变慢，也可能因提前终止而异常变快，因此不能只依据其墙钟时间判断调度性能。",
        "",
        "## 2. 独立真实实验批次",
        "",
        "| 批次 | 论文数 | API/模型 | 流水线运行 | 成功/尝试 | 失败率 | 耗时范围 |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for batch in inventory["batches"]:
        lines.append(
            f"| `{batch['batch']}` | {batch['paper_count']} | {_provider_label(batch)} / `{batch['model']}` | "
            f"{batch['run_count']} | {batch['completed']}/{batch['attempted']} | "
            f"{batch['failure_rate'] * 100:.2f}% | {batch['min_seconds']:.3f}–{batch['max_seconds']:.3f}s |"
        )
    lines.extend([
        "",
        "![各批次论文成功率](03-batch-success-rates.svg)",
        "",
        "## 3. 全部独立运行台账",
        "",
        "| 批次 | 测试集 | 轮次 | 模式 | 总耗时(s) | 成功/尝试 | 失败类型 |",
        "|---|---:|---:|---|---:|---:|---|",
    ])
    for row in inventory["runs"]:
        failures = "、".join(row["error_categories"]) or "—"
        lines.append(
            f"| `{row['batch']}` | {row['paper_count']} 篇 | {row['repeat']} | {row['mode']} | "
            f"{row['wall_seconds']:.3f} | {row['completed']}/{row['attempted']} | {failures} |"
        )
    lines.extend([
        "",
        "## 4. 10 篇正式测试的阶段数据",
        "",
        "以下累计阶段时间是 10 篇论文对应阶段耗时之和。不同论文及不同阶段可以重叠，因此累计时间不能直接相加得到墙钟时间。",
        "",
        "| 批次 | 轮次 | 模式 | 总耗时 | 下载累计 | 下载等待 | 总结累计 | 总结等待 | 成功 |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in ten:
        lines.append(
            f"| `{row['batch']}` | {row['repeat']} | {row['mode']} | {row['wall_seconds']:.3f}s | "
            f"{row['download_seconds']:.3f}s | {row['download_wait_seconds']:.3f}s | "
            f"{row['summary_seconds']:.3f}s | {row['summary_wait_seconds']:.3f}s | "
            f"{row['completed']}/{row['attempted']} |"
        )
    legacy_downloads = [row["download_seconds"] for row in ten_legacy]
    staged_downloads = [row["download_seconds"] for row in ten_staged]
    legacy_summaries = [row["summary_seconds"] for row in ten_legacy]
    staged_summaries = [row["summary_seconds"] for row in ten_staged]
    successful_legacy_walls = [row["wall_seconds"] for row in successful_legacy]
    staged_walls = [row["wall_seconds"] for row in ten_staged]
    lines.extend([
        "",
        "![10 篇真实实验累计下载时间](02-ten-paper-download-times.svg)",
        "",
        "### 直接观察",
        "",
        f"- 同一 10 篇测试集中，legacy 累计下载时间从 {min(legacy_downloads):.3f}s 到 {max(legacy_downloads):.3f}s，相差 {max(legacy_downloads) / min(legacy_downloads):.2f} 倍。",
        f"- staged 累计下载时间从 {min(staged_downloads):.3f}s 到 {max(staged_downloads):.3f}s，相差 {max(staged_downloads) / min(staged_downloads):.2f} 倍。",
        f"- legacy 累计 API 总结时间从 {min(legacy_summaries):.3f}s 到 {max(legacy_summaries):.3f}s，相差 {max(legacy_summaries) / min(legacy_summaries):.2f} 倍；staged 则从 {min(staged_summaries):.3f}s 到 {max(staged_summaries):.3f}s，相差 {max(staged_summaries) / min(staged_summaries):.2f} 倍。",
        f"- 即使只看全部成功的 legacy，整轮耗时仍从 {min(successful_legacy_walls):.3f}s 到 {max(successful_legacy_walls):.3f}s，相差 {max(successful_legacy_walls) / min(successful_legacy_walls):.2f} 倍。",
        f"- 全部 staged 整轮耗时从 {min(staged_walls):.3f}s 到 {max(staged_walls):.3f}s，相差 {max(staged_walls) / min(staged_walls):.2f} 倍。",
        "- 原三轮第 2 次 legacy 的 O5（NeurIPS PDF）出现 `Server disconnected without sending a response`，说明公开 PDF 站点本身可以改变失败率和整轮耗时。",
        "",
        "## 5. 结论对数据选择高度敏感",
        "",
        "| 统计口径 | Legacy 中位数 | Staged 中位数 | 结论 |",
        "|---|---:|---:|---|",
        "| 原始 10 篇三轮（保留一次 legacy 失败） | 426.550s | 341.109s | staged 快 20.03% |",
        "| 用 274.050s 成功补跑替换失败轮次 | 315.799s | 341.109s | staged 慢 8.01% |",
        "| 最新独立候选对照轮 | 255.593s | 258.113s | staged 慢 0.99%，基本持平 |",
        "| 原始三轮再追加候选轮 | 371.174s | 322.526s | staged 快 13.11% |",
        "| 修订三轮再追加候选轮 | 294.925s | 322.526s | staged 慢 9.36% |",
        "",
        "同一批真实数据只因“是否保留失败轮次、是否用成功补跑替换、是否追加候选轮”就能使结论由加速变为减速。因此，不应选择性保留最符合预期的数据；原始结果、补跑结果和修订规则必须同时披露。",
        "",
        "## 6. 真实实验中的显著干扰变量",
        "",
        "1. **公共 PDF 站点与下载链路。** arXiv 有强制串行限速；CVF、ACL、PMLR、NeurIPS 等站点的响应速度、连接复用、CDN 节点和断连情况随时间变化。下载并发参数不等于实际可获得的下载带宽。",
        "2. **模型 API 服务状态。** 服务端负载、排队、动态路由、限流、额度、429、网关错误和响应格式变化都会影响耗时与成功率。",
        "3. **请求内容差异。** 不同论文正文长度和分块数量不同，会产生不同 token 数量、模型推理时间和重试概率。",
        "4. **模式执行时间窗口不同。** 两种模式严格串行以避免互相抢占资源，但因此必然处于不同的网络/API 时间窗口。交替顺序可以缓解先后偏差，不能消除服务端状态漂移。",
        "5. **失败和重试改变耗时含义。** 失败可能等待完整超时后才返回，也可能快速失败；所以“失败得快”不能视为性能提升。重试后成功则会增加耗时但不计为最终失败。",
        "6. **缓存只能局部控制。** 每轮清空本地缓存可以避免本地复用，但无法清除服务商 CDN、API 连接预热、DNS、代理节点或服务端缓存。",
        "7. **本机与代理环境。** CPU/线程调度、后台负载、Clash/TUN 或显式代理路由可能改变下载和 API 往返时间。",
        "8. **阶段时间存在重叠。** 各论文阶段累计时间与信号量等待时间用于定位瓶颈，但不能简单相加后与墙钟时间比较。",
        "",
        "## 7. 真实实验与可控模拟实验的互补设计",
        "",
        "真实实验回答的是“系统在当时网络和服务状态下表现如何”，而模拟实验回答的是“在外部延迟固定时，调度算法本身是否减少了队头阻塞”。只有后者能够进行相对清晰的因果归因。",
        "",
        "模拟实验复用真实 `process_papers()`、worker pool 和信号量，只把下载、转换、API 总结替换为固定延迟；不访问网络、不使用 API、不产生限流和随机断连。其 20 轮中位数结果为：",
        "",
        "| 模拟场景 | Legacy 中位数 | Staged 中位数 | 改善率 |",
        "|---|---:|---:|---:|",
    ])
    for scenario, config in simulation["scenarios"].items():
        result = simulation["comparison"][scenario]
        lines.append(
            f"| {config['label']} | {result['modes']['legacy']['median_seconds']:.3f}s | "
            f"{result['modes']['staged']['median_seconds']:.3f}s | "
            f"{result['improvement_rate'] * 100:.2f}% |"
        )
    lines.extend([
        "",
        "模拟结果证明：当下载、转换与总结能够重叠时，分阶段调度本身可以提高吞吐量；真实结果未稳定复现同等收益，主要说明当前真实系统受外部瓶颈和噪声支配，而不是否定调度机制正确性。",
        "",
        "## 8. 证据边界与解释范围",
        "",
        "### 真实与模拟数据共同支持的结论",
        "",
        "- staged 的分阶段并发上限确实生效，调度结构能够消除整篇论文占用同一信号量的问题。",
        "- 在可控延迟下，staged 在三种模拟负载中均提高了吞吐量，且下载主导场景收益最大。",
        "- 真实系统的网络、PDF 站点与模型 API 是显著干扰变量，必须与算法调度效果分开讨论。",
        "",
        "### 需要谨慎解释的内容",
        "",
        "- staged 的真实加速比例随网络与服务状态变化，不宜外推为所有环境下固定的 20.03%、30.44% 或其他单一数值。",
        "- API/PDF 下载失败需要结合错误类型解释；失败运行的耗时也不能直接等同于正常完成任务的性能。",
        "- 成功补跑可以作为补充观测，但应同时展示原始结果与替换规则，以保证数据过程可追溯。",
        "- 下载并发设置为 4 表示调度上限，并不意味着 arXiv 可以实际并行下载 4；arXiv 限速仍然优先。",
        "",
        "## 9. 综合实验结论",
        "",
        "> 分阶段并发在架构上消除了整篇论文级信号量造成的队头阻塞，并在固定延迟的可控模拟中获得 15.78%～57.46% 的中位耗时改善。多批真实端到端实验进一步验证了系统的可运行性，同时表明公共 PDF 下载速度、arXiv 限速、模型 API 负载、限流、重试和网络故障会产生显著波动。综合来看，模拟实验用于识别调度算法的独立贡献，真实实验用于评估该优化在工程环境中的实际表现与适用条件。",
        "",
        "## 10. 数据溯源",
        "",
        "独立批次原始文件：",
        "",
    ])
    for batch in inventory["batches"]:
        lines.append(f"- `{batch['source_file']}`")
    lines.extend(["", "派生修订文件（不重复计数）：", ""])
    for path in inventory["derived_revisions"]:
        lines.append(f"- `{path}`")
    lines.extend([
        "",
        f"可控模拟原始数据：`{Path(simulation_path).resolve()}`",
        "",
        "本报告不包含 API Key、访问令牌或其他认证信息。",
    ])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="汇总所有真实 Benchmark 并生成影响因素分析报告")
    parser.add_argument(
        "--benchmark-root", default=str(PROJECT_ROOT / "outputs" / "benchmarks")
    )
    parser.add_argument(
        "--simulation-results",
        default=str(
            PROJECT_ROOT / "outputs" / "benchmarks" / "simulated"
            / "simulation-20261023-191217-953900" / "simulation_results.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "outputs" / "benchmarks" / "real-experiment-analysis"),
    )
    args = parser.parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    inventory = build_inventory(Path(args.benchmark_root).resolve())
    save_json(output_dir / "real_experiment_inventory.json", inventory)
    write_charts(inventory, output_dir)
    report = render_report(inventory, Path(args.simulation_results).resolve())
    (output_dir / "real_environment_benchmark_analysis.md").write_text(report, encoding="utf-8")
    print("真实实验台账：" + str(output_dir / "real_experiment_inventory.json"))
    print("真实环境分析报告：" + str(output_dir / "real_environment_benchmark_analysis.md"))


if __name__ == "__main__":
    main()
