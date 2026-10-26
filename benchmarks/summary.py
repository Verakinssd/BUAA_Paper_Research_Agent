"""Aggregate benchmark rows into summary.json / summary.md."""
import statistics


def percentile(values, q):
    ordered = sorted(values)
    if not ordered:
        return None
    pos = (len(ordered) - 1) * q
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (pos - low)


def stats(values):
    values = [v for v in values if v is not None]
    if not values:
        return {"n": 0}
    return {"n": len(values), "mean": round(statistics.fmean(values), 3),
            "median": round(statistics.median(values), 3),
            "p95": round(percentile(values, 0.95), 3),
            "min": round(min(values), 3), "max": round(max(values), 3)}


def fit_line(xs, ys):
    """Least squares y = slope * x + intercept; None if x has no spread."""
    n = len(xs)
    if n < 2 or len(set(xs)) < 2:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = my - slope * mx
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
    r2 = 1 - ss_res / ss_tot if ss_tot else 1.0
    return {"slope_seconds_per_paper": round(slope, 3), "intercept_seconds": round(intercept, 3),
            "r2": round(r2, 4)}


def group(rows, field):
    groups = {}
    for row in rows:
        groups.setdefault(row.get(field), []).append(row["query_seconds"])
    return {str(k): stats(v) for k, v in sorted(groups.items(), key=lambda kv: str(kv[0]))}


def llm_totals(rows):
    """Sum per-stage LLM counters and derive token speeds."""
    stages, total = {}, {}
    for row in rows:
        for stage, counters in (row.get("llm") or {}).items():
            for target in (stages.setdefault(stage, {}), total):
                for key, value in counters.items():
                    target[key] = target.get(key, 0) + value

    def speeds(c):
        out = dict(c)
        out["prefill_tps_server"] = (round(c["prompt_n"] / (c["prompt_ms"] / 1000), 2)
                                     if c.get("prompt_ms") else None)
        out["decode_tps_server"] = (round(c["predicted_n"] / (c["predicted_ms"] / 1000), 2)
                                    if c.get("predicted_ms") else None)
        # Includes prefill and HTTP overhead, so it understates pure decode speed.
        out["completion_tps_overall"] = (round(c["completion_tokens"] / c["seconds"], 2)
                                         if c.get("seconds") else None)
        for key in ("seconds", "prompt_ms", "predicted_ms"):
            if key in out:
                out[key] = round(out[key], 3)
        return out
    return {"total": speeds(total) if total else {},
            "by_stage": {s: speeds(c) for s, c in sorted(stages.items())}}


def summarize(rows, meta):
    rows = [r for r in rows if not r.get("warmup")]
    times = [r["query_seconds"] for r in rows]
    stage_sums = {}
    for row in rows:
        for name, seconds in (row.get("stages") or {}).items():
            stage_sums[name] = stage_sums.get(name, 0.0) + seconds
    total_stage = sum(stage_sums.values())
    repeats = {}
    for row in rows:
        repeats.setdefault(row.get("repeat", 0), []).append(row["query_seconds"])
    repeat_means = [round(statistics.fmean(v), 3) for _, v in sorted(repeats.items())]
    overlaps = [r["select_overlap"] for r in rows if r.get("select_overlap") is not None]
    ok = [r for r in rows if r.get("task_success")]
    return {
        "meta": meta,
        "n_queries": len(rows),
        "success_rate": round(len(ok) / len(rows), 4) if rows else None,
        "overall": stats(times),
        "overall_success_only": stats([r["query_seconds"] for r in ok]),
        "by_top_k": group(rows, "top_k"),
        "by_lang": group(rows, "lang"),
        "by_repeat_mean": repeat_means,
        "repeat_mean_stdev": (round(statistics.stdev(repeat_means), 3)
                              if len(repeat_means) > 1 else None),
        "stages": {name: {"mean_seconds": round(s / len(rows), 3),
                          "share": round(s / total_stage, 4) if total_stage else None}
                   for name, s in sorted(stage_sums.items())} if rows else {},
        "fit_time_vs_top_k": fit_line([r["top_k"] for r in rows], times),
        "select_overlap_mean": round(statistics.fmean(overlaps), 4) if overlaps else None,
        "llm": llm_totals(rows),
    }


def _row(label, s):
    if not s.get("n"):
        return f"| {label} | 0 | - | - | - | - | - |"
    return f"| {label} | {s['n']} | {s['mean']} | {s['median']} | {s['p95']} | {s['min']} | {s['max']} |"


def render_markdown(summary):
    meta = summary["meta"]
    lines = [f"# Benchmark `{meta.get('label')}`", "",
             f"- 模式：{meta.get('mode')}（负载：{meta.get('workload')}，并发：{meta.get('concurrent')}）",
             f"- 数据集哈希：`{meta.get('dataset_hash')}`；完整数据集：{meta.get('full_dataset')}",
             f"- 有效查询数：{summary['n_queries']}；任务成功率：{summary['success_rate']}", ""]
    lines += ["## 查询时间", "| 分组 | n | 平均(s) | 中位数(s) | P95(s) | 最小(s) | 最大(s) |",
              "|---|---|---|---|---|---|---|", _row("全部", summary["overall"])]
    lines += [_row(f"top_k={k}", s) for k, s in summary["by_top_k"].items()]
    lines += [_row(f"lang={k}", s) for k, s in summary["by_lang"].items()]
    if summary["stages"]:
        lines += ["", "## 阶段拆分", "| 阶段 | 平均(s) | 占比 |", "|---|---|---|"]
        lines += [f"| {n} | {s['mean_seconds']} | {s['share']} |" for n, s in summary["stages"].items()]
    if summary["fit_time_vs_top_k"]:
        f = summary["fit_time_vs_top_k"]
        lines += ["", f"耗时 ≈ {f['slope_seconds_per_paper']} s × top_k + {f['intercept_seconds']} s（R2={f['r2']}）"]
    total = summary["llm"]["total"]
    if total:
        lines += ["", "## LLM 调用",
                  f"- 调用数：{total.get('calls')}；prompt tokens：{total.get('prompt_tokens')}；"
                  f"completion tokens：{total.get('completion_tokens')}；被截断：{total.get('truncated')}",
                  f"- prefill：{total['prefill_tps_server']} tok/s（server）；decode：{total['decode_tps_server']} tok/s（server）；"
                  f"整体 completion：{total['completion_tps_overall']} tok/s"]
    if summary["select_overlap_mean"] is not None:
        lines += [f"- select 与参考选择平均重合率：{summary['select_overlap_mean']}"]
    if summary["repeat_mean_stdev"] is not None:
        lines += [f"- 各重复轮平均耗时：{summary['by_repeat_mean']}，标准差 {summary['repeat_mean_stdev']}"]
    return "\n".join(lines) + "\n"
