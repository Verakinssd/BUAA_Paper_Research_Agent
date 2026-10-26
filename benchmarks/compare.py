"""Compare two benchmark summaries (e.g. Q4 vs Q5).

Run: python -m benchmarks.compare outputs/bench-q4-*/summary.json outputs/bench-q5-*/summary.json
"""
import argparse
import json
from pathlib import Path

MUST_MATCH = ("dataset_hash", "mode", "workload", "workers")
DEFAULTS = {"workers": 1}  # summaries written before --workers existed


def check_comparable(a, b):
    """Return warnings; raise ValueError if the two runs measure different things."""
    for name, summary in (("A", a), ("B", b)):
        if not summary["meta"].get("full_dataset"):
            raise ValueError(f"{name} 不是完整 100 条数据集的结果，不能用于正式对比")
    for key in MUST_MATCH:
        left, right = a["meta"].get(key, DEFAULTS.get(key)), b["meta"].get(key, DEFAULTS.get(key))
        if left != right:
            raise ValueError(f"{key} 不一致：{left} vs {right}")
    return [f"{key} 不一致：{a['meta'].get(key)} vs {b['meta'].get(key)}"
            for key in ("concurrent", "repeats") if a["meta"].get(key) != b["meta"].get(key)]


def _cell(value):
    return "-" if value is None else str(value)


def _line(label, x, y, lower_is_better=True):
    ratio = round(y / x, 3) if x and y is not None else None
    return f"| {label} | {_cell(x)} | {_cell(y)} | {_cell(ratio)} |"


def compare(a, b):
    warnings = check_comparable(a, b)
    la, lb = a["meta"]["label"], b["meta"]["label"]
    lines = [f"# 对比：`{la}` vs `{lb}`", "", f"- 数据集哈希：`{a['meta']['dataset_hash']}`"]
    lines += [f"- 警告：{w}" for w in warnings]
    lines += ["", f"| 指标 | {la} | {lb} | {lb}/{la} |", "|---|---|---|---|"]
    lines += [_line("任务成功率", a["success_rate"], b["success_rate"])]
    for stat in ("mean", "median", "p95"):
        lines += [_line(f"查询时间 {stat}(s)", a["overall"].get(stat), b["overall"].get(stat))]
    for k in sorted(set(a["by_top_k"]) | set(b["by_top_k"])):
        lines += [_line(f"top_k={k} 平均(s)", a["by_top_k"].get(k, {}).get("mean"),
                        b["by_top_k"].get(k, {}).get("mean"))]
    for lang in sorted(set(a["by_lang"]) | set(b["by_lang"])):
        lines += [_line(f"lang={lang} 平均(s)", a["by_lang"].get(lang, {}).get("mean"),
                        b["by_lang"].get(lang, {}).get("mean"))]
    for stage in sorted(set(a["stages"]) | set(b["stages"])):
        lines += [_line(f"阶段 {stage} 平均(s)", a["stages"].get(stage, {}).get("mean_seconds"),
                        b["stages"].get(stage, {}).get("mean_seconds"))]
    ta, tb = a["llm"]["total"], b["llm"]["total"]
    for key in ("prefill_tps_server", "decode_tps_server", "completion_tps_overall"):
        lines += [_line(key + " (tok/s)", ta.get(key), tb.get(key))]
    tha, thb = a.get("throughput") or {}, b.get("throughput") or {}
    lines += [_line("吞吐 (条/小时)", tha.get("queries_per_hour"), thb.get("queries_per_hour"))]
    lines += [_line("select 与参考重合率", a["select_overlap_mean"], b["select_overlap_mean"])]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("a")
    parser.add_argument("b")
    args = parser.parse_args()
    load = lambda path: json.loads(Path(path).read_text(encoding="utf-8"))
    try:
        print(compare(load(args.a), load(args.b)))
    except ValueError as exc:
        raise SystemExit("无法对比：" + str(exc))


if __name__ == "__main__":
    main()
