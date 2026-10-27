"""Paired end-to-end A/B for pipeline changes such as PR #2's staged concurrency.

Real `main_agent.py`, real network, real model. Network speed drifts and arXiv throttles, so a
before/after pair measured an hour apart proves nothing. This runner therefore
  * runs every variant on the SAME query back to back, in a random order per pair,
  * gives every run an empty PDF cache (downloads and conversion are part of what is measured),
  * repeats the pairs, and
  * includes a control: an identical copy of the baseline. The spread between baseline and control
    is the noise floor; a difference smaller than that is not a result.

variants.json (first entry is the baseline):
  [{"label": "base",         "repo": "/path/to/base-checkout", "concurrent": true},
   {"label": "base-control", "repo": "/path/to/base-checkout", "concurrent": true, "control": true},
   {"label": "pr2-staged",   "repo": "/path/to/pr2-checkout",  "concurrent": true,
    "env": {"PIPELINE_MODE": "staged"}}]
`benchmarks.make_variants` writes such a file for the base / PR #2 / PR #3 / both checkouts.

    python -m benchmarks.ab_e2e --variants variants.json --limit 8 --repeats 2 --out outputs/ab-pr2
Runs a variant needs 2 server slots when it processes papers concurrently (LLAMA_PARALLEL=2); keep the
server configuration identical for every variant.
"""
import argparse
import asyncio
import json
import random
import statistics
import sys
import time
from pathlib import Path
from dotenv import load_dotenv
from benchmarks import preflight
from benchmarks.build_dataset import QUERIES_PATH, load_queries
from benchmarks.rank_eval import paired_bootstrap
from benchmarks.run_benchmark import collect_environment, read_rows, run_query_e2e, save_json, stratified
from benchmarks.summary import stats

NOISE_NOTE = "对照组（control）与基线配置完全相同，二者的差距就是噪声；B 与 A 的差距要明显大于它才算有效。"


def load_variants(path):
    variants = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(variants, list) or len(variants) < 2:
        raise ValueError("variants 至少要有基线和一个对比版本")
    labels = [v.get("label") for v in variants]
    if len(set(labels)) != len(labels) or not all(labels):
        raise ValueError("每个 variant 需要唯一的 label")
    for v in variants:
        repo = Path(v.get("repo", ""))
        if not (repo / "main_agent.py").is_file():
            raise ValueError(f"{v['label']}: repo 下没有 main_agent.py：{repo}")
    if variants[0].get("control"):
        raise ValueError("第一个 variant 是基线，不能是 control")
    return variants


def build_plan(variants, queries, repeats, seed):
    """[(repeat, query, variant)] - queries in stratified order, variants shuffled within each pair."""
    plan = []
    for repeat in range(repeats):
        for query in stratified(queries, seed + repeat):
            order = random.Random(f"{seed}-{repeat}-{query['qid']}").sample(variants, len(variants))
            plan += [(repeat, query, variant, position) for position, variant in enumerate(order)]
    return plan


def row_key(row):
    return (row["variant"], row["repeat"], row["qid"])


def paired(rows, baseline, other, field="query_seconds"):
    """Per-pair differences (other - baseline) over pairs where both runs succeeded."""
    base = {(r["repeat"], r["qid"]): r for r in rows if r["variant"] == baseline}
    diffs, ratios, dropped = [], [], 0
    for r in rows:
        if r["variant"] != other:
            continue
        b = base.get((r["repeat"], r["qid"]))
        if not b or not (b.get("task_success") and r.get("task_success")):
            dropped += 1
            continue
        x, y = b.get(field), r.get(field)
        if x is None or y is None:
            dropped += 1
            continue
        diffs.append(y - x)
        ratios.append(y / x if x else None)
    return diffs, [v for v in ratios if v is not None], dropped


def summarize(rows, variants):
    labels = [v["label"] for v in variants]
    baseline = labels[0]
    controls = [v["label"] for v in variants if v.get("control")]
    out = {"baseline": baseline, "controls": controls, "variants": {}, "vs_baseline": {}}
    for label in labels:
        mine = [r for r in rows if r["variant"] == label]
        ok = [r for r in mine if r.get("task_success")]
        out["variants"][label] = {
            "n": len(mine), "success_rate": round(len(ok) / len(mine), 4) if mine else None,
            "query_seconds": stats([r["query_seconds"] for r in ok]),
            "processing_seconds": stats([r["stages"].get("processing") for r in ok if r.get("stages")])}
    for label in labels[1:]:
        diffs, ratios, dropped = paired(rows, baseline, label)
        entry = {"pairs": len(diffs), "dropped_pairs": dropped}
        if diffs:
            low, high = paired_bootstrap(diffs, 5000)
            entry.update(mean_diff_seconds=round(statistics.fmean(diffs), 3),
                         ci95=[round(low, 3), round(high, 3)],
                         median_ratio=round(statistics.median(ratios), 4) if ratios else None,
                         faster=sum(d < -1e-9 for d in diffs), slower=sum(d > 1e-9 for d in diffs))
        out["vs_baseline"][label] = entry
    noise = [out["vs_baseline"][c] for c in controls if out["vs_baseline"][c].get("pairs")]
    out["noise_floor_seconds"] = (round(max(abs(n["mean_diff_seconds"]) for n in noise), 3) if noise else None)
    return out


def validity(summary, variants):
    reasons, warnings = [], []
    for label, v in summary["variants"].items():
        if not v["n"]:
            reasons.append(f"{label} 没有任何运行")
        elif not v["success_rate"]:
            reasons.append(f"{label} 的任务成功率为 0")
        elif v["success_rate"] < 0.8:
            warnings.append(f"{label} 成功率只有 {v['success_rate']}，配对样本会减少")
    if not summary["controls"]:
        warnings.append("没有 control 对照组，无法估计网络波动造成的噪声")
    for label, entry in summary["vs_baseline"].items():
        if entry.get("pairs") is not None and entry["pairs"] < 5:
            warnings.append(f"{label} 只有 {entry['pairs']} 个有效配对，结论不可靠")
        if entry["pairs"] and label not in summary["controls"] and summary["noise_floor_seconds"] is not None \
                and abs(entry["mean_diff_seconds"]) <= summary["noise_floor_seconds"]:
            warnings.append(f"{label} 与基线的差距（{entry['mean_diff_seconds']} s）不大于噪声（{summary['noise_floor_seconds']} s）")
    return {"ok": not reasons, "reasons": reasons, "warnings": warnings}


def render_markdown(summary, check, meta):
    lines = [f"# 端到端 A/B `{meta['label']}`", ""]
    if not check["ok"]:
        lines += ["> **结果无效：** " + "；".join(check["reasons"]), ""]
    if check["warnings"]:
        lines += ["> 注意：" + "；".join(check["warnings"]), ""]
    lines += [f"- 查询 {meta['queries']} 条 x 重复 {meta['repeats']} 次；每次运行使用全新 PDF 缓存；基线 `{summary['baseline']}`",
              f"- {NOISE_NOTE}", "", "## 各版本", "| 版本 | 运行数 | 成功率 | 平均(s) | 中位数(s) | P95(s) |", "|---|---|---|---|---|---|"]
    for label, v in summary["variants"].items():
        q = v["query_seconds"]
        cells = [q.get(k, "-") for k in ("mean", "median", "p95")] if q.get("n") else ["-"] * 3
        lines.append(f"| {label}{' (control)' if label in summary['controls'] else ''} | {v['n']} | "
                     f"{v['success_rate']} | {cells[0]} | {cells[1]} | {cells[2]} |")
    lines += ["", "## 相对基线的配对差值（对比版本 - 基线，负数表示更快）",
              "| 版本 | 有效配对 | 丢弃配对 | 平均差(s) | 95% 区间(s) | 中位耗时比 | 更快/更慢 |", "|---|---|---|---|---|---|---|"]
    for label, e in summary["vs_baseline"].items():
        if e["pairs"]:
            lines.append(f"| {label} | {e['pairs']} | {e['dropped_pairs']} | {e['mean_diff_seconds']:+} | "
                         f"[{e['ci95'][0]:+}, {e['ci95'][1]:+}] | {e['median_ratio']} | {e['faster']}/{e['slower']} |")
        else:
            lines.append(f"| {label} | 0 | {e['dropped_pairs']} | - | - | - | - |")
    if summary["noise_floor_seconds"] is not None:
        lines += ["", f"噪声基线：control 与基线的平均差为 {summary['noise_floor_seconds']} s。"]
    return "\n".join(lines) + "\n"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variants", required=True)
    p.add_argument("--label", default="ab")
    p.add_argument("--queries", default=str(QUERIES_PATH))
    p.add_argument("--limit", type=int, default=8, help="每轮用多少条查询（按 top_k 分层抽取）")
    p.add_argument("--top-k", type=int, choices=range(1, 6))
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", help="输出目录；指向已有目录即续跑")
    p.add_argument("--timeout", type=int, default=1800)
    p.add_argument("--skip-preflight", action="store_true")
    return p.parse_args(argv)


async def run(args, runner=run_query_e2e):
    variants = load_variants(args.variants)
    all_rows, dataset_hash = load_queries(args.queries)
    pool = [q for q in all_rows if not args.top_k or q["top_k"] == args.top_k]
    queries = stratified(pool, args.seed)[:args.limit]
    out = Path(args.out) if args.out else Path("outputs") / f"ab-{args.label}-{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    results = out / "results.jsonl"
    done = {row_key(r) for r in read_rows(results)}
    plan = build_plan(variants, queries, args.repeats, args.seed)

    meta_path = out / "run_meta.json"
    meta = {"label": args.label, "queries": len(queries), "repeats": args.repeats, "seed": args.seed,
            "dataset_hash": dataset_hash, "variants": variants, "qids": [q["qid"] for q in queries]}
    if meta_path.is_file():
        old = json.loads(meta_path.read_text(encoding="utf-8"))
        for key in ("repeats", "seed", "dataset_hash", "variants", "qids"):
            if old.get(key) != meta[key]:
                raise SystemExit(f"断点续跑参数与已有 run_meta.json 不一致：{key}")
    else:
        meta.update(started_at=time.strftime("%Y-%m-%d %H:%M:%S"), environment=collect_environment())
        save_json(meta_path, meta)

    pending = [item for item in plan if (item[2]["label"], item[0], item[1]["qid"]) not in done]
    print(f"{len(variants)} 个版本 x {len(queries)} 条查询 x {args.repeats} 轮 = {len(plan)} 次运行，待跑 {len(pending)}", flush=True)
    for index, (repeat, query, variant, position) in enumerate(pending, 1):
        row = await runner(query, out, bool(variant.get("concurrent")), args.timeout, repo=variant["repo"],
                           env={k: str(v) for k, v in (variant.get("env") or {}).items()},
                           args=variant.get("args") or (), name=f"{variant['label']}/r{repeat}-{query['qid']}")
        row.update(variant=variant["label"], repeat=repeat, position=position)
        with results.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[{index}/{len(pending)}] {variant['label']:<14} r{repeat} {query['qid']} k={query['top_k']} "
              f"{row['query_seconds']}s {'ok' if row.get('task_success') else 'FAILED'}", flush=True)

    rows = read_rows(results)
    summary = summarize(rows, variants)
    check = validity(summary, variants)
    summary["validity"] = check
    save_json(out / "summary.json", summary)
    (out / "summary.md").write_text(render_markdown(summary, check, meta), encoding="utf-8")
    print(render_markdown(summary, check, meta))
    print("结果目录：" + str(out))
    return summary


def main(argv=None, runner=run_query_e2e):
    args = parse_args(argv)
    load_dotenv()
    if not args.skip_preflight:
        variants = load_variants(args.variants)
        needs_two = any(v.get("concurrent") or (v.get("env") or {}).get("PIPELINE_MODE") == "staged" for v in variants)
        report = preflight.check(full=True, workers=1, concurrent=needs_two)
        print(preflight.render(report), flush=True)
        if not report.ok:
            return 1
    summary = asyncio.run(run(args, runner))
    return 0 if summary["validity"]["ok"] else 2


if __name__ == "__main__":
    sys.stdout.reconfigure(errors="replace")
    sys.exit(main())
