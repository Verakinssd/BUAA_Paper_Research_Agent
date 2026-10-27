"""Offline Top-k relevance evaluation of the ranking pipeline (first-stage ranking, optional rerank).

It replays the frozen raw search results (rank_pools) through the *real* `retrieve()` of the
checkout being tested, so any ranking method or reranker that code version has is exercised
as shipped, and scores the result against the shared relevance labels (rank_labels).

  rank mode    : ordering quality of the whole pool - nDCG@k, P@k, MRR, and how many relevant
                 papers survive the Task Agent's candidate cut (recall@M).
  --select     : also runs the Task Agent's select() on what it would really be shown
                 and scores the picks, i.e. the task book's "Top-k relevance". Needs the model.

    python -m benchmarks.rank_eval run --label base --pool-limit 8
    python -m benchmarks.rank_eval run --label bm25 --pool-limit 25 --ranking-method bm25 --select
    python -m benchmarks.rank_eval run --label ce --rerank http --select     # PR #3 checkout only
    python -m benchmarks.rank_eval compare outputs/rankeval-base-*/results.json outputs/rankeval-ce-*/results.json

Two runs are comparable only on the same pools, labels, pool limit and mode; `compare` checks that and
reports paired differences per idea with a bootstrap confidence interval (the idea is the unit,
since the five top_k queries of one idea are not independent).
"""
import argparse
import asyncio
import importlib.util
import json
import math
import os
import random
import statistics
import sys
import time
from pathlib import Path
from dotenv import load_dotenv
from research import sources
from research.task_agent import TaskAgent
from benchmarks.build_dataset import TOP_KS
from benchmarks import rank_labels
from benchmarks.rank_labels import RELEVANT, label_id, labels_hash, merged_scores
from benchmarks.rank_pools import POOLS_PATH, load_pools, paper_key, replay

RANK_KS = {"ndcg": (3, 5, 10), "p": (1, 3, 5)}
RECALL_M_DEFAULT = 8
RANK_METRICS = [f"ndcg@{k}" for k in RANK_KS["ndcg"]] + [f"p@{k}" for k in RANK_KS["p"]] + ["mrr", "recall@M"]
SELECT_METRICS = ["mean_rel", "precision", "ndcg"]
DENSE_METHODS = ("dense_linear", "hybrid_rrf")
RERANK_CHOICES = ("none", "local", "http")


# --- metrics -------------------------------------------------------------------------------

def gain(score):
    return 2 ** score - 1


def dcg(scores):
    return sum(gain(s) / math.log2(i + 2) for i, s in enumerate(scores))


def ndcg_at(ordered, k, pool):
    """nDCG with the ideal ranking taken from the whole pool; None if nothing in the pool is relevant."""
    ideal = dcg(sorted(pool, reverse=True)[:k])
    return dcg(ordered[:k]) / ideal if ideal else None


def precision_at(ordered, k):
    return sum(s >= RELEVANT for s in ordered[:k]) / k


def reciprocal_rank(ordered):
    for i, s in enumerate(ordered):
        if s >= RELEVANT:
            return 1 / (i + 1)
    return 0.0


def recall_at(ordered, m, pool):
    relevant = sum(s >= RELEVANT for s in pool)
    return sum(s >= RELEVANT for s in ordered[:m]) / relevant if relevant else None


def rank_metrics(ordered, m):
    out = {f"ndcg@{k}": ndcg_at(ordered, k, ordered) for k in RANK_KS["ndcg"]}
    out.update({f"p@{k}": precision_at(ordered, k) for k in RANK_KS["p"]})
    out.update(mrr=reciprocal_rank(ordered), **{"recall@M": recall_at(ordered, m, ordered)})
    return out


def select_metrics(chosen, pool, k):
    return {"mean_rel": sum(chosen) / len(chosen) if chosen else 0.0,
            "precision": precision_at(chosen, k),
            "ndcg": ndcg_at(chosen, k, pool)}


def paired_bootstrap(diffs, resamples=10000, seed=0):
    """95% percentile interval for the mean of paired per-idea differences."""
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(sum(rng.choice(diffs) for _ in range(n)) / n for _ in range(resamples))
    return means[int(0.025 * resamples)], means[int(0.975 * resamples) - 1]


def mean_or_none(values):
    values = [v for v in values if v is not None]
    return statistics.fmean(values) if values else None


# --- running ---------------------------------------------------------------------------------

def agent_candidates(ranked, stats, k):
    """What the Task Agent is shown. Mirrors server.build_report on the PR #3 branch: after a
    successful rerank only the top RERANKER_AGENT_TOP_M candidates are passed to select()."""
    rerank = (stats or {}).get("reranking")
    if isinstance(rerank, dict) and rerank.get("status") == "ok" and rerank.get("method") != "none":
        try:
            limit = max(k, int(os.getenv("RERANKER_AGENT_TOP_M", "8")))
        except ValueError:
            limit = max(k, 8)
        return ranked[:min(len(ranked), limit)]
    return ranked


def recall_cut():
    """M for recall@M: how many top candidates the Task Agent sees after a rerank (PR #3's setting)."""
    try:
        return max(1, int(os.getenv("RERANKER_AGENT_TOP_M", str(RECALL_M_DEFAULT))))
    except ValueError:
        return RECALL_M_DEFAULT


def resolve_method(requested):
    from research.ranking import DEFAULT_METHOD, VALID_METHODS
    method = requested or os.getenv("PAPER_RANKING_METHOD") or DEFAULT_METHOD
    return method if method in VALID_METHODS else DEFAULT_METHOD


def configure_rerank(rerank):
    """Pin the reranker explicitly so a value left in .env cannot change what a run measures."""
    available = importlib.util.find_spec("research.reranking") is not None
    if rerank != "none" and not available:
        raise SystemExit("这个代码版本没有 research.reranking（精排在 PR #3 分支），不能用 --rerank " + rerank)
    if available:
        os.environ["PAPER_RERANKING_METHOD"] = rerank
    return available


async def evaluate(pools, scores, *, limit, method, recall_m=RECALL_M_DEFAULT, select=False, agent=None,
                   rerank="none", progress=print):
    ideas, issues = [], {"unlabeled": 0, "rerank_failed": [], "select_errors": 0}
    for idea in pools["ideas"].values():
        with replay(idea, limit):
            ranked, stats = await sources.retrieve(idea["search_query"], limit, ranking_method=method)
        rerank_stats = stats.get("reranking") if isinstance(stats, dict) else None
        if rerank != "none" and not (isinstance(rerank_stats, dict) and rerank_stats.get("status") == "ok"):
            issues["rerank_failed"].append(idea["qkey"])  # a silent fallback would measure the wrong thing
        labelled = []
        for paper in ranked:
            score = scores.get(label_id(idea["qkey"], paper_key(paper)))
            if score is None:
                issues["unlabeled"] += 1
            labelled.append(0 if score is None else score)
        by_key = {paper_key(p): s for p, s in zip(ranked, labelled)}
        entry = {"qkey": idea["qkey"], "topic": idea["topic"], "lang": idea["lang"],
                 "search_query": idea["search_query"], "n_pool": len(ranked),
                 "n_relevant": sum(s >= RELEVANT for s in labelled),
                 "metrics": rank_metrics(labelled, recall_m), "reranking": rerank_stats}
        if select:
            entry["select"] = {}
            for k in TOP_KS:
                shown = agent_candidates(ranked, stats, k)
                try:
                    chosen = await agent.select(idea["idea"], shown, k)
                except Exception as exc:
                    issues["select_errors"] += 1
                    entry["select"][str(k)] = {"error": str(exc)}
                    continue
                chosen_scores = [by_key.get(paper_key(p), 0) for p in chosen]
                entry["select"][str(k)] = dict(select_metrics(chosen_scores, labelled, k),
                                               shown=len(shown), picked=[p["id"] for p in chosen])
        ideas.append(entry)
        progress(f"  {idea['search_query']}: pool={len(ranked)} relevant={entry['n_relevant']}")
    return ideas, issues


def aggregate(ideas):
    rank = {name: {"mean": mean_or_none(i["metrics"][name] for i in ideas),
                   "n": sum(i["metrics"][name] is not None for i in ideas)} for name in RANK_METRICS}
    out = {"rank": rank}
    if ideas and "select" in ideas[0]:
        select = {}
        for name in SELECT_METRICS:
            per_idea = [select_per_idea(i, name) for i in ideas]
            select[name] = {"mean": mean_or_none(per_idea), "n": sum(v is not None for v in per_idea)}
            for k in TOP_KS:
                values = [i["select"][str(k)].get(name) for i in ideas if "error" not in i["select"][str(k)]]
                select[f"{name}@k={k}"] = {"mean": mean_or_none(values), "n": sum(v is not None for v in values)}
        out["select"] = select
    return out


def select_per_idea(idea, name):
    """Average over the five top_k queries of one idea (None-safe), so the idea stays the unit."""
    return mean_or_none(v.get(name) for v in idea["select"].values() if "error" not in v)


def judge_validity(issues, allow_unlabeled, n_ideas):
    reasons, warnings = [], []
    if not n_ideas:
        reasons.append("没有任何想法被评测")
    if issues["unlabeled"] and not allow_unlabeled:
        reasons.append(f"有 {issues['unlabeled']} 个返回的论文没有相关性标签（先运行 rank_labels label）")
    elif issues["unlabeled"]:
        warnings.append(f"{issues['unlabeled']} 个论文没有标签，按 0 分计")
    if issues["rerank_failed"]:
        reasons.append(f"精排未成功（已回退到第一阶段顺序）的想法：{len(issues['rerank_failed'])} 个")
    if issues["select_errors"]:
        reasons.append(f"{issues['select_errors']} 次 select 失败")
    return {"ok": not reasons, "reasons": reasons, "warnings": warnings}


# --- reports ---------------------------------------------------------------------------------

def render_markdown(result):
    meta, agg = result["meta"], result["aggregate"]
    lines = [f"# 排序评测 `{meta['label']}`", ""]
    if not result["validity"]["ok"]:
        lines += ["> **结果无效，不要用于对比：** " + "；".join(result["validity"]["reasons"]), ""]
    if result["validity"]["warnings"]:
        lines += ["> 注意：" + "；".join(result["validity"]["warnings"]), ""]
    lines += [f"- 候选池：每源 {meta['pool_limit']} 篇（哈希 `{meta['pools_hash']}`）；标签哈希 `{meta['labels_hash']}`",
              f"- 第一阶段排序：{meta['ranking_method']}；精排：{meta['rerank']}；评测想法数：{len(result['ideas'])}",
              f"- 代码：{meta['environment'].get('git_sha')}"
              f"{'（有未提交改动）' if meta['environment'].get('git_dirty') else ''}", "",
              "## 候选池排序质量", "| 指标 | 均值 | 有效想法数 |", "|---|---|---|"]
    lines += [f"| {name.replace('@M', '@' + str(meta['recall_m']))} | {_fmt(v['mean'])} | {v['n']} |"
              for name, v in agg["rank"].items()]
    if "select" in agg:
        lines += ["", "## 任务 Agent 选出的 Top-k（任务书的 Top-k 相关性）", "| 指标 | 均值 | 有效想法数 |", "|---|---|---|"]
        lines += [f"| {name} | {_fmt(v['mean'])} | {v['n']} |" for name, v in agg["select"].items()]
    return "\n".join(lines) + "\n"


def _fmt(value):
    return "-" if value is None else f"{value:.4f}"


MUST_MATCH = ("pools_hash", "labels_hash", "pool_limit", "mode", "recall_m")


def comparable(a, b):
    for name, result in (("A", a), ("B", b)):
        if not result["validity"]["ok"]:
            raise ValueError(f"{name} 的结果无效：" + "；".join(result["validity"]["reasons"]))
    for key in MUST_MATCH:
        if a["meta"].get(key) != b["meta"].get(key):
            raise ValueError(f"{key} 不一致：{a['meta'].get(key)} vs {b['meta'].get(key)}")
    if [i["qkey"] for i in a["ideas"]] != [i["qkey"] for i in b["ideas"]]:
        raise ValueError("两次评测的想法集合不同")


def compare(a, b, resamples=10000):
    comparable(a, b)
    la, lb = a["meta"]["label"], b["meta"]["label"]
    lines = [f"# 对比：`{la}`（A）vs `{lb}`（B）", "",
             f"- 候选池每源 {a['meta']['pool_limit']} 篇；模式 {a['meta']['mode']}；想法数 {len(a['ideas'])}",
             f"- A：排序 {a['meta']['ranking_method']} / 精排 {a['meta']['rerank']}；"
             f"B：排序 {b['meta']['ranking_method']} / 精排 {b['meta']['rerank']}",
             "- 差值 = B - A，括号为 95% bootstrap 区间（按想法配对）；区间不含 0 才说明差异不是噪声", "",
             "| 指标 | A | B | 差值 | 95% 区间 | B胜/平/B负 |", "|---|---|---|---|---|---|"]
    sections = [("rank", RANK_METRICS, lambda i, n: i["metrics"][n])]
    if a["meta"]["mode"] == "rank+select":
        sections.append(("select", SELECT_METRICS, select_per_idea))
    for _, names, getter in sections:
        for name in names:
            shown = name.replace("@M", "@" + str(a["meta"]["recall_m"]))
            pairs = [(getter(x, name), getter(y, name)) for x, y in zip(a["ideas"], b["ideas"])]
            pairs = [(p, q) for p, q in pairs if p is not None and q is not None]
            if not pairs:
                lines.append(f"| {shown} | - | - | - | - | - |")
                continue
            diffs = [q - p for p, q in pairs]
            low, high = paired_bootstrap(diffs, resamples)
            wins = sum(d > 1e-9 for d in diffs)
            losses = sum(d < -1e-9 for d in diffs)
            lines.append(f"| {shown} | {statistics.fmean(p for p, _ in pairs):.4f} | "
                         f"{statistics.fmean(q for _, q in pairs):.4f} | {statistics.fmean(diffs):+.4f} | "
                         f"[{low:+.4f}, {high:+.4f}] | {wins}/{len(diffs) - wins - losses}/{losses} |")
    return "\n".join(lines) + "\n"


# --- CLI -------------------------------------------------------------------------------------

async def run(args, agent_factory=TaskAgent):
    pools = load_pools(args.pools)
    scores = merged_scores(args.labels, args.human)
    method = resolve_method(args.ranking_method)
    if method in DENSE_METHODS and not os.getenv("BGE_M3_SERVICE_URL"):
        raise SystemExit(f"{method} 需要 BGE 服务（BGE_M3_SERVICE_URL）；未设置时会悄悄退回 bm25，结果会标错方法")
    configure_rerank(args.rerank)
    from benchmarks.run_benchmark import collect_environment
    meta = {"label": args.label, "mode": "rank+select" if args.select else "rank",
            "pool_limit": args.pool_limit, "recall_m": recall_cut(), "pools_hash": pools["hash"], "labels_hash": labels_hash(scores),
            "ranking_method": method, "rerank": args.rerank, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "judges": sorted({r.get("judge", "") for r in rank_labels.load_scores(args.labels).values()}),
            "human_labels": len(rank_labels.load_scores(args.human)), "environment": collect_environment()}
    agent = agent_factory() if args.select else None
    print(f"评测 {len(pools['ideas'])} 个想法，每源 {args.pool_limit} 篇，排序 {method}，精排 {args.rerank}"
          f"{'，含 select' if args.select else ''}", flush=True)
    ideas, issues = await evaluate(pools, scores, limit=args.pool_limit, method=method, select=args.select,
                                   recall_m=meta["recall_m"], agent=agent, rerank=args.rerank)
    result = {"meta": meta, "ideas": ideas, "aggregate": aggregate(ideas),
              "validity": judge_validity(issues, args.allow_unlabeled, len(ideas))}
    out = Path(args.out) if args.out else Path("outputs") / f"rankeval-{args.label}-{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "results.md").write_text(render_markdown(result), encoding="utf-8")
    print(render_markdown(result))
    print("结果目录：" + str(out))
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="action", required=True)
    run_p = sub.add_parser("run")
    run_p.add_argument("--label", required=True)
    run_p.add_argument("--pool-limit", type=int, default=8, help="每个来源取多少篇；8 是生产设置，25 能拉开排序差距")
    run_p.add_argument("--ranking-method", help="默认读 PAPER_RANKING_METHOD")
    run_p.add_argument("--rerank", choices=RERANK_CHOICES, default="none")
    run_p.add_argument("--select", action="store_true", help="同时评测任务 Agent 的选择（需要模型服务）")
    run_p.add_argument("--allow-unlabeled", action="store_true")
    run_p.add_argument("--pools", default=str(POOLS_PATH))
    run_p.add_argument("--labels", default=str(rank_labels.LABELS_PATH))
    run_p.add_argument("--human", default=str(rank_labels.HUMAN_PATH))
    run_p.add_argument("--out")
    cmp_p = sub.add_parser("compare")
    cmp_p.add_argument("a")
    cmp_p.add_argument("b")
    return parser.parse_args(argv)


def main(argv=None, agent_factory=TaskAgent):
    args = parse_args(argv)
    if args.action == "compare":
        load = lambda path: json.loads(Path(path).read_text(encoding="utf-8"))
        try:
            print(compare(load(args.a), load(args.b)))
        except ValueError as exc:
            raise SystemExit("无法对比：" + str(exc))
        return 0
    load_dotenv()
    result = asyncio.run(run(args, agent_factory=agent_factory))
    return 0 if result["validity"]["ok"] else 2


if __name__ == "__main__":
    sys.stdout.reconfigure(errors="replace")
    sys.exit(main())
