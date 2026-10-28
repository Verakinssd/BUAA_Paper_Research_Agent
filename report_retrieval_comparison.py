"""验证人工标签并生成 arXiv 查询对比报告，不联网、不改变标签。"""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def percent(n, denominator):
    return f"{n / denominator:.1%}" if denominator else "不适用"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    folder = args.run_dir if args.run_dir.is_absolute() else ROOT / args.run_dir
    source = folder / "results.json"
    data = json.loads(source.read_text(encoding="utf-8"))
    grouped, summary, digests = {}, [], {}
    digests["results.json"] = hashlib.sha256(source.read_bytes()).hexdigest()
    for row in data["rows"]:
        if row["status"] != "ok":
            raise ValueError("失败请求不能作为零结果参与比较")
        pair = grouped.setdefault(row["query_id"], {})
        if row["method"] in pair:
            raise ValueError("同主题同方法出现重复记录")
        pair[row["method"]] = row

    lines = ["# arXiv 查询优化对比报告", "",
             "## 实验范围与评价口径", "",
             f"- 每主题每种方法最多获取 {data['limit']} 篇；只比较 arXiv，未改变项目排序、去重及 OpenAlex。",
             "- baseline 为完整短语查询；concept 为三个主题人工配置的概念组合查询，不是通用自动查询生成器。",
             "- 人工标签：0=无关，1=部分相关，2=高度相关；主指标把标签 2 视为相关，同时列出标签≥1的结果。",
             "- 标注依据为标题和摘要，过程中有 AI 解释与复核建议，未经独立双人标注；不构成全文质量评价。",
             "- 请求耗时包含 HTTP 请求与解析，不含请求间等待；每组只有一次观测，不能据此判断长期速度优势。",
             "- 这是有数量上限的候选覆盖对比，没有完整相关论文全集，不能计算真实召回率，也不代表 Top-k 或端到端任务成功率。",
             "", "## 方法对比", "",
             "|主题|方法|返回数|高度相关数|高度相关比例|标签≥1数量|标签≥1比例|耗时/秒|请求数|",
             "|---|---|---:|---:|---|---:|---|---:|---:|"]
    details = []
    for qid, pair in sorted(grouped.items()):
        if set(pair) != {"baseline", "concept"}:
            raise ValueError(qid + " 缺少两种方法之一")
        path = folder / f"{qid}_labels.json"
        labels = json.loads(path.read_text(encoding="utf-8"))
        if labels["query_id"] != qid:
            raise ValueError("标注主题 ID 不匹配")
        digests[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        mapping = {}
        for p in labels["papers"]:
            if p["stable_id"] in mapping:
                raise ValueError("重复标注 ID")
            if type(p["label"]) is not int or p["label"] not in (0, 1, 2):
                raise ValueError("存在未标注或不合法标签：" + p["title"])
            if not p.get("reason", "").strip():
                raise ValueError("缺少理由：" + p["title"])
            mapping[p["stable_id"]] = p
        sets = {method: {p["stable_id"] for p in row["papers"]}
                for method, row in pair.items()}
        if set(mapping) != sets["baseline"] | sets["concept"]:
            raise ValueError(qid + " 标注集与候选并集不一致")
        counts = {}
        for method in ("baseline", "concept"):
            row = pair[method]
            ids = sets[method]
            if len(ids) != len(row["papers"]) or len(ids) != row["returned_count"]:
                raise ValueError("返回数量与唯一论文数量不一致")
            c = Counter(mapping[sid]["label"] for sid in ids)
            n, high, broad = len(ids), c[2], c[1] + c[2]
            counts[method] = {"returned": n, "highly_relevant": high,
                              "at_least_partly_relevant": broad,
                              "highly_relevant_ratio": high / n if n else None,
                              "request_seconds": row["request_seconds"],
                              "request_count": row["request_count"]}
            lines.append(f"|{qid}|{method}|{n}|{high}|{percent(high,n)}|{broad}|{percent(broad,n)}|{row['request_seconds']}|{row['request_count']}|")
        added = sets["concept"] - sets["baseline"]
        lost = sets["baseline"] - sets["concept"]
        added_high = sum(mapping[sid]["label"] == 2 for sid in added)
        lost_high = sum(mapping[sid]["label"] == 2 for sid in lost)
        delta = counts["concept"]["highly_relevant"] - counts["baseline"]["highly_relevant"]
        summary.append({"query_id": qid, "methods": counts,
                        "added_count": len(added), "added_highly_relevant": added_high,
                        "not_retained_count": len(lost), "not_retained_highly_relevant": lost_high,
                        "highly_relevant_delta": delta})
        details.extend(["", f"## {qid}：{pair['baseline']['idea']}", "",
                        f"高度相关论文数：{counts['baseline']['highly_relevant']} → {counts['concept']['highly_relevant']}，净变化 {delta:+d}。",
                        f"新增唯一论文 {len(added)} 篇，其中高度相关 {added_high} 篇；原查询结果有 {len(lost)} 篇未出现在改进查询前 {data['limit']} 篇中，其中高度相关 {lost_high} 篇。",
                        "注意：未出现在前若干篇不表示查询完全无法匹配该论文。", "",
                        "|方法|实际查询|", "|---|---|"])
        for method in ("baseline", "concept"):
            details.append(f"|{method}|{cell(pair[method]['params']['search_query'])}|")
        details.extend(["", "|论文|标签|结果归属|标注理由|", "|---|---:|---|---|"])
        for sid, p in sorted(mapping.items()):
            membership = "两种共有" if sid in sets["baseline"] & sets["concept"] else ("concept新增" if sid in added else "仅baseline")
            details.append(f"|[{cell(p['title'])}]({p['url']})|{p['label']}|{membership}|{cell(p['reason'])}|")
    lines.extend(details)
    lines.extend(["", "## 解释与下一步", "",
                  "以上结果限于本次三个已参与查询设计的主题，尚不能外推到任意研究想法。",
                  "应结合新增相关论文、未保留的原查询相关论文及相关比例评价收益；不能仅凭返回数量增加判断优化成功。",
                  "若要接入主流程，建议增加独立测试主题，并保留完整短语基线及可切换策略。", ""])
    (folder / "comparison.md").write_text("\n".join(lines), encoding="utf-8")
    (folder / "comparison_metrics.json").write_text(json.dumps(
        {"relevant_threshold": 2, "input_sha256": digests, "topics": summary},
        ensure_ascii=False, indent=2), encoding="utf-8")
    for row in summary:
        b, c = row["methods"]["baseline"], row["methods"]["concept"]
        print(f"{row['query_id']}：高度相关 {b['highly_relevant']}/{b['returned']} → "
              f"{c['highly_relevant']}/{c['returned']}；新增高度相关 {row['added_highly_relevant']}，"
              f"未保留原查询高度相关 {row['not_retained_highly_relevant']}")
    print("报告：" + str(folder / "comparison.md"))


if __name__ == "__main__":
    main()
