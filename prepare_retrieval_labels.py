"""合并同主题论文，隐藏方法和排名，生成手工标注文件。"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    folder = args.run_dir
    if not folder.is_absolute():
        folder = ROOT / folder
    data = json.loads((folder / "results.json").read_text(encoding="utf-8"))
    groups = {}
    for row in data["rows"]:
        if row["status"] != "ok":
            raise ValueError("存在失败查询，请先排查：" + row["query_id"] + "/" + row["method"])
        group = groups.setdefault(row["query_id"], {"idea": row["idea"], "papers": {}, "methods": []})
        group["methods"].append(row["method"])
        for paper in row["papers"]:
            group["papers"].setdefault(paper["stable_id"], paper)

    for qid, group in sorted(groups.items()):
        if sorted(group["methods"]) != ["baseline", "concept"]:
            raise ValueError(qid + " 缺少或重复实验方法")
        target = folder / f"{qid}_labels.json"
        if target.exists():
            print("保留已有标注：" + str(target))
            continue
        papers = []
        for sid, paper in sorted(group["papers"].items()):
            papers.append({
                "stable_id": sid,
                "title": paper["title"],
                "abstract": paper.get("abstract", ""),
                "url": paper["url"],
                "label": None,
                "reason": "",
            })
        payload = {"query_id": qid, "idea": group["idea"],
                   "rubric": {"0": "无关", "1": "部分相关", "2": "高度相关"},
                   "papers": papers}
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"{qid}：{len(papers)} 篇唯一论文，已保存 {target}")


if __name__ == "__main__":
    main()
