"""Build the fixed 100-query benchmark set: 20 ideas x top_k 1..5.

Run: python -m benchmarks.build_dataset
"""
import hashlib
import json
from pathlib import Path

QUERIES_PATH = Path(__file__).with_name("queries.jsonl")
TOP_KS = (1, 2, 3, 4, 5)

# (topic, lang, idea, search_query). search_query is fixed so the main Agent's
# idea-to-English translation cannot add run-to-run variance. They are short topic
# phrases (what the main Agent is told to produce) because arXiv matches them as an
# exact phrase: long sentences return nothing.
IDEAS = [
    ("peft", "zh", "我想研究大语言模型的参数高效微调方法。",
     "parameter-efficient fine-tuning"),
    ("rag", "zh", "我想研究检索增强生成（RAG）如何减少大模型幻觉。",
     "retrieval-augmented generation"),
    ("quant", "zh", "我想了解大语言模型的低比特量化与推理加速技术。",
     "language model quantization"),
    ("kvcache", "zh", "我想研究长上下文推理中的 KV Cache 压缩方法。",
     "KV cache compression"),
    ("speculative", "zh", "我想研究大模型推测解码以加快生成速度。",
     "speculative decoding"),
    ("multimodal", "zh", "我想研究视觉语言多模态大模型的对齐与评测。",
     "vision-language models"),
    ("rlhf", "zh", "我想研究基于人类反馈的强化学习（RLHF）及其替代方法。",
     "reinforcement learning from human feedback"),
    ("gnn", "zh", "我想研究图神经网络在节点分类中的过平滑问题。",
     "over-smoothing"),
    ("timeseries", "zh", "我想研究基于 Transformer 的时间序列预测方法。",
     "time series forecasting"),
    ("medical", "zh", "我想研究医学影像分割中的小样本学习方法。",
     "medical image segmentation"),
    ("recsys", "zh", "我想研究大语言模型在序列推荐系统中的应用。",
     "sequential recommendation"),
    ("codegen", "zh", "我想研究代码大模型的自动生成与程序修复能力。",
     "code generation"),
    ("safety", "zh", "我想研究大模型越狱攻击与防御方法。",
     "jailbreak attacks"),
    ("agent", "zh", "我想研究基于大语言模型的智能体工具调用与规划。",
     "LLM agents"),
    ("diffusion", "zh", "我想研究扩散模型的高效采样与加速方法。",
     "diffusion model sampling"),
    ("moe", "en", "I want to study mixture-of-experts architectures for scaling language models.",
     "mixture of experts"),
    ("continual", "en", "I want to learn about continual learning and catastrophic forgetting in neural networks.",
     "catastrophic forgetting"),
    ("ssl", "en", "I am interested in self-supervised contrastive representation learning for images.",
     "contrastive learning"),
    ("federated", "en", "I want to explore federated learning with privacy guarantees and non-IID data.",
     "federated learning"),
    ("cot", "en", "I want to study chain-of-thought reasoning in large language models.",
     "chain-of-thought reasoning"),
]


def build_queries():
    rows = []
    for i, (topic, lang, idea, search_query) in enumerate(IDEAS):
        for j, top_k in enumerate(TOP_KS):
            rows.append({"qid": f"q{i * len(TOP_KS) + j + 1:03d}", "topic": topic, "lang": lang,
                         "idea": idea, "search_query": search_query, "top_k": top_k})
    return rows


def dumps(rows):
    return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)


def validate(rows):
    """Raise ValueError unless rows are the intended fixed 100-query set."""
    if len(rows) != 100:
        raise ValueError(f"数据集必须正好 100 条，当前 {len(rows)}")
    if len({r["qid"] for r in rows}) != len(rows):
        raise ValueError("qid 重复")
    for k in TOP_KS:
        if sum(r["top_k"] == k for r in rows) != 20:
            raise ValueError(f"top_k={k} 必须正好 20 条")
    langs = {lang: sum(r["lang"] == lang for r in rows) for lang in ("zh", "en")}
    if langs != {"zh": 75, "en": 25}:
        raise ValueError(f"中英配比必须为 75/25，当前 {langs}")
    for r in rows:
        if not all(isinstance(r.get(f), str) and r[f].strip()
                   for f in ("qid", "topic", "lang", "idea", "search_query")):
            raise ValueError(f"{r.get('qid')} 字段缺失")


def load_queries(path=QUERIES_PATH):
    """Return (rows, dataset_hash). The hash is over the exact file bytes."""
    raw = Path(path).read_bytes()
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    validate(rows)
    return rows, hashlib.sha256(raw).hexdigest()[:16]


if __name__ == "__main__":
    data = build_queries()
    validate(data)
    QUERIES_PATH.write_text(dumps(data), encoding="utf-8", newline="\n")
    print(f"已写入 {QUERIES_PATH}，100 条，哈希 {load_queries()[1]}")
