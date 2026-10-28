# arXiv 查询构造对比实验

本提交新增独立实验脚本、三个主题的人工概念配置、原始响应、人工标签及对比报告。尚未修改 research/sources.py 或接入项目主流程，也不改动 OpenAlex、排序或并发模块。

## 复现已有报告（离线）

在仓库根目录使用项目 Python 环境运行：

```bash
python report_retrieval_comparison.py --run-dir evaluation/retrieval_runs/20261023T053451908655Z
```

报告输出到同目录 comparison.md，机器可读指标为 comparison_metrics.json。命令会重写这两个派生文件，不改变原始结果或人工标签。

## 重新采集（联网）

```bash
python collect_retrieval_comparison.py
python collect_retrieval_comparison.py --run --limit 10
python prepare_retrieval_labels.py --run-dir outputs/retrieval/实际运行目录
```

第一条只预览，第二条向 arXiv 发出六次查询，第三条生成去除方法和排名信息的待标注文件。标注完成后使用报告脚本生成对应报告。依赖复用项目 requirements.txt；网络代理等设置读取本机 .env，不提交该文件。

## 本次结果

|主题|完整短语返回/高度相关|概念组合返回/高度相关|
|---|---|---|
|q01|0 / 0|10 / 6|
|q02|1 / 0|10 / 9|
|q03|3 / 1|10 / 8|

以标签2为高度相关；标签1为部分相关。q03原查询有1篇高度相关论文不在概念组合的前10篇中，因此不能声称新方法保留所有原查询相关论文。

## 限制与后续工作

- 三个主题参与了概念设计，尚未验证新主题泛化；不是通用自动概念提取。
- 评分依据为标题和摘要，使用过AI辅助解释与复核，未经独立双人标注；部分理由仍可进一步人工复核。
- 每种查询只有一次观测，不构成速度显著性结论。请求耗时不包含3.1秒间隔等待。
- 最多取10篇，没有相关论文全集，不能称为真实召回率提升，也不能作为端到端任务成功率或排序评测。
- 实验文件中的论文摘要和XML为来源元数据，权利归原作者或相应权利人；本提交不包含论文PDF或模型权重。
- 后续计划增加独立主题验证，并在验证后再接入可切换的主流程策略。

## 文件说明

- collect_retrieval_comparison.py：完整短语与概念组合查询采集。
- prepare_retrieval_labels.py：同主题按arXiv ID合并候选，生成标注文件。
- report_retrieval_comparison.py：校验标签、计算指标并输出Markdown报告。
- evaluation/retrieval_queries.json：三组原查询与人工概念组。
- evaluation/retrieval_runs/：保留的原始实验与标签；不包含API密钥、账号数据库和环境文件。
