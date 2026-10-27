# 优化实验设计

任务书要求：从“增加免费数据来源 / 改进相关性排序 / 实现并发处理”中选做优化，并比较**优化前后的 Top-k 相关性、任务成功率或运行时间**。本文规定怎么测，保证任何优化都能和同一个基线公平对比。

## 对比规则（所有实验通用）

1. **只差一个变量。** 同一台机器、同一份 `.env`、同样的 llama-server 参数（`LLAMA_PARALLEL` 等）；两个版本之间只能是代码或开关不同。
2. **固定输入。** 查询集、冻结语料、候选池、相关性标签都带哈希，`compare` 发现不一致就拒绝对比。
3. **固定解码。** `LLM_TEMPERATURE=0`、`LLM_ENABLE_THINKING` 显式设置；预检不通过不开始。
4. **估计噪声。** 联网实验（端到端）必须带 control（与基线配置完全相同的副本），差距小于 control 与基线的差距就不算结果。
5. **无效就不算。** 成功率为 0、没有 LLM 调用、精排悄悄回退、有论文没标签、select 失败，结果都标“无效”，脚本退出码 2，`compare` 拒绝使用。
6. **报区间不只报均值。** 排序对比以“想法”为单位做配对 bootstrap（20 个想法），区间含 0 就说“没有可检出的差异”。

## 四组实验

| 编号 | 回答什么 | 工具 | 指标 | 需要联网 | 需要模型 |
|---|---|---|---|---|---|
| E0 模型层基线 | 这块 GPU 上 Q4（或 Q5）的推理耗时 | `run_all`（stage 模式） | 查询耗时、各阶段占比、tokens/s、成功率 | 否 | 是 |
| E1 并发（PR #2） | 分阶段并发是否真的缩短端到端运行时间 | `ab_e2e` | 端到端耗时配对差值、成功率 | 是 | 是 |
| E2 排序（PR #3） | 精排是否让 Top-k 更相关 | `rank_eval` | nDCG@k、P@k、MRR、recall@M；任务 Agent 选出的 Top-k 的平均相关度/精确率/nDCG | 否 | 仅 `--select` 和打标签 |
| E3 数据源 | 新增来源是否让候选更相关/成功率更高 | 尚未实现 | — | 是 | 是 |

E0 是模型层面的参考，**不能当作 E1/E2 的优化前后对比**：它用冻结语料，不下载、不转换，也不检索。

### E0 模型层基线

```bash
python -m benchmarks.run_all --label baseline-q4          # 冒烟 3 条后跑完整 100 条
```
冒烟实测（RTX PRO 5000 48GB，Q4）：3 条全部成功，解码约 58 tok/s、预填充约 2000 tok/s；按平均每条约 97 s 估算，100 条约 2.7 小时（估算值，未跑完整）。

### E1 并发（PR #2）

基线是**当前 main 的整篇并发 2**（`--concurrent`），不是串行：PR #2 要替换的就是它。

```bash
python -m benchmarks.make_variants --root ../variants          # 生成 base / pr2 / pr3 工作树和 variants-pr2.json
LLAMA_PARALLEL=2 ...                                             # 服务端至少 2 个槽位，所有版本保持一致
python -m benchmarks.ab_e2e --variants ../variants/variants-pr2.json --limit 8 --repeats 2 --label pr2
```
- `variants-pr2.json` 含 4 个版本：`base-concurrent`（基线）、`base-control`（噪声）、`base-serial`（参考）、`pr2-staged`（`PIPELINE_MODE=staged`）。
- 同一条查询的各版本背靠背跑、顺序随机；每次运行用全新 PDF 缓存，下载和转换都计入耗时。
- 看 `summary.md` 的“配对差值”和 95% 区间；差值要明显大于“噪声基线”。
- 规模：4 版本 x 8 条 x 2 轮 = 64 次运行，按每次约 2 分钟估算约 2 小时（估算，未实测）。
- PR #2 自带的真实实验中位耗时 315.8 s → 341.1 s，并没有变快；它的“有效”结论来自模拟实验，不能用作真实运行时间的证据。本实验就是为了给出真实数据。

### E2 排序（PR #3）

数据（一次性，已完成的部分已入库）：
- `benchmarks/rank_pools.json.gz`：20 个想法 x 两个来源各 25 篇的**原始检索结果**（联网冻结，已完成）。`--pool-limit 8` 取出生产设置的候选池（每想法约 15 篇），`--pool-limit 25` 取出更大的池（约 45 篇，排序差异更容易显现）。
- `benchmarks/rank_labels.jsonl`：每篇论文对其想法的 0–3 分相关度（**待生成**，需要模型）。

```bash
# 1) 打标签（模型服务运行中；共 864 对，预计十几分钟到半小时，未实测）
python -m benchmarks.rank_labels label --workers 1
python -m benchmarks.rank_labels coverage
# 2) 人工抽查：导出 60 条，由同学只填 human_score 列（先别看 judge 列），再算一致性并写入
python -m benchmarks.rank_labels sample --n 60 --out rank_sample.csv
python -m benchmarks.rank_labels agreement --csv rank_sample_filled.csv --save
# 3) 在各版本工作树里分别评测（排序质量不需要模型）
cd ../variants/base && python -m benchmarks.rank_eval run --label base-bm25 --pool-limit 25 --ranking-method bm25
cd ../variants/pr3  && python -m benchmarks.rank_eval run --label pr3-ce   --pool-limit 25 --ranking-method bm25 --rerank http
# 4) 任务书说的 Top-k 相关性：加 --select，让任务 Agent 在它真正会看到的候选里选 Top-k 再打分
python -m benchmarks.rank_eval compare outputs/rankeval-base-*/results.json outputs/rankeval-pr3-*/results.json
```
- 精排用 `--rerank http` 时要先启动 PR #3 的重排服务（`deploy/serve_reranker.py`，默认端口 8003）；`--rerank local` 在进程内加载模型，需要装 `requirements-reranker.txt`。`rank_eval` 会校验精排真的成功，回退到第一阶段顺序算无效，防止测成“没精排”。
- 精排开启后任务 Agent 只看到前 `RERANKER_AGENT_TOP_M`（默认 8）篇，`rank_eval` 按同样的规则截取；`recall@M` 衡量这一截取丢掉了多少相关论文。
- 稠密/混合排序（`dense_linear`、`hybrid_rrf`）需要 BGE 服务，未设置 `BGE_M3_SERVICE_URL` 会直接报错，避免悄悄退回 bm25 却标成别的方法。
- 评测的第一阶段排序用的是 `retrieve()` 本身，所以任何版本自带的排序方法、精排原样运行。

**排序质量 vs. 最终 Top-k：** 候选池只有约 15 篇时，Top-5 占了三分之一，排序方法之间的差距会被任务 Agent 的选择抹平。因此同时报告 `--pool-limit 8`（生产设置）和 `--pool-limit 25`（压力设置）。

### E3 数据源

尚未实现新来源。若选做它：实现新来源后用 `rank_pools` 重新冻结（新数据集哈希）、重新打标签，E0 基线也要重跑。

## 版本

| 版本 | 构成 | 状态 |
|---|---|---|
| base | 基准分支（main + 基准工具，行为不变） | 可用 |
| pr2 | base + PR #2 | 合并无冲突；全量测试仅有 main 上已有的 14 个失败 |
| pr3 | base + PR #3 | 合并无冲突；同上 |
| both | base + PR #2 + PR #3 | **冲突**，`make_variants` 会报告并移除，不会留下半合并的目录 |

冲突在 `research/pipeline.py`：PR #3 在结果字典里加了 5 个排名字段（`base_rank` 等），PR #2 重写了同一个函数（`process_one`）。需要其中一个 PR 先 rebase，改动只是把这 5 个字段加进 PR #2 的新写法。在此之前不做“两个优化叠加”的实验；任务书只要求选做一项，各自对基线比较即可。

## 已知局限

- **裁判和被评对象同属一个模型家族**，可能偏向自己的选择。对策是人工抽查 + 一致性指标（`agreement`），人工标签覆盖裁判标签。也可用 `JUDGE_BASE_URL`/`JUDGE_MODEL`/`JUDGE_API_KEY` 换一个更强的裁判。
- 只有 20 个想法，置信区间会比较宽；差异不显著时如实写“没有可检出的差异”。
- 评测用的检索词是数据集里的英文短语 `search_query`，生产中由主 Agent 的 LLM 生成，二者不完全一样。
- 联网实验受 arXiv 限流和网络波动影响，这是要放 control 和重复的原因。
- E1 的 A/B 只比较运行时间和成功率；总结质量没有评分。
