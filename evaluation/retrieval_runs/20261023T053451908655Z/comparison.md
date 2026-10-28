# arXiv 查询优化对比报告

## 实验范围与评价口径

- 每主题每种方法最多获取 10 篇；只比较 arXiv，未改变项目排序、去重及 OpenAlex。
- baseline 为完整短语查询；concept 为三个主题人工配置的概念组合查询，不是通用自动查询生成器。
- 人工标签：0=无关，1=部分相关，2=高度相关；主指标把标签 2 视为相关，同时列出标签≥1的结果。
- 标注依据为标题和摘要，过程中有 AI 解释与复核建议，未经独立双人标注；不构成全文质量评价。
- 请求耗时包含 HTTP 请求与解析，不含请求间等待；每组只有一次观测，不能据此判断长期速度优势。
- 这是有数量上限的候选覆盖对比，没有完整相关论文全集，不能计算真实召回率，也不代表 Top-k 或端到端任务成功率。

## 方法对比

|主题|方法|返回数|高度相关数|高度相关比例|标签≥1数量|标签≥1比例|耗时/秒|请求数|
|---|---|---:|---:|---|---:|---|---:|---:|
|q01|baseline|0|0|不适用|0|不适用|4.033|1|
|q01|concept|10|6|60.0%|10|100.0%|1.251|1|
|q02|baseline|1|0|0.0%|1|100.0%|0.752|1|
|q02|concept|10|9|90.0%|10|100.0%|1.216|1|
|q03|baseline|3|1|33.3%|3|100.0%|0.649|1|
|q03|concept|10|8|80.0%|10|100.0%|2.577|1|

## q01：研究低资源条件下大语言模型的参数高效微调方法

高度相关论文数：0 → 6，净变化 +6。
新增唯一论文 10 篇，其中高度相关 6 篇；原查询结果有 0 篇未出现在改进查询前 10 篇中，其中高度相关 0 篇。
注意：未出现在前若干篇不表示查询完全无法匹配该论文。

|方法|实际查询|
|---|---|
|baseline|all:"parameter efficient fine tuning large language models"|
|concept|(all:"parameter efficient" OR all:"low rank adaptation" OR all:"LoRA") AND (all:"large language model" OR all:"large language models" OR all:"LLM")|

|论文|标签|结果归属|标注理由|
|---|---:|---|---|
|[QA-LoRA: Quantization-Aware Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2309.14717v2)|2|concept新增|将量化与低秩适配结合，明确针对大语言模型微调的时间和内存开销进行优化|
|[ALoRA: Allocating Low-Rank Adaptation for Fine-tuning Large Language Models](https://arxiv.org/abs/2403.16187v2)|2|concept新增|通过动态调整低秩参数并分配参数预算，直接研究大语言模型的参数高效微调|
|[OLoRA: Orthonormal Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2406.01775v2)|2|concept新增|通过改进LoRA初始化加快微调收敛，并保留参数量和显存方面的效率优势|
|[Exploring Advanced Large Language Models with LLMsuite](https://arxiv.org/abs/2407.12036v2)|1|concept新增|探讨了大型语言模型的技术进展、固有局限性的缓解方案，并总结了包括参数高效微调在内的多种模型训练与优化策略|
|[BA-LoRA: Bias-Alleviating Low-Rank Adaptation to Mitigate Catastrophic Inheritance in Large Language Models](https://arxiv.org/abs/2408.04556v8)|1|concept新增|使用了LoRA，但主要研究具体应用，没有讨论资源效率或微调方法改进|
|[Flexora: Flexible Low Rank Adaptation for Large Language Models](https://arxiv.org/abs/2408.10774v5)|1|concept新增|使用了LoRA，但主要研究具体应用，没有讨论资源效率或微调方法改进|
|[LLM-NEO: Parameter Efficient Knowledge Distillation for Large Language Models](https://arxiv.org/abs/2411.06839v2)|2|concept新增|将模型压缩与参数高效微调结合，旨在解决大语言模型的轻量化和低成本转移问题。|
|[Gated Integration of Low-Rank Adaptation for Continual Learning of Large Language Models](https://arxiv.org/abs/2505.15424v2)|1|concept新增|使用LoRA但是目标不同，增加了学习工作量，与提升资源效率矛盾|
|[C-LoRA: Contextual Low-Rank Adaptation for Uncertainty Estimation in Large Language Models](https://arxiv.org/abs/2505.17773v3)|2|concept新增|通过引入“针对每个样本的上下文 LoRA 模块”来动态调整不确定性，解决数据稀缺的少样本时过度自信预测问题|
|[TLoRA: Task-aware Low Rank Adaptation of Large Language Models](https://arxiv.org/abs/2604.18124v1)|2|concept新增|明确指出现有的 LoRA 变体往往会“增加训练复杂度或降低实际效率”，而 TLoRA 则是为了解决这一痛点。|

## q02：研究检索增强生成中降低幻觉的方法

高度相关论文数：0 → 9，净变化 +9。
新增唯一论文 10 篇，其中高度相关 9 篇；原查询结果有 1 篇未出现在改进查询前 10 篇中，其中高度相关 0 篇。
注意：未出现在前若干篇不表示查询完全无法匹配该论文。

|方法|实际查询|
|---|---|
|baseline|all:"retrieval augmented generation hallucination"|
|concept|(all:"retrieval augmented generation" OR all:"RAG") AND (all:"hallucination" OR all:"hallucinations" OR all:"factuality")|

|论文|标签|结果归属|标注理由|
|---|---:|---|---|
|[MultiHop-RAG: Benchmarking Retrieval-Augmented Generation for Multi-Hop Queries](https://arxiv.org/abs/2401.15391v1)|1|concept新增|主要涉及 RAG 系统在多跳查询（复杂推理和检索）任务中的性能基准测试与数据集构建。虽然文章在背景中提到了 RAG 可以缓解幻觉，并且其对模型事实性和推理能力的评估与减少幻觉有一定关联，但其核心贡献并非直接提出降低 RAG 幻觉的方法，也不是专门针对幻觉的检测或评估指标，因此与“降低 RAG 幻觉”这一核心目标的联系较为间接。|
|[Reducing hallucination in structured outputs via Retrieval-Augmented Generation](https://arxiv.org/abs/2404.08189v1)|2|concept新增|文章明确针对大模型在生成结构化输出时的幻觉问题，提出并实现了一个特定的 RAG 系统架构来显著降低幻觉的发生。文章的核心贡献直接围绕“利用/优化 RAG 系统来减少幻觉”展开|
|[ReDeEP: Detecting Hallucination in Retrieval-Augmented Generation via Mechanistic Interpretability](https://arxiv.org/abs/2410.11414v2)|2|concept新增|文章不仅直接提出了能够支撑降低幻觉的检测方法（ReDeEP），而且还在最后明确提出了一种减轻（降低）幻觉的方法（AARF），符合“直接研究 RAG 幻觉的降低方法|
|[Hyper-RAG: Combating LLM Hallucinations using Hypergraph-Driven Retrieval-Augmented Generation](https://arxiv.org/abs/2504.08758v1)|2|concept新增|文章直接提出了一种名为 Hyper-RAG 的创新方法，其核心研究目标和实验验证都明确指向“对抗”和“减轻”大模型生成中的幻觉问题。|
|[Finetune-RAG: Fine-Tuning Language Models to Resist Hallucination in Retrieval-Augmented Generation](https://arxiv.org/abs/2505.10792v3)|2|concept新增|文章直接提出了一种名为 Finetune-RAG 的微调方法来抵抗（降低）RAG 系统中因检索不完美而导致的幻觉，同时还提出了一个专门的评估管道（Bench-RAG）来进行压力测试。|
|[Removal of Hallucination on Hallucination: Debate-Augmented RAG](https://arxiv.org/abs/2505.18581v1)|2|concept新增|文章直接指出了 RAG 场景下由于检索错误导致的“幻觉叠加幻觉”问题，并明确提出了一种名为 DRAG 的辩论增强框架来专门解决这一问题。文章的实验结果也直接证明了该方法能够减少 RAG 引发的幻觉|
|[Probabilistic distances-based hallucination detection in LLMs with RAG](https://arxiv.org/abs/2506.09886v2)|2|concept新增|文章明确提出了一种专门针对 RAG 系统的幻觉检测方法（基于概率距离）。根据您的判断标准，能够直接支撑降低幻觉的“检测、评估方法”被归类为最高相关性。|
|[Theoretical Foundations and Mitigation of Hallucination in Large Language Models](https://arxiv.org/abs/2507.22915v1)|1|仅baseline|文章在缓解策略中提及了检索增强生成（RAG），并且深入探讨了模型事实性，但其核心是利用 RAG 等手段来缓解 LLM 的幻觉，而不是专门研究如何降低 RAG 系统本身的幻觉。|
|[MultiRAG: A Knowledge-guided Framework for Mitigating Hallucination in Multi-source Retrieval Augmented Generation](https://arxiv.org/abs/2508.03553v1)|2|concept新增|文章明确指出了多源 RAG 场景下自身会引入和加剧幻觉的问题（由于信息冲突等原因），并直接提出了一种名为 MultiRAG 的新框架来专门“缓解多源检索增强生成中的幻觉”。|
|[FAIR-RAG: Faithful Adaptive Iterative Refinement for Retrieval-Augmented Generation](https://arxiv.org/abs/2510.22344v1)|2|concept新增|该文章明确指出现有 RAG 方法在复杂查询中“会传播噪声或未能收集全面的上下文”（这是导致 RAG 幻觉的直接原因），并提出 FAIR-RAG 框架来确保最终的“严格忠实的生成”（strictly faithful generation）。在生成任务中，“忠实”（faithful）是“幻觉”（hallucination）的直接反义词，通过填补证据空白和分析门控机制来确保生成的忠实性，本质上就是一种直接研究降低 RAG 幻觉的方法。这符合最高相关性标准。|
|[Stable-RAG: Mitigating Retrieval-Permutation-Induced Hallucinations in Retrieval-Augmented Generation](https://arxiv.org/abs/2601.02993v4)|2|concept新增|文章明确指出了 RAG 系统中由于“检索文档排列顺序”会导致模型产生幻觉，并直接提出了一种名为 Stable-RAG 的新方法来专门缓解这种由排列引起的幻觉。文章的核心贡献和实验验证都直接围绕“降低 RAG 幻觉”这一目标展开，完全符合“直接研究 RAG 幻觉的降低方法”这一最高判断标准。|

## q03：研究大语言模型多智能体协作完成复杂任务的方法

高度相关论文数：1 → 8，净变化 +7。
新增唯一论文 10 篇，其中高度相关 8 篇；原查询结果有 3 篇未出现在改进查询前 10 篇中，其中高度相关 1 篇。
注意：未出现在前若干篇不表示查询完全无法匹配该论文。

|方法|实际查询|
|---|---|
|baseline|all:"large language model multi agent collaboration"|
|concept|(all:"large language model" OR all:"large language models" OR all:"LLM") AND (all:"multi agent" OR all:"multi-agent") AND (all:"collaboration" OR all:"cooperation" OR all:"coordination")|

|论文|标签|结果归属|标注理由|
|---|---:|---|---|
|[LLM-Coordination: Evaluating and Analyzing Multi-agent Coordination Abilities in Large Language Models](https://arxiv.org/abs/2310.03903v3)|2|concept新增|文章直接研究了 LLM 多智能体在合作环境下的协调、联合规划以及心智理论（协作推理）能力，并提出了专门评估这些协作机制的基准测试|
|[MechAgents: Large language model multi-agent collaborations can solve mechanics problems, generate new data, and integrate knowledge](https://arxiv.org/abs/2311.08166v1)|2|仅baseline|明确指出该研究构建了基于 LLM 的多智能体框架，并且详细探讨了智能体在解决复杂任务时的协作机制，包括“双智能体团队的编写、执行和自我纠正代码”，以及“更大智能体组在规划、公式化、编码、执行和批评方面的强化分工”，这种分工和相互纠正机制的设计符合“直接研究 LLM 多智能体的分工、协作推理或任务完成机制”的最高评分标准。|
|[ProtAgents: Protein discovery via large language model multi-agent collaborations combining physics and machine learning](https://arxiv.org/abs/2402.04268v1)|1|仅baseline|该文章虽然构建了一个基于 LLM 的多智能体协作平台（ProtAgents）来处理复杂任务，但其核心贡献和实验场景都高度集中在“从头设计蛋白质”这一特定的垂直应用领域。智能体的分工（如蛋白质结构分析、基于物理的模拟）是专门针对该应用定制的，对通用或可迁移的底层协作机制的探讨相对有限。|
|[A Large Language Model-based multi-agent manufacturing system for intelligent shopfloor](https://arxiv.org/abs/2405.16887v2)|2|concept新增|摘要中明确指出该研究旨在解决传统多智能体协商（negotiation）机制的局限性，并专门设计了包含招标（Bid Inviter）、投标（Bidder）、思考和决策等模块的 LLM 多智能体协作框架。这种基于招投标的交互设计直接研究了智能体之间的通信、协调与任务分配机制，符合最高评分标准。|
|[Towards Collaborative Intelligence: Propagating Intentions and Reasoning for Multi-Agent Coordination with Large Language Models](https://arxiv.org/abs/2407.12532v1)|2|concept新增|摘要明确提出构建基于 LLM 的协作智能体框架，直接深入研究了智能体之间的意图传播、通信消息转换以及规划与执行模块的协作机制。其核心贡献就是解决多智能体系统的通信与协调问题，完全符合“直接研究 LLM 多智能体的分工、通信、协调、协作推理机制”的最高标准。|
|[MedAide: Information Fusion and Anatomy of Medical Intents via LLM-based Agent Collaboration](https://arxiv.org/abs/2410.12532v3)|2|concept新增|摘要中明确提出了一个多智能体协作框架，并重点设计了一种“轮换智能体协作机制”（包含动态角色轮换和决策层面的信息融合）。虽然这是一篇医疗领域的应用论文，但其提出的动态角色轮换机制属于可迁移的协作机制设计，且文章对此进行了大量实验分析，符合“应用论文提出可迁移的协作机制，并通过实验分析该机制，也可以评为 2”的判定标准。|
|[Towards Effective GenAI Multi-Agent Collaboration: Design and Evaluation for Enterprise Applications](https://arxiv.org/abs/2412.05449v1)|2|concept新增|摘要明确提出研究并设计了多智能体协作框架中的“协作协议”以及“协调和路由能力”，并详细探讨了“并行通信”和“智能体间的高效消息转发”等底层通信与任务协作机制。这完全符合“直接研究 LLM 多智能体的分工、通信、协调、协作推理或任务完成机制”的最高评分标准。|
|[TradingAgents: Multi-Agents LLM Financial Trading Framework](https://arxiv.org/abs/2412.20138v7)|1|concept新增|摘要表明该研究提出了一个受现实交易公司启发的多智能体协作框架。虽然它使用了 LLM 多智能体并设计了辩论与协作流程，但智能体被分配了“牛市/熊市研究员”、“基础分析师”等高度领域特定的角色。其主要贡献和实验评估指标（如累积回报、夏普比率）都集中在金融交易这一特定应用上，其协作机制与特定场景深度绑定，并未明确提出或验证通用的、可迁移的协作机制。|
|[Don't Just Demo, Teach Me the Principles: A Principle-Based Multi-Agent Prompting Strategy for Text Classification](https://arxiv.org/abs/2502.07165v1)|2|concept新增|摘要中明确提出了一种具有清晰分工的多智能体协作工作流：多个智能体负责独立生成原则，一个总结智能体（finalizer agent）负责合并协调，最后由分类器智能体（classifier agent）执行任务。这种“生成-汇总-执行”的设计属于可迁移的分工与协作机制，且摘要明确提到通过消融实验分析了“多智能体协作 LLM 框架”的作用，符合“提出可迁移的协作机制，并通过实验分析该机制”的最高评分标准。|
|[ARCANE: A Multi-Agent Framework for Interpretable and Configurable Alignment](https://arxiv.org/abs/2512.06196v1)|1|concept新增|摘要表明该研究提出了一个多智能体框架（ARCANE）来解决模型对齐和奖励模型构建的问题。虽然它将“对齐”视为一个多智能体协作问题，但文章的核心贡献、优化方法（GSPO）和实验评估主要集中在“大模型对齐与偏好学习”这一特定应用领域，对于多智能体之间的具体分工、通信协议和协作推理机制的探讨非常有限。|
|[Hierarchical Multi-agent Large Language Model Reasoning for Autonomous Functional Materials Discovery](https://arxiv.org/abs/2512.13930v1)|2|concept新增|摘要中明确提出了在功能材料发现这一复杂任务中，设计并对比了单智能体基线与三种具体的多智能体协作机制（“同行评审”、“分类排序”和“分类表单”）。这属于应用论文中提出了具体的协作推理机制并进行了实验分析，符合“提出可迁移的协作机制，并通过实验分析该机制”的最高评分标准。|
|[From Visual Perception to Deep Empathy: An Automated Assessment Framework for House-Tree-Person Drawings Using Multimodal LLMs and Multi-Agent Collaboration](https://arxiv.org/abs/2512.21360v1)|1|仅baseline|摘要表明该研究在临床心理学（房树人测试）的具体应用中使用了多智能体系统。虽然提到了智能体之间“通过划分角色，将特征识别与心理推断解耦”，但这是一种高度绑定于特定垂直领域应用的分工方式。其核心贡献和实验评估（如与人类专家对比的语义相似度）均集中于心理学评估这一特定任务，对通用或可迁移的协作方法的讨论有限|
|[Absorbing State Phase Transitions in Multi-Agent Search](https://arxiv.org/abs/2609.38327v2)|2|concept新增|摘要明确指出该研究专注于“多智能体通信拓扑”的设计，并从理论和实验层面深入探讨了多智能体系统中的“临界通信度”以及智能体之间的实际沟通行为。这直接研究了 LLM 多智能体系统的底层通信、协调与任务完成机制，完全符合最高评分标准。|

## 解释与下一步

以上结果限于本次三个已参与查询设计的主题，尚不能外推到任意研究想法。
应结合新增相关论文、未保留的原查询相关论文及相关比例评价收益；不能仅凭返回数量增加判断优化成功。
若要接入主流程，建议增加独立测试主题，并保留完整短语基线及可切换策略。
