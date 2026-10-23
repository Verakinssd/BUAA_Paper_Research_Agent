# README

* 将“整篇并发”改成“分阶段并发”：修改前的 [pipeline.py](research/pipeline.py) 使用一个信号量包住整篇论文处理流程，并发度固定为 2。拆成：
  - PDF 下载：并发 3～4
  - PDF 转换：并发 1，避免 CPU 和内存压力
  - API 总结：并发 2，避免 API 限流
  - arXiv 下载：继续保持限速
* 完善对比。做了benchmark对比，选用十篇论文（5篇arxiv，5篇其他渠道，在benchmark_manifest_10.json）生成了真实环境下时间对比详细报告。由于报告结果受网络速度等客观因素影响过大，还进行了模拟实验，证明了并发策略优化后的有效性。
* **完整 benchmark 及说明报告在 `outputs/benchmarks/` 文件夹下**。

# Paper Research Agent

一个本地论文研究 Agent：根据研究想法检索 arXiv 和 OpenAlex，筛选 Top-K 论文，读取可访问的 PDF，并生成逐篇总结和研究报告。

## 快速部署

### 1. 准备环境

建议使用 Linux、Python 3.11、32 GB 内存、30 GB 以上可用磁盘和 24 GB 显存的 NVIDIA GPU。需预装 Git、curl、CMake、C/C++ 编译器、NVIDIA 驱动及与驱动兼容的 CUDA Toolkit，并确保 `nvcc` 在 PATH 中。

```bash
git clone <YOUR_REPOSITORY_URL> Paper-Research-Agent
cd Paper-Research-Agent

conda create -n paper-agent python=3.11 -y
conda activate paper-agent
python -m pip install -r requirements.txt
```

### 2. 安装 llama.cpp

以下命令使用已验证的 `b11586` 版本，构建支持 NVIDIA CUDA 的 `llama-server`：

```bash
git clone --branch b11586 --depth 1 https://github.com/ggml-org/llama.cpp.git
cmake -S llama.cpp -B llama.cpp/build   -DGGML_CUDA=ON   -DCMAKE_BUILD_TYPE=Release   -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.2/bin/nvcc   -DCMAKE_CUDA_ARCHITECTURES=89   -DTHREADS_PREFER_PTHREAD_FLAG=ON
cmake --build llama.cpp/build --config Release --target llama-server -j 4
```

生成的程序通常位于 `llama.cpp/build/bin/llama-server`。CPU 或其他加速后端请按 llama.cpp 官方文档选择相应构建选项。

### 3. 准备模型

将 Qwen3.8-27B 的 `Q4_K_M` GGUF 放入项目的 `models/` 目录。

```bash
mkdir -p models
curl -k --location --fail --retry 5 --continue-at - \
  --output models/Qwen3.8-27B-Q4_K_M.gguf \
  https://huggingface.co/bartowski/Qwen3.8-27B-GGUF/resolve/main/Qwen3.8-27B-Q4_K_M.gguf
```

需要在网页切换到 Q5 时，再下载 `Q5_K_M`（约 20.9 GB）。保留 Q4 文件：

```bash
curl --location --fail --retry 20 --retry-all-errors \
  --retry-delay 5 --connect-timeout 20 --continue-at - \
  --output models/Qwen3.8-27B-Q5_K_M.gguf.part \
  https://huggingface.co/bartowski/Qwen3.8-27B-GGUF/resolve/main/Qwen3.8-27B-Q5_K_M.gguf \
&& mv models/Qwen3.8-27B-Q5_K_M.gguf.part models/Qwen3.8-27B-Q5_K_M.gguf
```

### 4. 配置项目

```bash
cp .env.example .env
```

编辑 `.env`，至少填写下面两项的绝对路径：

```dotenv
GGUF_MODEL_PATH=/absolute/path/to/Paper-Research-Agent/models/Qwen3.8-27B-Q4_K_M.gguf
LLAMA_SERVER_BIN=/absolute/path/to/llama.cpp/build/bin/llama-server
```

默认配置使用单个并行槽和每请求 8192 token 上下文。

### 5. 启动模型服务

在第一个终端中运行：

```bash
conda activate paper-agent
cd Paper-Research-Agent
python deploy/serve_gguf.py --managed
```

保持该进程运行。推理监听 `127.0.0.1:8000`，模型切换接口监听 `127.0.0.1:8001`。已有旧模型服务时，先等待研究任务结束，在旧服务终端按 Ctrl+C，再执行上述命令。省略 `--managed` 仍可使用原来的固定模型模式。

### 6. 打开网页界面

网页运行示例：

![Paper Research Agent 网页运行示例](demo.png)

首次部署先在项目目录创建管理员（交互式设置用户名和密码，无默认账号）：

```bash
python manage_users.py create-admin
```

在同一台服务器的第二个终端中运行：

```bash
conda activate paper-agent
cd Paper-Research-Agent
python web_app.py --port 7860
```

浏览器打开 `http://127.0.0.1:7860`，注册普通用户或登录管理员账号后即可开始。个人空间支持修改用户名、头像、签名和密码；管理员还可管理用户身份、启用/禁用账号、重置用户密码，以及切换全局模型。

如果运行在远程服务器，可用编辑器的端口转发功能，或在自己的电脑上运行：

```bash
ssh -N -L 7860:127.0.0.1:7860 用户名@运行网页的服务器
```

### 命令行运行（可选）

在同一台服务器的第二个终端中运行：

```bash
conda activate paper-agent
cd Paper-Research-Agent

python check_local_model.py
python main_agent.py \
  --idea "我想研究LoRA" \
  --top-k 3 \
  --concurrent
```

命令行结果保存在 `outputs/<run_id>/`：

- `report.md`：最终研究报告
- `topk.json`：Top-K 论文和处理状态
- `metrics.json`：耗时与成功状态
- `fulltext/`、`summaries/`：论文文本和逐篇总结

确认完整成功时，检查 `metrics.json` 中 `task_success=true`，且 `completed_count` 等于请求的 `--top-k`。候选不足时实际数量可能少于 K；PDF 不可访问或来源限流也可能导致部分失败。

## 网络与常见问题

- 论文检索和 PDF 下载需要服务器能够访问 arXiv 与 OpenAlex。
- 使用系统代理时，在 `.env` 中设置 `PAPER_TRUST_ENV=1`，并在启动前配置 `HTTP_PROXY`、`HTTPS_PROXY`；也可以直接设置 `PAPER_PROXY_URL=http://代理地址:端口`。
- `Connection refused` 通常表示模型服务未启动，或 `LLM_BASE_URL` 与监听端口不一致。

## 测试

```bash
python -m unittest discover -s tests -v
```

## 相关性排序方法扩展

在原来basic方法的基础上，新增了bm25、dense_linear、hybrid_rrf方法。dense_linear、hybrid_rrf方法需要部署embedding模型。这里用的BGE-M3模型。

### 部署BGE-M3

安装pytorch、FlagEmbedding、FastAPI

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -U FlagEmbedding
pip install -U huggingface_hub
pip install fastapi "uvicorn[standard]"
```

### 启动BGE-M3服务

```bash
uvicorn deploy.serve_bge:app --host 127.0.0.1 --port 8002
```
### 测试BGE-M3服务

```bash
curl -X POST http://127.0.0.1:8000/embed \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "LoRA训练",
      "多智能体学习"
    ]
  }'
```

### 配置相关性排序方法

在`.env`中配置相关性排序方法

```dotenv
PAPER_RANKING_METHOD=bm25
```

如果没有`PAPER_RANKING_METHOD`，默认使用`basic`方法。

---

## 新增功能和使用说明

### 1. 不用本地显卡，也可以使用自己的 API

电脑显存不足时，可以在网页中填写网络 API，不用下载本地大模型，也不用启动 `llama-server`。原来的本地模型功能仍然可以使用。

首次安装时，在项目目录执行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

如果还没有 `.env` 文件，先执行下面的命令；已有 `.env` 时不要重复复制，以免覆盖自己的配置：

```bash
cp .env.example .env
```

用文本编辑器打开 `.env`，设置：

```dotenv
PAPER_RANKING_METHOD=bm25
PDF_LAYOUT_MODE=legacy
```

`bm25` 是按关键词给论文排序的方法，不需要额外运行向量模型。`legacy` 是直接提取 PDF 文字的方式。使用网络 API 时，本地模型路径可以留空。

启动网页：

```bash
python web_app.py --port 7860
```

浏览器打开 `http://127.0.0.1:7860`，按下面的步骤操作：

1. 注册自己的账号并登录。
2. 点击工作台的“配置个人 API”，也可以进入“个人空间 → 模型与个人 API”。
3. 模型来源选择“网络 API”。
4. 填写 API 地址（Base URL）、模型名称和 API Key。
5. 上下文长度按模型实际支持的范围填写，不清楚时可以先保留默认的 `32768`。
6. 点击“测试连接”，成功后点击“保存模型配置”。测试连接本身不会保存配置。
7. 回到工作台输入研究想法，点击“开始研究”。

当前页面需要手动填写地址和模型名称，没有“读取模型”或“记住 Key”选项。“测试连接”会实际发送一次简短的工具调用请求，不只是读取模型列表。模型需支持 OpenAI 兼容的 Chat Completions 接口和工具调用（Tool Calling）。

每个账号保存自己的 API 配置。API Key 会加密保存，网页不会显示已保存的原 Key。保存后把输入框留空，会保留原 Key；更换 API 地址时需要重新填写 Key。当前加密文件为 `data/api_settings.key`，账号和加密后的配置保存在 `data/accounts.sqlite3`。迁移项目数据时需要一起保留。

### 2. 在页面下载论文全文 Markdown

进入“研究记录”，打开一项任务，在论文卡片中可以看到：

- “查看 PDF”：查看下载的论文原文。
- “下载全文 Markdown”：下载由 PDF 转换得到的全文，文件生成后才会显示按钮。
- “阅读全文总结”：查看模型生成的单篇论文总结，与全文 Markdown 不是同一个内容。
- “研究报告”：查看本次研究的综合报告，并可下载报告。

网页任务的文件保存在项目目录中：

```text
outputs/_web/runs/任务ID/运行ID/
├── fulltext/       论文 PDF 和转换后的全文 Markdown
├── summaries/      单篇论文总结
├── report.md       综合研究报告
└── metrics.json    本次任务的运行结果
```

论文缓存默认保存在 `outputs/_cache/fulltext/`，方便以后复用。当前没有自动清理功能，关闭网页或重启电脑不会自动删除这些文件，但项目没有自动备份功能。

### 3. 可以在页面选择是否并发处理

工作台“论文数量”旁新增了“并发处理”勾选框，默认不勾选。使用免费 API 时，建议先选择 Top 1，并保持不勾选，成功后再尝试更多论文。

这个开关控制的是一项研究内部是否并行处理，不是让多个人同时开始研究。目前一个网页服务进程同一时间只运行一个研究任务；其他人仍然可以打开网站、登录和查看自己的记录。

默认未设置 `PIPELINE_MODE` 时，上述开关保持原行为：不勾选为串行，勾选后以整篇论文为单位并发 2。需要启用分阶段调度时，在 `.env` 中设置：

```dotenv
PIPELINE_MODE=staged
DOWNLOAD_CONCURRENCY=4
CONVERT_CONCURRENCY=1
SUMMARY_CONCURRENCY=2
PIPELINE_MAX_IN_FLIGHT=8
```

`staged` 只在对应阶段持有信号量：下载完成后立即释放下载槽，再等待转换槽；转换完成后释放转换槽，再等待总结槽。arXiv 下载仍由原有限速锁串行执行并保持至少 3.1 秒请求间隔。`PIPELINE_MAX_IN_FLIGHT` 使用固定 worker 数限制同时在流水线中的论文数量。

### 4. 对比 legacy 与 staged

仓库提供固定测试集 `benchmark_manifest_10.json` 和独立脚本 `scripts/benchmark_pipeline.py`。脚本按顺序运行两种模式，每轮使用独立输出目录和缓存，并自动生成 `benchmark_results.json` 与 `benchmark_report.md`。

Benchmark 是独立 CLI，不会解密任一网页账号保存的个人 API 配置。使用网络模型时，请先把同一套模型配置写入本机 `.env`（该文件已被 Git 忽略），例如 `LLM_BACKEND=openai`、`LLM_BASE_URL=https://openrouter.ai/api/v1`、`LLM_MODEL=固定模型ID`、`LLM_API_KEY=你的Key`、`LLM_ALLOW_NO_KEY=0`；报告只记录非秘密配置，不记录 Key。

真实测试会重复下载论文并调用模型 API，可能产生费用、消耗免费额度或触发服务商限流。确认风险后运行：

```bash
python scripts/benchmark_pipeline.py \
  --manifest benchmark_manifest_10.json \
  --repeats 3 \
  --download-concurrency 4 \
  --convert-concurrency 1 \
  --summary-concurrency 2 \
  --max-in-flight 8 \
  --acknowledge-api-cost
```

结果默认保存到 `outputs/benchmarks/benchmark-<时间>/`。也可以用 `--mode legacy` 或 `--mode staged` 单独执行一种模式；默认 `--mode both` 仍是严格顺序执行，不会同时运行两种模式。为了公平比较，应使用固定模型 ID；`openrouter/free` 可能在不同请求间选择不同模型，不适合严谨性能结论。

### 5. 建立临时 HTTPS 测试链接

先完成前面的依赖安装和配置。打开两个终端，都进入项目目录，并使用项目的 Python 环境。

**终端 1：启动测试网页。**

```bash
.venv/bin/python deploy/serve_https_preview.py --port 7862
```

如果使用的是 Conda 环境，激活后把 `.venv/bin/python` 换成 `python` 即可。看到 `Uvicorn running on http://127.0.0.1:7862` 后，保持运行。这是专门给 HTTPS 隧道使用的入口，不要直接用这个 HTTP 地址登录，也不要把它改为监听 `0.0.0.0`。

**终端 2：建立隧道。**

```bash
ssh -T -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -o ExitOnForwardFailure=yes -R 80:127.0.0.1:7862 nokey@localhost.run
```

首次连接可能询问是否信任服务器。核对主机名和指纹后，输入完整的 `yes` 并回车，不是 `y`。如果以后提示服务器指纹发生变化，应先核实，不要直接关闭检查。

成功后，终端会显示类似 `https://xxxx.lhr.life` 的地址。复制终端里实际生成的地址发给别人，不要照抄这里的示例。别人可以使用自己的网络打开，不需要你的校园网学号或密码。

使用时注意：

- 两个终端都要保持运行，电脑保持联网和唤醒。隧道命令不会自动启动网站。
- 测试者在网站注册自己的账号，填写自己的 API Key；这些数据保存在运行项目的电脑上。链接经过第三方隧道转发，只分享给需要测试的人。
- 免费地址会定期更换，没有保证的固定时长。最新地址会显示在终端 2，旧网页不会自动跳转，换地址后需要重新登录。
- 停止分享：在终端 2 按 `Ctrl+C`。如果也要关闭网站，等研究任务结束后，在终端 1 按 `Ctrl+C`。
- 临时链接适合短期测试。要长期使用固定地址，需要稳定隧道或服务器部署。
