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
cmake -S llama.cpp -B llama.cpp/build \
  -DGGML_CUDA=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build llama.cpp/build --config Release --target llama-server -j 4
```

生成的程序通常位于 `llama.cpp/build/bin/llama-server`。CPU 或其他加速后端请按 llama.cpp 官方文档选择相应构建选项。

### 3. 准备模型

将 Qwen3.8-27B 的 `Q4_K_M` GGUF 放入项目的 `models/` 目录。

```bash
mkdir -p models
curl --location --fail --retry 5 --continue-at - \
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

在同一台服务器的第二个终端中运行：

```bash
conda activate paper-agent
cd Paper-Research-Agent
python web_app.py --port 7860
```

浏览器打开 `http://127.0.0.1:7860`，输入研究想法即可开始。

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
- 可先运行 `python main_agent.py --mock --top-k 3` 检查 MCP、PDF 处理和输出流程。

## 测试

```bash
python -m unittest discover -s tests -v
```
