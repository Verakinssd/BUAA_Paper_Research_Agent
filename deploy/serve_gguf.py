"""Serve a user-supplied local GGUF with llama.cpp; never download weights."""
import argparse
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from dotenv import load_dotenv


def build_command(args):
    if not args.model:
        raise ValueError("请传 --model /绝对路径/model.gguf，或在 .env 设置 GGUF_MODEL_PATH")
    model = Path(args.model).expanduser().resolve()
    if not model.is_file() or model.suffix.lower() != ".gguf":
        raise ValueError("模型路径必须是已下载的 .gguf 文件，不能是模型目录或 safetensors")
    with model.open("rb") as stream:
        if stream.read(4) != b"GGUF":
            raise ValueError("文件头不是 GGUF；请检查下载结果")
    if os.getenv("SLURM_JOB_ID"):
        mask = os.getenv("CUDA_VISIBLE_DEVICES", "")
        if not mask or mask in ("-1", "NoDevFiles"):
            raise ValueError("当前 Slurm 作业未分配可见 GPU，请先申请 GPU")
    if (args.context <= 0 or args.parallel <= 0 or args.gpu_layers < 0
            or not 1 <= args.port <= 65535):
        raise ValueError("context/parallel 必须为正数，gpu-layers 不能为负，port 必须为有效端口")
    # llama.cpp divides the total context across parallel slots. Treat our
    # --context as the per-request budget so enabling a second slot does not
    # silently reduce an 8K client to 4K.
    total_context = args.context * args.parallel
    return [args.server_bin, "--model", str(model), "--offline",
            "--alias", "qwen-local", "--host", "127.0.0.1", "--port", str(args.port),
            "--ctx-size", str(total_context), "--parallel", str(args.parallel),
            "--n-gpu-layers", str(args.gpu_layers), "--split-mode", "none", "--main-gpu", "0",
            "--batch-size", "256", "--ubatch-size", "128", "--flash-attn", "on",
            "--jinja", "--chat-template-kwargs", '{"enable_thinking":false}']


def main():
    load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.getenv("GGUF_MODEL_PATH"), help="本地 GGUF 文件路径")
    parser.add_argument("--server-bin", default=os.getenv("LLAMA_SERVER_BIN") or "llama-server")
    parser.add_argument("--context", type=int, default=int(os.getenv("LLM_CONTEXT_TOKENS", "8192")))
    parser.add_argument("--parallel", type=int, default=int(os.getenv("LLAMA_PARALLEL", "1")),
                        help="并行槽位数；--context 是每个槽位的上下文长度")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--gpu-layers", type=int, default=99, help="GPU 层数；99 尝试完整卸载到 GPU")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--managed", action="store_true", help="启用网页 Q4/Q5 模型切换")
    parser.add_argument("--model-id", choices=["q4_k_m", "q5_k_m"], help="可切换模式的初始模型")
    parser.add_argument("--control-port", type=int, default=8001, help="仅监听本机的模型切换端口")
    args = parser.parse_args()
    if args.managed:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from research.model_manager import create_control_app, model_profiles
        profiles = model_profiles(args=args)
        initial_id = args.model_id or ("q5_k_m" if "Q5_K_M" in (args.model or "") else "q4_k_m")
        initial_args = profiles[initial_id]["args"]
    else:
        initial_args = args
    try:
        command = build_command(initial_args)
        if args.managed and (not 1 <= args.control_port <= 65535 or args.control_port == args.port):
            raise ValueError("control-port 必须是与推理端口不同的有效端口")
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print("CUDA_VISIBLE_DEVICES=" + os.getenv("CUDA_VISIBLE_DEVICES", "<未设置，使用可见设备中的逻辑 GPU 0>"), flush=True)
    print(shlex.join(command), flush=True)
    if args.dry_run:
        return
    if not shutil.which(args.server_bin):
        parser.error("找不到 llama-server；请用 --server-bin 指定支持该模型的 CUDA 版本可执行文件")
    if args.managed:
        import uvicorn
        print("网页模型切换已启用；Q4/Q5 共享一份 GPU，推理别名仍为 qwen-local。", flush=True)
        uvicorn.run(create_control_app(args, initial_id), host="127.0.0.1", port=args.control_port)
        return
    # Pass the inherited environment unchanged; Slurm owns the GPU mask.
    raise SystemExit(subprocess.call(command))


if __name__ == "__main__":
    main()
