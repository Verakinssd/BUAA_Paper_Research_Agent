"""Launch one local vLLM process shared by both Agents (Linux/WSL2)."""
import argparse
import os
import shlex
import shutil
import subprocess

PROFILES = {
    "3090-bf16": {"model": "Qwen/Qwen3-8B", "dtype": "bfloat16", "length": 8192, "seqs": 2, "tp": 1},
    "3090-awq": {"model": "Qwen/Qwen3-14B-AWQ", "dtype": "half", "length": 8192, "seqs": 2, "tp": 1},
    "3080-awq": {"model": "Qwen/Qwen3-8B-AWQ", "dtype": "half", "length": 8192, "seqs": 1, "tp": 1},
    "dual3090-bf16": {"model": "Qwen/Qwen3-14B", "dtype": "bfloat16", "length": 8192, "seqs": 2, "tp": 2},
}


def build_command(args):
    profile = PROFILES[args.profile]
    inherited = os.getenv("CUDA_VISIBLE_DEVICES", "")
    in_slurm = bool(os.getenv("SLURM_JOB_ID"))
    if in_slurm and args.devices is not None:
        raise ValueError("Slurm 作业中不要传 --devices；保留调度器分配的 CUDA_VISIBLE_DEVICES")
    if in_slurm and (not inherited or inherited in ("-1", "NoDevFiles")):
        raise ValueError("Slurm 作业中没有可用的 CUDA_VISIBLE_DEVICES，请先申请 GPU 并进入 srun 作业步骤")
    selected = args.devices if args.devices is not None else (inherited or "0")
    devices = [d.strip() for d in selected.split(",") if d.strip()]
    if len(devices) < profile["tp"] or len(set(devices)) != len(devices):
        raise ValueError(f"该配置至少需要 {profile['tp']} 个不同的已分配/已选择 GPU ID")
    command = ["vllm", "serve", args.model or profile["model"],
               "--served-model-name", "qwen-local", "--dtype", profile["dtype"],
               "--host", "127.0.0.1", "--port", str(args.port),
               "--tensor-parallel-size", str(profile["tp"]),
               "--max-model-len", str(args.context or profile["length"]),
               "--max-num-seqs", str(profile["seqs"]), "--gpu-memory-utilization", "0.90",
               "--enable-auto-tool-choice", "--tool-call-parser", "hermes", "--enforce-eager"]
    # None means preserve the scheduler/process environment without renumbering.
    override = None if in_slurm or (args.devices is None and inherited) else ",".join(devices)
    return command, override


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=PROFILES, default="3090-bf16")
    parser.add_argument("--devices", help="Only outside Slurm: physical GPU indices, e.g. 0 or 0,1")
    parser.add_argument("--model", help="Optional local model directory")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--context", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    command, devices = build_command(args)
    prefix = "继承 CUDA_VISIBLE_DEVICES=" + os.getenv("CUDA_VISIBLE_DEVICES", "") if devices is None else "CUDA_VISIBLE_DEVICES=" + devices
    print(prefix + " " + shlex.join(command), flush=True)
    if args.dry_run:
        return
    if not shutil.which("vllm"):
        raise SystemExit("未找到 vllm，请激活模型服务环境并安装 vLLM。")
    env = dict(os.environ)
    if devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = devices
    subprocess.run(command, env=env, check=True)


if __name__ == "__main__":
    main()
