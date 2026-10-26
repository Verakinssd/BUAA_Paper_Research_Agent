"""Checks to run before spending hours on a benchmark.

The failure it prevents: the model server is down (or misconfigured), the run "finishes"
quickly with a 0% success rate, and nobody notices until the comparison is written up.

Errors stop the run; warnings are printed. Settings that change the generated text
(temperature, thinking) are errors for a full-dataset run because `compare` only makes
sense between runs that decode the same way.
"""
import os
import time
import httpx


class Report:
    def __init__(self):
        self.errors, self.warnings, self.info = [], [], {}

    @property
    def ok(self):
        return not self.errors


def _get(url, timeout):
    return httpx.get(url, timeout=timeout, trust_env=False)


def check(env=None, *, full=True, workers=1, concurrent=False, timeout=60):
    """Return a Report. `env` defaults to os.environ; `full` means a formal full-dataset run."""
    env = os.environ if env is None else env
    report = Report()
    base = env.get("LLM_BASE_URL", "").rstrip("/")
    model = env.get("LLM_MODEL", "")
    if not base or not model:
        report.errors.append("未设置 LLM_BASE_URL / LLM_MODEL（检查 .env）")
        return report

    pinned = report.errors if full else report.warnings
    suffix = "" if full else "（调试运行只警告；完整运行必须固定）"
    temperature = env.get("LLM_TEMPERATURE", "")
    try:
        if float(temperature) != 0:
            pinned.append(f"LLM_TEMPERATURE={temperature}，应为 0，否则两次运行的输出不可复现{suffix}")
    except ValueError:
        pinned.append("未设置 LLM_TEMPERATURE：会使用服务端默认采样，应显式设为 0" + suffix)
    if env.get("LLM_ENABLE_THINKING") not in ("0", "1"):
        pinned.append("未设置 LLM_ENABLE_THINKING（0 或 1），不同运行的思考开关可能不一致" + suffix)

    root = base.removesuffix("/v1")
    try:
        health = _get(root + "/health", 10)
        if health.status_code != 200:
            report.errors.append(f"模型服务 {root}/health 返回 {health.status_code}（模型可能还在加载）")
            return report
    except httpx.HTTPError as exc:
        report.errors.append(f"连不上模型服务 {root}：{exc.__class__.__name__}。请先启动 deploy/serve_gguf.py")
        return report

    try:
        props = _get(root + "/props", 10).json()
        ctx = props.get("default_generation_settings", {}).get("n_ctx")
        slots = props.get("total_slots")
        report.info.update(n_ctx_per_slot=ctx, total_slots=slots,
                           model=str(props.get("model_path") or props.get("model_alias") or ""))
        need_ctx = int(env.get("LLM_CONTEXT_TOKENS", "8192"))
        if ctx and ctx < need_ctx:
            report.errors.append(f"服务端每个槽位的上下文只有 {ctx}，小于 LLM_CONTEXT_TOKENS={need_ctx}，"
                                 "长文分块会超出上下文。请调大 serve_gguf 的 --context")
        need_slots = workers * (2 if concurrent else 1)
        if slots and slots < need_slots:
            report.warnings.append(f"服务端只有 {slots} 个并行槽位，少于需要的 {need_slots}（workers={workers}"
                                   f"{'，--concurrent' if concurrent else ''}）；多出的请求会排队，单条耗时被拉长。"
                                   f"请用 LLAMA_PARALLEL={need_slots} 重启模型服务")
    except (httpx.HTTPError, ValueError):
        report.warnings.append("读不到服务端 /props（非 llama.cpp？），无法核对上下文和槽位")

    body = {"model": model, "max_tokens": 8, "temperature": 0,
            "messages": [{"role": "user", "content": "Reply with the single word OK."}]}
    headers = {"Authorization": "Bearer " + env["LLM_API_KEY"]} if env.get("LLM_API_KEY") else {}
    started = time.perf_counter()
    try:
        reply = httpx.post(base + "/chat/completions", json=body, headers=headers,
                           timeout=timeout, trust_env=False)
        if reply.status_code != 200:
            report.errors.append(f"试调用失败：HTTP {reply.status_code} {reply.text[:120]}")
        elif not reply.json().get("choices"):
            report.errors.append("试调用没有返回 choices，模型服务异常")
        else:
            report.info["probe_seconds"] = round(time.perf_counter() - started, 2)
    except (httpx.HTTPError, ValueError) as exc:
        report.errors.append(f"试调用失败：{exc.__class__.__name__} {exc}")
    return report


def render(report):
    lines = [f"错误：{m}" for m in report.errors] + [f"警告：{m}" for m in report.warnings]
    if report.ok:
        info = report.info
        lines.append("预检通过：" + "，".join(f"{k}={v}" for k, v in info.items() if v not in (None, "")))
    return "\n".join(lines)
