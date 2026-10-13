"""Small Chat Completions client. No paid service or model is hard-coded."""
import asyncio
import json
import os
import httpx


def parse_json(text):
    text = text.strip()
    if text.startswith("```"):
        text = "\n".join(text.splitlines()[1:-1])
    return json.loads(text)


class LLM:
    def __init__(self):
        self.base = os.getenv("LLM_BASE_URL", "").rstrip("/")
        self.key = os.getenv("LLM_API_KEY", "")
        self.model = os.getenv("LLM_MODEL", "")
        self.trust_env = os.getenv("LLM_TRUST_ENV", "1") == "1"
        self.max_tokens = int(os.getenv("LLM_MAX_TOKENS", "2048"))
        self.tokenize = os.getenv("LLM_TOKENIZE", "0") == "1"
        self.context_tokens = int(os.getenv("LLM_CONTEXT_TOKENS", "8192"))
        self.backend = os.getenv("LLM_BACKEND", "vllm").lower()
        if self.backend not in ("llamacpp", "vllm"):
            raise ValueError("LLM_BACKEND 必须为 llamacpp 或 vllm")
        self.server_root = self.base.removesuffix("/v1")
        self.tokenize_url = os.getenv("LLM_TOKENIZE_URL", self.base.removesuffix("/v1") + "/tokenize")
        self.options = {}
        for env, field, cast in (("LLM_TEMPERATURE", "temperature", float),
                                 ("LLM_TOP_P", "top_p", float), ("LLM_TOP_K", "top_k", int)):
            if os.getenv(env):
                self.options[field] = cast(os.environ[env])
        if os.getenv("LLM_ENABLE_THINKING") in ("0", "1"):
            self.options["chat_template_kwargs"] = {
                "enable_thinking": os.environ["LLM_ENABLE_THINKING"] == "1"}
        if self.max_tokens <= 0 or self.context_tokens <= 0:
            raise ValueError("生成长度和上下文长度必须大于 0")
        if not self.base or not self.model:
            raise ValueError("请在 .env 设置 LLM_BASE_URL 和 LLM_MODEL；离线调试使用 --mock。")
        if not self.key and os.getenv("LLM_ALLOW_NO_KEY") != "1":
            raise ValueError("请设置 LLM_API_KEY；无鉴权本地服务可设置 LLM_ALLOW_NO_KEY=1。")

    def headers(self):
        return {"Authorization": f"Bearer {self.key}"} if self.key else {}

    async def fits(self, messages, tools=None, max_tokens=None):
        if not self.tokenize:
            return True
        payload = {"model": self.model, "messages": messages, "add_generation_prompt": True}
        if tools:
            payload["tools"] = tools
        if "chat_template_kwargs" in self.options:
            payload["chat_template_kwargs"] = self.options["chat_template_kwargs"]
        async with httpx.AsyncClient(timeout=30, trust_env=self.trust_env) as client:
            try:
                if self.backend == "llamacpp":
                    props = await client.get(self.server_root + "/props", headers=self.headers())
                    self.check_preflight_response(props, "/props")
                    template = await client.post(self.server_root + "/apply-template",
                                                 json=payload, headers=self.headers())
                    self.check_preflight_response(template, "/apply-template")
                    try:
                        server_limit = int(props.json()["default_generation_settings"]["n_ctx"])
                        prompt = template.json()["prompt"]
                        if server_limit <= 0 or not isinstance(prompt, str):
                            raise ValueError
                    except (KeyError, TypeError, ValueError):
                        raise RuntimeError("llama.cpp 上下文/模板响应无效，请检查 llama-server 版本") from None
                    payload = {"content": prompt, "add_special": False, "parse_special": True}
                response = await client.post(self.tokenize_url, json=payload, headers=self.headers())
            except httpx.TransportError:
                raise RuntimeError("本地上下文检查请求失败，请确认模型服务已启动") from None
        self.check_preflight_response(response, "/tokenize")
        try:
            value = response.json()
            if self.backend == "llamacpp":
                tokens = value["tokens"]
                if not isinstance(tokens, list):
                    raise ValueError
                count = len(tokens)
            else:
                server_limit = int(value["max_model_len"])
                count = int(value["count"])
            if count < 0 or server_limit <= 0:
                raise ValueError
        except (KeyError, TypeError, ValueError):
            raise RuntimeError("本地分词响应无效，请检查 LLM_BACKEND 与模型服务是否一致") from None
        limit = min(self.context_tokens, server_limit)
        return count + (max_tokens or self.max_tokens) + 128 <= limit

    def check_preflight_response(self, response, endpoint):
        if response.is_error:
            raise RuntimeError(f"{endpoint} HTTP {response.status_code}；请检查 {self.backend} 服务版本与配置")

    async def fits_text(self, system, user, max_tokens=None):
        return await self.fits(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens=max_tokens,
        )

    async def chat(self, messages, tools=None, max_tokens=None):
        if not await self.fits(messages, tools, max_tokens):
            raise ValueError("输入与预留输出超过本地模型上下文；请减少候选数或提高模型服务的上下文长度")
        payload = {"model": self.model, "messages": messages,
                   "max_tokens": max_tokens or self.max_tokens, **self.options}
        if tools:
            payload.update(tools=tools, tool_choice="auto")
        async with httpx.AsyncClient(timeout=300, trust_env=self.trust_env) as client:
            for attempt in range(3):
                try:
                    response = await client.post(
                        self.base + "/chat/completions", json=payload, headers=self.headers()
                    )
                except httpx.TransportError:
                    if attempt == 2:
                        raise RuntimeError("LLM 网络请求失败") from None
                    await asyncio.sleep(2 ** attempt)
                    continue
                if response.status_code in (429, 500, 502, 503, 504) and attempt < 2:
                    await asyncio.sleep(2 ** attempt)
                    continue
                if response.is_error:
                    # Avoid leaking keys, provider URLs, or response bodies into logs.
                    raise RuntimeError(f"LLM HTTP {response.status_code}，请检查配置/模型/额度。")
                choice = response.json()["choices"][0]
                if choice.get("finish_reason") == "length":
                    raise ValueError("模型输出达到长度上限，未作为完整结果使用；请增加 LLM_MAX_TOKENS 并预留上下文")
                message = choice["message"]
                return {k: v for k, v in message.items()
                        if k in ("role", "content", "tool_calls")}
        raise RuntimeError("LLM 请求未完成")

    async def text(self, system, user, max_tokens=None):
        result = await self.chat([
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ], max_tokens=max_tokens)
        content = result.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("模型返回空文本")
        return content

    async def merge_notes(self, system, label, notes, max_tokens=None, reduce_max_tokens=None):
        """Reduce all notes recursively when a local model cannot fit them together."""
        notes = list(notes)
        for _ in range(10):
            user = label + "\n\n" + "\n\n".join(notes)
            if await self.fits_text(system, user, max_tokens=max_tokens):
                return await self.text(system, user, max_tokens=max_tokens)
            if len(notes) < 2:
                raise ValueError("单条笔记已超出上下文，请增大上下文或缩短输出长度")
            reduced = []
            for i in range(0, len(notes), 2):
                pair = notes[i:i + 2]
                if len(pair) == 1:
                    reduced.extend(pair)
                else:
                    reduced.append(await self.text(
                        system + "先压缩这些笔记到 300 字左右；保留所有论文 ID、关键证据页码和不确定性。",
                        label + "\n\n" + "\n\n".join(pair),
                        max_tokens=reduce_max_tokens or max_tokens,
                    ))
            notes = reduced
        raise ValueError("笔记层次压缩未收敛")
