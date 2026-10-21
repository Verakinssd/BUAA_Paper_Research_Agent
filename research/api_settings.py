"""Per-account model settings. Secrets never enter public responses or job files."""
import json
import os
from urllib.parse import urlsplit

from cryptography.fernet import Fernet, InvalidToken
import httpx
from starlette.exceptions import HTTPException
from .llm import api_error, provider_options


DEFAULTS = dict(mode="local", base_url="", model="", context_tokens=32768)
FIELDS = set(DEFAULTS) | {"api_key"}


class APISettings:
    def __init__(self, accounts):
        self.accounts = accounts

    def _cipher(self):
        # Keep this file when migrating/backing up the account database.
        path = self.accounts.data_dir / "api_settings.key"
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            pass
        else:
            with os.fdopen(fd, "wb") as stream:
                stream.write(Fernet.generate_key())
        return Fernet(path.read_bytes())

    def read(self, user_id):
        with self.accounts._connection() as connection:
            self.accounts._user(connection, user_id)
            row = connection.execute("SELECT value FROM model_settings WHERE user_id = ?", (user_id,)).fetchone()
        if row is None:
            return {**DEFAULTS, "api_key": ""}
        value = json.loads(row[0])
        try:
            value["api_key"] = self._cipher().decrypt(value["api_key"].encode()).decode() if value["api_key"] else ""
        except (InvalidToken, ValueError):
            raise HTTPException(503, "无法解密个人 API Key，请恢复 data/api_settings.key。") from None
        return value

    @staticmethod
    def public(config):
        return {**{field: config[field] for field in DEFAULTS}, "has_api_key": bool(config["api_key"])}

    def prepare(self, user_id, payload):
        if not isinstance(payload, dict) or set(payload) - FIELDS:
            raise HTTPException(400, "模型配置包含不支持的字段。")
        current = self.read(user_id)
        config = {**current, **payload}
        if config["mode"] not in ("local", "remote"):
            raise HTTPException(400, "请选择本地模型或网络 API。")
        for field, limit in (("base_url", 2048), ("model", 200), ("api_key", 4096)):
            value = config[field]
            if not isinstance(value, str) or len(value) > limit or any(ord(c) < 32 or ord(c) == 127 for c in value):
                raise HTTPException(400, "API 地址、模型名称或 Key 格式不正确。")
            config[field] = value.strip()
        if config["api_key"] and (not config["api_key"].isascii() or any(c.isspace() for c in config["api_key"])):
            raise HTTPException(400, "API Key 只能包含 ASCII 字符，且不能包含空格。")
        # Omit api_key to retain it. An explicit empty string deletes it.
        config["base_url"] = config["base_url"].rstrip("/")
        if type(config["context_tokens"]) is not int or not 4096 <= config["context_tokens"] <= 2_000_000:
            raise HTTPException(400, "上下文长度需为 4096–2000000 的整数。")
        if config["base_url"]:
            try:
                url = urlsplit(config["base_url"])
                valid = (url.hostname and not any(c.isspace() for c in config["base_url"])
                         and url.port != 0 and not url.username and not url.password
                         and not url.query and not url.fragment
                         and (url.scheme == "https" or (url.scheme == "http" and url.hostname in {"localhost", "127.0.0.1", "::1"})))
            except ValueError:
                valid = False
            if not valid:
                raise HTTPException(400, "网络 API 地址需使用 HTTPS（本机测试允许 HTTP），且不含账号、查询参数或片段。")
            if url.path.endswith("/chat/completions"):
                raise HTTPException(400, "请填写 Base URL，不要包含 /chat/completions。")
        # Never reuse a saved secret for a newly entered endpoint.
        if (config["base_url"] != current["base_url"] and current["api_key"]
                and "api_key" not in payload):
            raise HTTPException(400, "更换 API 地址时请重新填写 API Key。")
        if config["mode"] == "remote" and not all(config[f] for f in ("base_url", "model", "api_key")):
            raise HTTPException(400, "网络模式需要填写 API 地址、模型名称和 API Key。")
        return config

    def save(self, user_id, payload):
        # Serialize secret-file creation and database changes across concurrent saves.
        with self.accounts._connection(write=True) as connection:
            config = self.prepare(user_id, payload)
            stored = {**config, "api_key": self._cipher().encrypt(config["api_key"].encode()).decode() if config["api_key"] else ""}
            connection.execute("INSERT INTO model_settings (user_id, value) VALUES (?, ?) "
                               "ON CONFLICT(user_id) DO UPDATE SET value = excluded.value",
                               (user_id, json.dumps(stored)))
        return self.public(config)


def remote_environment(config):
    return dict(LLM_BACKEND="openai", LLM_BASE_URL=config["base_url"],
                LLM_API_KEY=config["api_key"], LLM_MODEL=config["model"],
                LLM_TOKENIZE="0", LLM_ALLOW_NO_KEY="0", LLM_TRUST_ENV="0",
                LLM_MAX_TOKENS=str(min(8192, config["context_tokens"] // 2)),
                LLM_CONTEXT_TOKENS=str(config["context_tokens"]))


async def test_connection(config):
    """One small paid request; verifies tool use rather than just listing models."""
    tool = {"type": "function", "function": {"name": "connection_check", "description": "Verify connection",
            "parameters": {"type": "object", "properties": {}}}}
    payload = dict(model=config["model"], max_tokens=128,
                   messages=[{"role": "user", "content": "Call connection_check now."}],
                   tools=[tool], tool_choice="auto", **provider_options(config["base_url"]))
    try:
        async with httpx.AsyncClient(timeout=12, trust_env=False) as client:
            response = await client.post(config["base_url"] + "/chat/completions", json=payload,
                                         headers={"Authorization": "Bearer " + config["api_key"]})
        if response.is_error:
            raise HTTPException(400, api_error(response, config["base_url"]))
        calls = response.json()["choices"][0]["message"].get("tool_calls") or []
        if not any(call.get("function", {}).get("name") == "connection_check" for call in calls):
            raise HTTPException(400, "接口已响应，但未返回工具调用；请选择支持 Tool Calling 的模型。")
    except httpx.HTTPError:
        raise HTTPException(400, "无法连接 API 或请求超时，请检查网络与 API 地址。") from None
    except httpx.InvalidURL:
        raise HTTPException(400, "API 地址格式不正确，请检查 Base URL。") from None
    except (ValueError, KeyError, IndexError, TypeError, AttributeError):
        raise HTTPException(400, "API 返回格式不兼容，请使用支持 Chat Completions 和 Tool Calling 的接口。") from None
    return {"ok": True, "message": "连接成功，模型支持工具调用。请保存配置后开始研究。"}
