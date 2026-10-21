import asyncio
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

import httpx

from research.api_settings import APISettings, remote_environment
from research.llm import LLM, api_error, provider_options
from research.task_agent import TaskAgent
from web_app import create_app


CONFIG = dict(mode="remote", base_url="https://provider.example/v1", model="tool-model",
              api_key="test-personal-secret-123", context_tokens=32768)


class PersonalAPITests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.app = create_app(output_root=self.root)
        self.manager = self.app.state.manager
        self.accounts = self.app.state.accounts
        self.settings = APISettings(self.accounts)
        self.user = self.accounts.create_user("api_user", "test-password")
        self.other = self.accounts.create_user("api_other", "test-password")
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url="http://testserver")
        self.client.cookies.set("paper_session", self.accounts.create_session(self.user["id"]))

    async def asyncTearDown(self):
        await self.manager.shutdown()
        await self.client.aclose()
        self.temp.cleanup()

    async def test_encrypted_persistence_user_isolation_and_clear(self):
        response = await self.client.put("/api/settings/model", json=CONFIG)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertTrue(response.json()["has_api_key"])
        self.assertNotIn(CONFIG["api_key"], response.text)
        self.assertNotIn(CONFIG["api_key"].encode(), self.accounts.db_path.read_bytes())
        restored = APISettings(self.accounts).read(self.user["id"])
        self.assertEqual(restored, CONFIG)
        self.assertEqual(self.settings.read(self.other["id"])["mode"], "local")
        response = await self.client.get("/api/settings/model?user_id=" + self.other["id"])
        self.assertEqual(response.json()["model"], CONFIG["model"])
        response = await self.client.put("/api/settings/model", json={"model": "another-model"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.settings.read(self.user["id"])["api_key"], CONFIG["api_key"])
        self.assertEqual((self.accounts.data_dir / "api_settings.key").stat().st_mode & 0o777, 0o600)
        response = await self.client.delete("/api/settings/model")
        self.assertEqual(response.json()["mode"], "local")
        self.assertFalse(response.json()["has_api_key"])
        self.assertEqual(self.settings.read(self.user["id"])["api_key"], "")

    async def test_validation_auth_and_endpoint_secret_reuse(self):
        for update in ({"mode": "unknown"}, {"context_tokens": True}, {"context_tokens": 0},
                       {"base_url": "file:///tmp/key"}, {"base_url": "https://host/v1?key=x"},
                       {"base_url": "https://user:password@host/v1"}, {"base_url": "http://external.example/v1"},
                       {"base_url": "https://host/v1/chat/completions"}, {"api_key": "bad\nkey"},
                       {"api_key": "中文密钥"}, {"api_key": "bad key"},
                       {"base_url": "https://bad host/v1"}, {"owner_id": self.other["id"]}):
            response = await self.client.put("/api/settings/model", json={**CONFIG, **update})
            self.assertEqual(response.status_code, 400, update)
        self.settings.save(self.user["id"], CONFIG)
        response = await self.client.put("/api/settings/model", json={"base_url": "https://different.example/v1"})
        self.assertEqual(response.status_code, 400)
        self.client.cookies.clear()
        for method, path in (("GET", "/api/settings/model"), ("PUT", "/api/settings/model"),
                             ("DELETE", "/api/settings/model"), ("POST", "/api/settings/model/test")):
            self.assertEqual((await self.client.request(method, path, json=CONFIG)).status_code, 401)

    async def test_probe_calls_tool_and_sanitizes_provider_errors(self):
        original = httpx.AsyncClient
        requests = []
        def handler(request):
            requests.append(request)
            return httpx.Response(200, json={"choices": [{"message": {"tool_calls": [
                {"function": {"name": "connection_check", "arguments": "{}"}}]}}]})
        def mock_client(**kwargs):
            return original(transport=httpx.MockTransport(handler), **kwargs)
        with patch("research.api_settings.httpx.AsyncClient", mock_client):
            response = await self.client.post("/api/settings/model/test", json=CONFIG)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0].headers["authorization"], "Bearer " + CONFIG["api_key"])
        self.assertEqual(json.loads(requests[0].content)["tool_choice"], "auto")
        self.assertEqual(self.settings.read(self.user["id"])["mode"], "local")  # Testing never saves.
        for status, body in ((401, {"error": CONFIG["api_key"]}), (200, {"choices": [{"message": {"content": "hello"}}]}),
                             (200, {"wrong": "format"})):
            with patch("research.api_settings.httpx.AsyncClient", lambda **kw: original(
                    transport=httpx.MockTransport(lambda req: httpx.Response(status, json=body)), **kw)):
                response = await self.client.post("/api/settings/model/test", json=CONFIG)
            self.assertEqual(response.status_code, 400)
            self.assertNotIn(CONFIG["api_key"], response.text)

    async def test_remote_job_uses_immutable_snapshot_and_never_persists_key(self):
        self.settings.save(self.user["id"], CONFIG)
        self.manager.health = AsyncMock(side_effect=AssertionError("local health must not be called"))
        self.manager.model_control.catalog = AsyncMock(side_effect=AssertionError("local model must not be required"))
        response = await self.client.get("/api/status")
        self.assertEqual(response.json()["model"]["mode"], "remote")
        with patch.object(self.manager, "execute", new_callable=AsyncMock):
            response = await self.client.post("/api/jobs", json={"idea": "remote research"})
            self.assertEqual(response.status_code, 202, response.text)
            job = self.manager.get(response.json()["id"])
            await self.manager.tasks[job["id"]]
        self.settings.save(self.user["id"], {"mode": "local", "api_key": ""})
        self.assertEqual(self.manager.job_configs[job["id"]], CONFIG)
        self.assertNotIn(CONFIG["api_key"], json.dumps(job))

        async def spawn(*args, **kwargs):
            env = kwargs["env"]
            self.assertEqual(env["LLM_API_KEY"], CONFIG["api_key"])
            self.assertEqual(env["LLM_BACKEND"], "openai")
            self.assertEqual(env["LLM_TOKENIZE"], "0")
            # Real child process exercises environment propagation and redaction.
            return await original_spawn(args[0], "-c", "import os; print(os.environ['LLM_API_KEY'])", **kwargs)
        original_spawn = asyncio.create_subprocess_exec
        with patch("web_app.asyncio.create_subprocess_exec", spawn):
            await self.manager.execute(job)
        self.assertNotIn(job["id"], self.manager.job_configs)
        self.assertIn("[redacted]", " ".join(job["logs"]))
        self.assertNotIn(CONFIG["api_key"], json.dumps(job))
        self.assertNotIn(CONFIG["api_key"], (self.manager.manifests / (job["id"] + ".json")).read_text())


class RemoteLLMTests(unittest.IsolatedAsyncioTestCase):
    async def test_free_router_falls_back_when_reasoning_is_mandatory(self):
        original = httpx.AsyncClient
        options = []
        def handler(request):
            options.append(json.loads(request.content)["reasoning"])
            if len(options) == 1:
                return httpx.Response(400, json={"error": {"message": "Reasoning is mandatory for this endpoint and cannot be disabled."}})
            return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": "complete"}}]})
        llm = LLM(remote_environment({**CONFIG, "base_url": "https://openrouter.ai/api/v1"}))
        with patch("research.llm.httpx.AsyncClient", lambda **kw: original(transport=httpx.MockTransport(handler), **kw)):
            self.assertEqual(await llm.text("system", "hello"), "complete")
        self.assertEqual(options, [{"enabled": False}, {"effort": "low"}])

    async def test_truncation_retries_once_with_larger_budget(self):
        original = httpx.AsyncClient
        budgets = []
        def handler(request):
            budgets.append(json.loads(request.content)["max_tokens"])
            return httpx.Response(200, json={"choices": [{"finish_reason": "length" if len(budgets) == 1 else "stop",
                "message": {"role": "assistant", "content": "complete"}}]})
        llm = LLM(remote_environment(CONFIG))
        with patch("research.llm.httpx.AsyncClient", lambda **kw: original(transport=httpx.MockTransport(handler), **kw)):
            self.assertEqual(await llm.text("system", "hello", max_tokens=384), "complete")
        self.assertEqual(budgets, [384, 8192])
        budgets.clear()
        def always_cutoff(request):
            budgets.append(json.loads(request.content)["max_tokens"])
            return httpx.Response(200, json={"choices": [{"finish_reason": "length", "message": {"content": "partial"}}]})
        with patch("research.llm.httpx.AsyncClient", lambda **kw: original(transport=httpx.MockTransport(always_cutoff), **kw)):
            with self.assertRaisesRegex(ValueError, "截断"):
                await llm.text("system", "hello", max_tokens=384)
        self.assertEqual(len(budgets), 2)
        # Chunk planning must reserve the largest retry, not just 2048 tokens.
        self.assertFalse(await llm.fits_text("system", "x" * 25000, max_tokens=2048))
        self.assertTrue(await llm.fits_text("system", "x" * 12000, max_tokens=2048))

    async def test_zhipu_options_and_safe_business_errors(self):
        base = "https://open.bigmodel.cn/api/paas/v4"
        self.assertEqual(provider_options(base), {"thinking": {"type": "disabled"}})
        self.assertEqual(provider_options("https://other.example/v1"), {})
        self.assertEqual(provider_options("https://openrouter.ai/api/v1"), {"reasoning": {"enabled": False}})
        for code in ("1113", "1302", "1305", "1308", "1310"):
            response = httpx.Response(429, json={"error": {"code": code, "message": CONFIG["api_key"]}})
            message = api_error(response, base)
            self.assertIn(code, message)
            self.assertNotIn(CONFIG["api_key"], message)
        message = api_error(httpx.Response(429, text="not json"), base)
        self.assertIn("429", message)

    async def test_remote_ignores_local_tokenizer_and_extensions_for_both_agents(self):
        requests = []
        def handler(request):
            requests.append(request)
            return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "ok"}}]})
        original = httpx.AsyncClient
        env = {**remote_environment(CONFIG), "LLM_TOP_K": "20", "LLM_ENABLE_THINKING": "0", "LLM_TOKENIZE": "1"}
        with patch.dict(os.environ, env), patch("research.llm.httpx.AsyncClient", lambda **kw: original(
                transport=httpx.MockTransport(handler), **kw)):
            for llm in (LLM(), TaskAgent().llm):
                self.assertEqual(await llm.text("system", "hello"), "ok")
                self.assertFalse(await llm.fits_text("system", "文" * 40000))
        self.assertEqual(len(requests), 2)
        for request in requests:
            self.assertEqual(request.url.path, "/v1/chat/completions")
            payload = json.loads(request.content)
            self.assertNotIn("top_k", payload)
            self.assertNotIn("chat_template_kwargs", payload)
            self.assertEqual(payload["model"], CONFIG["model"])


if __name__ == "__main__":
    unittest.main()
