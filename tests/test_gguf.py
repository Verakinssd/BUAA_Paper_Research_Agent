import argparse
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import httpx
from deploy.serve_gguf import build_command
from research.llm import LLM
from test_local import LOCAL_ENV


class GGUFLauncherTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.model = Path(self.folder.name) / "model with spaces.gguf"
        # Header fixture tests path validation only, never loaded by a model server.
        self.model.write_bytes(b"GGUF")
        self.args = argparse.Namespace(model=str(self.model), server_bin="/opt/llama-server",
                                       context=8192, parallel=1, gpu_layers=99, port=8000)

    def test_local_only_and_allocated_mask_preserved(self):
        with patch.dict(os.environ, {"SLURM_JOB_ID": "123", "CUDA_VISIBLE_DEVICES": "4"}):
            command = build_command(self.args)
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "4")
        self.assertEqual(command[command.index("--model") + 1], str(self.model))
        self.assertIn("--offline", command)
        self.assertNotIn("--hf-repo", command)
        self.assertEqual(command[command.index("--split-mode") + 1], "none")
        self.assertEqual(command[command.index("--parallel") + 1], "1")

    def test_parallel_slots_preserve_per_request_context(self):
        self.args.parallel = 2
        command = build_command(self.args)
        self.assertEqual(command[command.index("--parallel") + 1], "2")
        self.assertEqual(command[command.index("--ctx-size") + 1], "16384")

    def test_no_automatic_model_fallback(self):
        self.args.model = None
        with self.assertRaisesRegex(ValueError, "GGUF_MODEL_PATH"):
            build_command(self.args)

    def test_missing_or_invalid_file_rejected(self):
        self.args.model = str(self.model.with_name("missing.gguf"))
        with self.assertRaisesRegex(ValueError, "已下载"):
            build_command(self.args)
        self.args.model = str(self.model)
        self.model.write_bytes(b"HTML")
        with self.assertRaisesRegex(ValueError, "文件头"):
            build_command(self.args)

    def test_unallocated_slurm_gpu_rejected(self):
        with patch.dict(os.environ, {"SLURM_JOB_ID": "123", "CUDA_VISIBLE_DEVICES": ""}):
            with self.assertRaisesRegex(ValueError, "先申请 GPU"):
                build_command(self.args)


class LlamaCppClientTests(unittest.IsolatedAsyncioTestCase):
    async def invoke(self, handler, tools=None):
        original = httpx.AsyncClient
        def client(**kwargs):
            return original(transport=httpx.MockTransport(handler), **kwargs)
        env = {**LOCAL_ENV, "LLM_BACKEND": "llamacpp"}
        with patch.dict(os.environ, env), patch("research.llm.httpx.AsyncClient", new=client):
            return await LLM().chat([{"role": "user", "content": "hello"}], tools)

    async def test_template_and_tools_are_counted_before_generation(self):
        requests = []
        tools = [{"type": "function", "function": {"name": "search_papers", "parameters": {"type": "object"}}}]
        def handler(request):
            payload = json.loads(request.content) if request.content else None
            requests.append((request.url.path, payload))
            if request.url.path == "/props":
                return httpx.Response(200, json={"default_generation_settings": {"n_ctx": 8192}})
            if request.url.path == "/apply-template":
                self.assertEqual(payload["tools"], tools)
                self.assertEqual(payload["chat_template_kwargs"], {"enable_thinking": False})
                return httpx.Response(200, json={"prompt": "<formatted with tools>"})
            if request.url.path == "/tokenize":
                self.assertEqual(payload, {"content": "<formatted with tools>", "add_special": False, "parse_special": True})
                return httpx.Response(200, json={"tokens": [1, 2, 3]})
            self.assertEqual(payload["tools"], tools)
            self.assertEqual(payload["tool_choice"], "auto")
            return httpx.Response(200, json={"choices": [{"finish_reason": "tool_calls", "message": {
                "role": "assistant", "content": None, "tool_calls": [{"id": "call1", "type": "function",
                "function": {"name": "search_papers", "arguments": '{"query":"demo"}'}}]}}]})
        result = await self.invoke(handler, tools)
        self.assertEqual(result["tool_calls"][0]["function"]["name"], "search_papers")
        self.assertEqual([path for path, _ in requests], ["/props", "/apply-template", "/tokenize", "/v1/chat/completions"])

    async def test_server_context_smaller_than_env_blocks_request(self):
        def handler(request):
            if request.url.path == "/props":
                return httpx.Response(200, json={"default_generation_settings": {"n_ctx": 2048}})
            if request.url.path == "/apply-template":
                return httpx.Response(200, json={"prompt": "formatted"})
            self.assertEqual(request.url.path, "/tokenize")
            return httpx.Response(200, json={"tokens": [1] * 1000})
        with self.assertRaisesRegex(ValueError, "上下文"):
            await self.invoke(handler)

    async def test_missing_props_fails_closed(self):
        def handler(request):
            if request.url.path == "/props":
                return httpx.Response(200, json={})
            self.assertEqual(request.url.path, "/apply-template")
            return httpx.Response(200, json={"prompt": "formatted"})
        with self.assertRaisesRegex(RuntimeError, "上下文/模板响应无效"):
            await self.invoke(handler)

    async def test_unsupported_endpoint_reports_backend(self):
        def handler(request):
            self.assertEqual(request.url.path, "/props")
            return httpx.Response(404, json={})
        with self.assertRaisesRegex(RuntimeError, "llamacpp"):
            await self.invoke(handler)
