import json
import os
import unittest
from unittest.mock import patch
import httpx
from research.llm import LLM
from research.task_agent import TaskAgent

LOCAL_ENV = {
    "LLM_BACKEND": "vllm",
    "LLM_BASE_URL": "http://127.0.0.1:8000/v1", "LLM_API_KEY": "", "LLM_MODEL": "qwen-local",
    "LLM_ALLOW_NO_KEY": "1", "LLM_ENABLE_THINKING": "0", "LLM_MAX_TOKENS": "1024",
    "LLM_TOKENIZE": "1", "LLM_CONTEXT_TOKENS": "8192", "LLM_TRUST_ENV": "0",
}


class LocalLLMTests(unittest.IsolatedAsyncioTestCase):
    async def invoke(self, handler):
        original = httpx.AsyncClient
        def client(**kwargs):
            return original(transport=httpx.MockTransport(handler), **kwargs)
        with patch.dict(os.environ, LOCAL_ENV), patch("research.llm.httpx.AsyncClient", new=client):
            return await LLM().chat([{"role": "user", "content": "hello"}])

    async def test_local_options_sent_and_tokenize_checked(self):
        requests = []
        def handler(request):
            payload = json.loads(request.content)
            requests.append((request.url.path, payload))
            if request.url.path == "/tokenize":
                return httpx.Response(200, json={"count": 30, "max_model_len": 8192, "tokens": []})
            return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "ok"}}]})
        message = await self.invoke(handler)
        self.assertEqual(message["content"], "ok")
        self.assertEqual([p for p, _ in requests], ["/tokenize", "/v1/chat/completions"])
        self.assertEqual(requests[1][1]["chat_template_kwargs"], {"enable_thinking": False})
        self.assertEqual(requests[1][1]["max_tokens"], 1024)

    async def test_observer_receives_usage_and_defaults_to_off(self):
        body = {"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "ok"}}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 3},
                "timings": {"predicted_per_second": 42.0}}
        def handler(request):
            if request.url.path == "/tokenize":
                return httpx.Response(200, json={"count": 10, "max_model_len": 8192, "tokens": []})
            return httpx.Response(200, json=body)
        original = httpx.AsyncClient
        def client(**kwargs):
            return original(transport=httpx.MockTransport(handler), **kwargs)
        with patch.dict(os.environ, LOCAL_ENV), patch("research.llm.httpx.AsyncClient", new=client):
            llm = LLM()
            self.assertIsNone(llm.observer)
            await llm.chat([{"role": "user", "content": "hello"}])
            seen = []
            llm.observer = lambda *args: seen.append(args)
            await llm.chat([{"role": "user", "content": "hello"}])
            llm.observer = lambda *args: 1 / 0  # a broken observer must not break inference
            with self.assertLogs("research.llm", "ERROR"):
                result = await llm.chat([{"role": "user", "content": "hello"}])
            self.assertEqual(result["content"], "ok")
        self.assertEqual(len(seen), 1)
        elapsed, usage, timings, finish = seen[0]
        self.assertGreaterEqual(elapsed, 0)
        self.assertEqual(usage["completion_tokens"], 3)
        self.assertEqual(timings["predicted_per_second"], 42.0)
        self.assertEqual(finish, "stop")

    async def test_prompt_overflow_stops_before_generation(self):
        def handler(request):
            self.assertEqual(request.url.path, "/tokenize")
            return httpx.Response(200, json={"count": 8000, "max_model_len": 8192, "tokens": []})
        with self.assertRaisesRegex(ValueError, "上下文"):
            await self.invoke(handler)

    async def test_cutoff_output_not_accepted_as_complete(self):
        def handler(request):
            if request.url.path == "/tokenize":
                return httpx.Response(200, json={"count": 10, "max_model_len": 8192, "tokens": []})
            return httpx.Response(200, json={"choices": [{"finish_reason": "length", "message": {"role": "assistant", "content": '{"unfinished":'}}]})
        with self.assertRaisesRegex(ValueError, "长度上限"):
            await self.invoke(handler)

    async def test_adaptive_chunks_cover_entire_document(self):
        class SmallContextLLM:
            def __init__(self):
                self.inputs = []
                self.limits = []
                self.max_tokens = 2048
            async def fits_text(self, system, user, max_tokens=None):
                return len(user) < 2000
            async def text(self, system, user, max_tokens=None):
                self.inputs.append(user)
                self.limits.append(max_tokens)
                return "短笔记"
            async def merge_notes(self, system, label, notes, max_tokens=None, reduce_max_tokens=None):
                return await LLM.merge_notes(self, system, label, notes, max_tokens, reduce_max_tokens)
        agent = TaskAgent(llm=SmallContextLLM())
        markdown = "abcdefghijklmnop" * 1800
        await agent.summarize({"id": "P1", "title": "Document"}, markdown)
        fragments = [text.split("\n", 2)[2] for text in agent.llm.inputs if "\n片段 " in text]
        self.assertGreater(len(fragments), 1)
        self.assertEqual("".join(fragments), markdown)
        self.assertTrue(all(limit == 384 for text, limit in zip(agent.llm.inputs, agent.llm.limits)
                            if "\n片段 " in text))

    async def test_small_tail_is_merged_when_context_allows(self):
        class LargeContextLLM:
            def __init__(self):
                self.inputs = []
                self.max_tokens = 2048
            async def fits_text(self, system, user, max_tokens=None):
                return True
            async def text(self, system, user, max_tokens=None):
                self.inputs.append(user)
                return "短笔记"
            async def merge_notes(self, system, label, notes, max_tokens=None, reduce_max_tokens=None):
                return "完整总结"
        agent = TaskAgent(llm=LargeContextLLM())
        markdown = "x" * 49000
        await agent.summarize({"id": "P1", "title": "Document"}, markdown)
        fragments = [text.split("\n", 2)[2] for text in agent.llm.inputs if "\n片段 " in text]
        self.assertEqual([len(fragment) for fragment in fragments], [24000, 25000])
        self.assertEqual("".join(fragments), markdown)


if __name__ == "__main__":
    unittest.main()
