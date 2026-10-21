import asyncio
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

import httpx
from starlette.concurrency import run_in_threadpool

from web_app import create_app, JobManager, now_iso, PROJECT
from support import FakeTaskAgent, sample_papers, fake_download


class WebTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.app = create_app(output_root=self.root)
        self.manager = self.app.state.manager
        await self.manager.startup()
        self.real_health = self.manager.health
        self.manager.health = AsyncMock(return_value={"model": {"connected": True}})
        self.manager.model_control.catalog = AsyncMock(return_value={"managed": False})
        self.real_spawn = asyncio.create_subprocess_exec
        async def spawn_fixture(*command, **kwargs):
            command = [str(PROJECT / "tests/run_agent_fixture.py") if arg == str(PROJECT / "main_agent.py") else arg
                       for arg in command]
            return await self.real_spawn(*command, **kwargs)
        self.spawn_patch = patch("web_app.asyncio.create_subprocess_exec", new=spawn_fixture)
        self.spawn_patch.start()
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url="http://testserver")
        self.user = await run_in_threadpool(self.app.state.accounts.create_user,
                                           "web_test_admin", "test-password", role="admin")
        token = await run_in_threadpool(self.app.state.accounts.create_session, self.user["id"])
        self.client.cookies.set("paper_session", token)

    async def asyncTearDown(self):
        await self.manager.shutdown()
        await self.client.aclose()
        self.spawn_patch.stop()
        self.temp.cleanup()

    async def wait_job(self, job_id):
        task = self.manager.tasks.get(job_id)
        if task:
            await asyncio.wait_for(asyncio.shield(task), 30)
        response = await self.client.get(f"/api/jobs/{job_id}")
        self.assertEqual(response.status_code, 200)
        return response.json()

    async def test_cli_mcp_job_produces_downloadable_artifacts(self):
        response = await self.client.post("/api/jobs", json={"idea": "论文检索与总结", "top_k": 3, "concurrent": True})
        self.assertEqual(response.status_code, 202)
        job_id = response.json()["id"]
        snapshot = await self.wait_job(job_id)
        self.assertEqual(snapshot["status"], "completed", snapshot)
        self.assertNotIn("mock", snapshot)
        self.assertEqual(snapshot["completed_count"], 3)
        self.assertEqual(snapshot["metrics"]["requested_top_k"], 3)
        self.assertEqual(snapshot["stage"], "done")
        self.assertIn("TEST report", snapshot["report"])
        self.assertIn("report_completed", [e["event"] for e in snapshot["events"]])
        self.assertNotIn(str(PROJECT), json.dumps(snapshot))
        self.assertNotIn("fixture_text", json.dumps(snapshot))
        for paper in snapshot["papers"]:
            self.assertTrue(paper["pdf_available"])
            pdf = await self.client.get(paper["pdf_url"])
            self.assertEqual(pdf.status_code, 200)
            self.assertTrue(pdf.content.startswith(b"%PDF-"))
            self.assertTrue(paper["markdown_available"])
            fulltext = await self.client.get(f'/api/jobs/{job_id}/papers/{paper["id"]}/markdown')
            self.assertEqual(fulltext.status_code, 200)
            self.assertIn("attachment", fulltext.headers["content-disposition"])
            self.assertIn("text/markdown", fulltext.headers["content-type"])
            self.assertTrue(fulltext.text.strip())
            self.assertNotIn("TEST summary", fulltext.text)
        report = await self.client.get(f"/api/jobs/{job_id}/report")
        self.assertEqual(report.status_code, 200)
        self.assertIn("attachment", report.headers["content-disposition"])
        restored = JobManager(output_root=self.root)
        await restored.startup()
        self.assertEqual(restored.snapshot(restored.get(job_id))["completed_count"], 3)

    async def test_validation_and_origin_guards(self):
        for payload in [{}, {"idea": " "}, {"idea": "x", "top_k": True},
                        {"idea": "x", "top_k": 6}, {"idea": "x", "mock": "false"},
                        {"idea": "x" * 2001}, []]:
            response = await self.client.post("/api/jobs", json=payload)
            self.assertEqual(response.status_code, 400, payload)
            self.assertIn("error", response.json())
        response = await self.client.post("/api/jobs", json={"idea": "x"}, headers={"origin": "https://unrelated.test"})
        self.assertEqual(response.status_code, 403)
        response = await self.client.post("/api/jobs", content="{}")
        self.assertEqual(response.status_code, 415)
        self.assertEqual((await self.client.get("/api/jobs/not-an-id")).status_code, 404)

    async def test_removed_mode_parameter_cannot_create_a_job(self):
        for value in (True, False):
            response = await self.client.post("/api/jobs", json={"idea": "test", "mock": value})
            self.assertEqual(response.status_code, 400)
        self.assertFalse(self.manager.jobs)

    async def test_archived_demo_runs_are_hidden_but_files_are_preserved(self):
        job = dict(id="c" * 32, idea="archived test", top_k=3, mock=True, status="completed",
                   created_at=now_iso(), updated_at=now_iso(), logs=[])
        self.manager.persist(job)
        run = self.root / ("d" * 12)
        run.mkdir()
        metrics = run / "metrics.json"
        metrics.write_text(json.dumps({"mode": "mock", "task_success": True, "completed_count": 3}))
        await self.manager.startup()
        self.assertEqual((await self.client.get("/api/jobs")).json()["jobs"], [])
        self.assertEqual((await self.client.get(f'/api/jobs/{job["id"]}')).status_code, 404)
        self.assertEqual((await self.client.get(f'/api/jobs/{run.name}')).status_code, 404)
        self.assertTrue(metrics.is_file())
        self.assertTrue((self.manager.manifests / (job["id"] + ".json")).is_file())

    async def test_second_active_job_conflicts_and_queued_job_can_cancel(self):
        with patch.object(self.manager, "execute", new_callable=AsyncMock):
            first = await self.client.post("/api/jobs", json={"idea": "first"})
            second = await self.client.post("/api/jobs", json={"idea": "second"})
            self.assertEqual(second.status_code, 409)
            response = await self.client.post(f'/api/jobs/{first.json()["id"]}/cancel')
            self.assertEqual(response.json()["status"], "cancelled")
            self.assertIsNone(self.manager.active_id)

    async def test_running_child_is_stopped(self):
        response = await self.client.post("/api/jobs", json={"idea": "cancel"})
        job_id = response.json()["id"]
        for _ in range(100):
            process = self.manager.processes.get(job_id)
            if process:
                break
            await asyncio.sleep(0.01)
        self.assertIsNotNone(process)
        result = await self.client.post(f"/api/jobs/{job_id}/cancel")
        self.assertEqual(result.json()["status"], "cancelled")
        self.assertIsNotNone(process.returncode)
        self.assertNotIn(job_id, self.manager.processes)

    @unittest.skipUnless(Path("/proc/self").exists(), "process-tree assertion uses Linux /proc")
    async def test_cancel_closes_separate_mcp_session_and_preserves_gpu_mask(self):
        # Delay the test model after MCP discovery to inspect separate-session cleanup.
        with patch.dict(os.environ, {"TEST_AGENT_DELAY": "30", "LLM_BASE_URL": "http://127.0.0.1:1/v1",
                                     "LLM_MODEL": "fixture", "LLM_TOKENIZE": "0",
                                     "LLM_ALLOW_NO_KEY": "1", "LLM_TRUST_ENV": "0",
                                     "CUDA_VISIBLE_DEVICES": "5,7"}), \
                patch("web_app.asyncio.create_subprocess_exec", wraps=asyncio.create_subprocess_exec) as spawn:
            response = await self.client.post("/api/jobs", json={"idea": "cancel MCP"})
            job_id = response.json()["id"]
            for _ in range(200):
                if any("已通过 MCP" in line for line in self.manager.jobs[job_id]["logs"]):
                    break
                await asyncio.sleep(0.01)
            self.assertTrue(any("已通过 MCP" in line for line in self.manager.jobs[job_id]["logs"]))
            process = self.manager.processes[job_id]
            children_path = Path(f"/proc/{process.pid}/task/{process.pid}/children")
            child_pids = [int(pid) for pid in children_path.read_text().split()]
            self.assertTrue(child_pids)
            self.assertEqual(spawn.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"], "5,7")
            result = await self.client.post(f"/api/jobs/{job_id}/cancel")
            self.assertEqual(result.json()["status"], "cancelled")
            for pid in child_pids:
                self.assertFalse(Path(f"/proc/{pid}").exists(), f"MCP child {pid} survived cancellation")

    async def test_restart_marks_unfinished_job_failed(self):
        job = dict(id="a" * 32, idea="interrupted", top_k=3, concurrent=False,
                   status="running", created_at=now_iso(), updated_at=now_iso(), logs=[])
        self.manager.persist(job)
        restored = JobManager(output_root=self.root)
        await restored.startup()
        self.assertEqual(restored.get(job["id"])["status"], "failed")
        self.assertIn("中断", restored.get(job["id"])["error"])

    async def test_history_shortfall_is_partial_and_paths_cannot_escape(self):
        run = self.root / ("b" * 12)
        run.mkdir()
        (run / "metrics.json").write_text(json.dumps({"task_success": True, "mode": "live", "completed_count": 2, "selected_count": 2}))
        (run / "main_trace.json").write_text(json.dumps([{"tool": "build_report", "arguments": {"idea": "shortfall", "top_k": 3}}]))
        (run / "selection.json").write_text(json.dumps([{"id": "P1", "title": "A"}]))
        secret = self.root / "secret.txt"
        secret.write_text("do not expose")
        (run / "report.md").symlink_to(secret)
        response = await self.client.get("/api/jobs")
        job = response.json()["jobs"][0]
        self.assertNotIn("mock", job)
        self.assertEqual(job["status"], "partial")
        self.assertEqual((await self.client.get(f'/api/jobs/{job["id"]}/report')).status_code, 404)
        self.assertEqual((await self.client.get(f'/api/jobs/{job["id"]}/papers/P999/pdf')).status_code, 404)

    async def test_health_bypasses_proxy_for_local_service(self):
        response = httpx.Response(200, json={"data": []})
        with patch.dict(os.environ, {"LLM_BASE_URL": "http://127.0.0.1:8000/v1", "LLM_MODEL": "test-model"}), \
                patch("web_app.httpx.AsyncClient") as client:
            client.return_value.__aenter__.return_value.get = AsyncMock(return_value=response)
            health = await self.real_health()
            self.assertTrue(health["model"]["connected"])
            self.assertFalse(client.call_args.kwargs["trust_env"])
            self.assertLessEqual(client.call_args.kwargs["timeout"], 2)
            await self.real_health()
            self.assertEqual(client.call_count, 1)

    async def test_redaction_removes_keys_credentials_and_local_paths(self):
        with patch.dict(os.environ, {"LLM_API_KEY": "a-very-private-token"}):
            value = self.manager.clean("a-very-private-token http://user:pass@proxy.test /private/location/.env")
        self.assertNotIn("a-very-private-token", value)
        self.assertNotIn("user:pass", value)
        self.assertNotIn("/private/location", value)


class TopKRegressionTests(unittest.IsolatedAsyncioTestCase):
    async def test_two_candidates_cannot_fulfill_requested_top_three(self):
        import server
        fixture = sample_papers(2)
        sources = {"arxiv": {"status": "ok"}, "openalex": {"status": "ok"}, "ranking_method": "bm25"}
        with tempfile.TemporaryDirectory() as folder, \
                patch.dict(os.environ, {"PAPER_OUTPUT_ROOT": folder, "PAPER_CACHE_DIR": folder + "/cache"}), \
                patch.dict(server.config, {"concurrent": False}), \
                patch("server.retrieve", new=AsyncMock(return_value=(fixture, sources))), \
                patch("server.TaskAgent", FakeTaskAgent), \
                patch("research.pipeline.download_pdf", fake_download):
            found = await server.search_papers("test")
            result = await server.build_report(found["run_id"], "test", 3)
            self.assertEqual(result["metrics"]["completed_count"], 2)
            self.assertEqual(result["metrics"]["requested_top_k"], 3)
            self.assertFalse(result["metrics"]["task_success"])
            result = await server.build_report(found["run_id"], "test", 2)
            self.assertTrue(result["metrics"]["task_success"])
        server.state.pop(found["run_id"], None)


if __name__ == "__main__":
    unittest.main()
