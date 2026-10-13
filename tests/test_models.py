"""CPU-only process fixtures test switching, not GGUF inference or GPU capacity."""
import argparse
import asyncio
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

import httpx
from starlette.exceptions import HTTPException

from research.model_manager import ModelController, ModelControlClient, create_control_app, file_state, model_profiles
from web_app import create_app, now_iso


FAKE_SERVER = '''import argparse, json, os, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument("--model")
p.add_argument("--port", type=int)
p.add_argument("--parallel", type=int)
p.add_argument("--ctx-size", type=int)
a,_=p.parse_known_args()
print("FAKE CPU process fixture: no GGUF loading, no GPU inference", flush=True)
if Path(a.model + ".fail").exists():
    print("SIMULATED out of memory", flush=True)
    sys.exit(2)
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        body=json.dumps({"model_path":a.model,"mask":os.environ.get("CUDA_VISIBLE_DEVICES"),
                         "parallel":a.parallel,"ctx_size":a.ctx_size}).encode()
        self.send_response(200)
        self.send_header("Content-Length",str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def log_message(self,*args): pass
HTTPServer(("127.0.0.1",a.port),Handler).serve_forever()
'''


class ModelProcessTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name)
        self.binary = self.root / "fake llama-server"
        self.binary.write_text(f"#!{sys.executable}\n" + FAKE_SERVER, encoding="utf-8")
        self.binary.chmod(0o755)
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            self.port = probe.getsockname()[1]
        self.env = patch.dict(os.environ, {"SLURM_JOB_ID": "123", "CUDA_VISIBLE_DEVICES": "5,7",
            "GGUF_MODEL_PATH": "", "GGUF_Q4_MODEL_PATH": "", "GGUF_Q5_MODEL_PATH": "",
            "LLAMA_Q5_PARALLEL": "1", "LLAMA_Q5_GPU_LAYERS": "99"})
        self.env.start()
        args = argparse.Namespace(model=None, server_bin=str(self.binary), context=8192,
                                  parallel=2, port=self.port, gpu_layers=99)
        self.profiles = model_profiles(self.root, args)
        for profile in self.profiles.values():
            profile["path"].parent.mkdir(parents=True, exist_ok=True)
            # The fake executable does not parse weights. Never pass these files to llama.cpp.
            profile["path"].write_bytes(b"GGUF")
            profile["expected_bytes"] = 4
        self.controller = ModelController(self.profiles, timeout=5)

    async def asyncTearDown(self):
        await self.controller.close()
        self.env.stop()
        self.folder.cleanup()

    async def load(self, model_id):
        self.controller.switch(model_id)
        await asyncio.wait_for(asyncio.shield(self.controller.task), 10)
        return self.controller.snapshot()

    async def test_switch_reaps_old_process_and_preserves_scheduler_mask(self):
        self.assertEqual((await self.load("q4_k_m"))["status"], "ready")
        old = self.controller.process
        result = await self.load("q5_k_m")
        self.assertEqual(result["current_id"], "q5_k_m")
        self.assertIsNotNone(old.returncode)
        self.assertNotEqual(old.pid, self.controller.process.pid)
        async with httpx.AsyncClient(trust_env=False) as client:
            props = (await client.get(f"http://127.0.0.1:{self.port}/props")).json()
        self.assertEqual(props["mask"], "5,7")
        self.assertEqual(props["parallel"], 1)
        self.assertEqual(props["ctx_size"], 8192)
        self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "5,7")

    async def test_partial_or_invalid_file_never_unloads_working_model(self):
        await self.load("q4_k_m")
        old = self.controller.process
        profile = self.profiles["q5_k_m"]
        profile["path"].unlink()
        profile["path"].with_suffix(".gguf.part").write_bytes(b"GG")
        self.assertEqual(file_state(profile)["file_status"], "downloading")
        with self.assertRaises(HTTPException) as failure:
            self.controller.switch("q5_k_m")
        self.assertEqual(failure.exception.status_code, 400)
        profile["path"].write_bytes(b"GGUFtruncated")
        self.assertEqual(file_state(profile)["file_status"], "invalid")
        with self.assertRaises(HTTPException):
            self.controller.switch("q5_k_m")
        self.assertIs(self.controller.process, old)
        self.assertIsNone(old.returncode)

    async def test_failed_q5_restores_q4_and_reports_failure(self):
        await self.load("q4_k_m")
        Path(str(self.profiles["q5_k_m"]["path"]) + ".fail").touch()
        result = await self.load("q5_k_m")
        self.assertEqual(result["current_id"], "q4_k_m")
        self.assertEqual(result["status"], "ready")
        self.assertIn("已恢复", result["error"])
        self.assertIn("SIMULATED", result["error"])

    async def test_parallel_switches_rejected_and_same_model_is_idempotent(self):
        self.controller.switch("q4_k_m")
        with self.assertRaises(HTTPException) as failure:
            self.controller.switch("q5_k_m")
        self.assertEqual(failure.exception.status_code, 409)
        await self.controller.task
        process = self.controller.process
        self.controller.switch("q4_k_m")
        self.assertIs(self.controller.process, process)

    async def test_initial_failure_does_not_claim_ready(self):
        Path(str(self.profiles["q4_k_m"]["path"]) + ".fail").touch()
        result = await self.load("q4_k_m")
        self.assertEqual(result["status"], "failed")
        self.assertIsNone(result["current_id"])
        self.assertIsNone(self.controller.process)

    async def test_shutdown_cancels_loading_and_reaps_child(self):
        self.controller.switch("q4_k_m")
        for _ in range(100):
            if self.controller.process:
                break
            await asyncio.sleep(.01)
        process = self.controller.process
        self.assertIsNotNone(process)
        await self.controller.close()
        self.assertIsNotNone(process.returncode)
        self.assertEqual(self.controller.snapshot()["status"], "idle")

    async def test_lost_allocation_rejected_before_unloading(self):
        await self.load("q4_k_m")
        old = self.controller.process
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}):
            with self.assertRaisesRegex(ValueError, "先申请 GPU"):
                self.controller.switch("q5_k_m")
        self.assertIs(self.controller.process, old)
        self.assertIsNone(old.returncode)

    async def test_control_http_routes_load_and_validate_requests(self):
        args = self.profiles["q4_k_m"]["args"]
        with patch("research.model_manager.ModelController", return_value=self.controller):
            app = create_control_app(args, "q4_k_m", self.root)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://control") as client:
            self.assertTrue((await client.get("/models")).json()["managed"])
            for body in ([], {}, {"model_id": []}, {"model_id": "../model"}):
                self.assertEqual((await client.post("/switch", json=body)).status_code, 400)
            self.assertEqual((await client.post("/switch", json={"model_id": "q4_k_m"},
                                              headers={"Origin": "https://unrelated.test"})).status_code, 403)
            self.assertEqual((await client.post("/switch", content="{}")).status_code, 415)
            response = await client.post("/switch", json={"model_id": "q4_k_m"})
            self.assertEqual(response.status_code, 202)
            await self.controller.task
            state = (await client.get("/models")).json()
            self.assertEqual(state["current_id"], "q4_k_m")
            self.assertEqual(state["status"], "ready")


class ModelWebTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.app = create_app(output_root=Path(self.folder.name))
        self.manager = self.app.state.manager
        await self.manager.startup()
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url="http://testserver")
        self.catalog = dict(managed=True, status="ready", current_id="q4_k_m", target_id=None,
            models=[dict(id="q4_k_m", label="Qwen Q4", available=True),
                    dict(id="q5_k_m", label="Qwen Q5", available=True)])
        self.manager.model_control.catalog = AsyncMock(return_value=self.catalog)
        self.manager.model_control.switch = AsyncMock(return_value=dict(self.catalog, status="loading", target_id="q5_k_m"))

    async def asyncTearDown(self):
        await self.manager.shutdown()
        await self.client.aclose()
        self.folder.cleanup()

    async def test_switch_refused_during_research(self):
        self.manager.jobs["a" * 32] = dict(id="a" * 32, status="running", created_at=now_iso())
        self.assertTrue((await self.client.get("/api/models")).json()["blocked_by_job"])
        response = await self.client.post("/api/models/switch", json={"model_id": "q5_k_m"})
        self.assertEqual(response.status_code, 409)
        self.manager.model_control.switch.assert_not_awaited()

    async def test_switch_validation_and_same_origin(self):
        for value in ({}, [], {"model_id": "../../../file"}, {"model_id": True}, {"model_id": []}):
            response = await self.client.post("/api/models/switch", json=value)
            self.assertEqual(response.status_code, 400)
        response = await self.client.post("/api/models/switch", json={"model_id": "q5_k_m"}, headers={"Origin": "https://elsewhere.test"})
        self.assertEqual(response.status_code, 403)
        self.assertEqual((await self.client.post("/api/models/switch", content="{}")).status_code, 415)
        response = await self.client.post("/api/models/switch", json={"model_id": "q5_k_m"})
        self.assertEqual(response.status_code, 202)
        self.assertEqual(response.json()["target_id"], "q5_k_m")

    async def test_live_job_waits_for_ready_and_records_actual_model(self):
        self.catalog["status"] = "loading"
        response = await self.client.post("/api/jobs", json={"idea": "test", "top_k": 3})
        self.assertEqual(response.status_code, 409)
        self.catalog.update(status="ready", current_id="q5_k_m")
        with patch.object(self.manager, "execute", new_callable=AsyncMock):
            response = await self.client.post("/api/jobs", json={"idea": "test", "top_k": 3})
            self.assertEqual(response.status_code, 202)
            job = response.json()
            self.assertEqual(job["model_id"], "q5_k_m")
            saved = json.loads((self.manager.manifests / (job["id"] + ".json")).read_text())
            self.assertEqual(saved["model_label"], "Qwen Q5")

    async def test_control_client_ignores_proxy_and_keeps_unmanaged_compatible(self):
        controller = ModelControlClient(Path(self.folder.name))
        with patch("research.model_manager.httpx.AsyncClient") as client:
            client.return_value.__aenter__.return_value.get = AsyncMock(side_effect=httpx.ConnectError("offline"))
            catalog = await controller.catalog()
            self.assertFalse(catalog["managed"])
            self.assertFalse(client.call_args.kwargs["trust_env"])
            self.assertEqual(len(catalog["models"]), 2)


if __name__ == "__main__":
    unittest.main()
