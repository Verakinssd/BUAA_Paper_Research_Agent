"""Own one llama-server at a time; expose local control separately from inference."""
import argparse
import asyncio
from collections import deque
from contextlib import asynccontextmanager, suppress
import os
import json
from pathlib import Path
import shutil
import socket
import time

import httpx
from starlette.applications import Starlette
from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from deploy.serve_gguf import build_command


PROJECT = Path(__file__).resolve().parents[1]


def model_profiles(project=PROJECT, args=None):
    """Fixed IDs, local files only. Sizes refer to the documented bartowski files."""
    project = Path(project)
    defaults = dict(server_bin=os.getenv("LLAMA_SERVER_BIN", "llama-server"),
                    context=int(os.getenv("LLM_CONTEXT_TOKENS", "8192")),
                    parallel=int(os.getenv("LLAMA_PARALLEL", "1")), port=8000, gpu_layers=99)
    if args:
        defaults.update({key: getattr(args, key) for key in defaults})
    profiles = {}
    for quant, size in (("Q4_K_M", 17442399968), ("Q5_K_M", 20923877088)):
        filename = f"Qwen3.8-27B-{quant}.gguf"
        configured = os.getenv(f"GGUF_{quant[:2]}_MODEL_PATH")
        default_model = getattr(args, "model", None) or os.getenv("GGUF_MODEL_PATH")
        if not configured and default_model:
            default_quant = "Q5_K_M" if "Q5_K_M" in str(default_model) else "Q4_K_M"
            if quant == default_quant:
                configured = default_model
        path = Path(configured or project / "models" / filename).expanduser().resolve()
        options = dict(defaults, model=str(path))
        if quant.startswith("Q5"):
            options.update(parallel=int(os.getenv("LLAMA_Q5_PARALLEL", "1")),
                           gpu_layers=int(os.getenv("LLAMA_Q5_GPU_LAYERS", "99")))
        profiles[quant.lower()] = dict(id=quant.lower(), label=f"Qwen3.8-27B · {quant}",
            filename=filename, path=path, expected_bytes=size, args=argparse.Namespace(**options))
    return profiles


def file_state(profile):
    path = profile["path"]
    partial = path.with_suffix(path.suffix + ".part")
    try:
        if not path.is_file():
            size = partial.stat().st_size if partial.is_file() else 0
            return dict(available=False, file_status="downloading" if size else "missing",
                        downloaded_bytes=size)
        size = path.stat().st_size
        with path.open("rb") as stream:
            valid = stream.read(4) == b"GGUF"
        ready = valid and size == profile["expected_bytes"]
        return dict(available=ready, file_status="ready" if ready else "invalid", downloaded_bytes=size)
    except OSError:
        return dict(available=False, file_status="invalid", downloaded_bytes=0)


def public_profiles(profiles):
    return [dict(id=p["id"], label=p["label"], filename=p["filename"],
                 expected_bytes=p["expected_bytes"], parallel=p["args"].parallel,
                 context=p["args"].context, **file_state(p)) for p in profiles.values()]


class ModelController:
    def __init__(self, profiles, timeout=300):
        self.profiles = profiles
        self.timeout = timeout
        self.current_id = None
        self.target_id = None
        self.status = "idle"
        self.error = None
        self.process = None
        self.reader = None
        self.task = None
        self.logs = deque(maxlen=12)

    def snapshot(self):
        if self.status == "ready" and self.process and self.process.returncode is not None:
            self.status = "failed"
            self.error = "模型进程已退出，请重新加载模型。"
            self.current_id = None
        return dict(managed=True, status=self.status, current_id=self.current_id,
                    target_id=self.target_id, error=self.error,
                    models=public_profiles(self.profiles))

    def switch(self, model_id):
        if model_id not in self.profiles:
            raise HTTPException(400, "未知模型，请选择 Q4_K_M 或 Q5_K_M")
        if self.task and not self.task.done():
            raise HTTPException(409, "模型正在切换，请等待完成")
        if not file_state(self.profiles[model_id])["available"]:
            raise HTTPException(400, "模型未下载完成或文件大小/文件头不正确，请完成下载后刷新")
        # Validate binary, parameters and Slurm allocation BEFORE unloading anything.
        command = build_command(self.profiles[model_id]["args"])
        if not shutil.which(command[0]):
            raise HTTPException(400, "找不到配置的 llama-server 程序")
        if self.status == "ready" and model_id == self.current_id and self.process.returncode is None:
            return self.snapshot()
        previous = self.current_id if self.status == "ready" else None
        self.target_id, self.status, self.error = model_id, "unloading", None
        self.task = asyncio.create_task(self._switch(model_id, previous))
        return self.snapshot()

    async def _read_logs(self, process):
        async for line in process.stdout:
            text = line.decode("utf-8", errors="replace").rstrip()[:2000]
            self.logs.append(text)
            print(text, flush=True)

    async def _stop(self):
        process = self.process
        if process and process.returncode is None:
            with suppress(ProcessLookupError):
                process.terminate()
            try:
                await asyncio.wait_for(process.wait(), 15)
            except asyncio.TimeoutError:
                with suppress(ProcessLookupError):
                    process.kill()
                await process.wait()
        if self.reader:
            await asyncio.gather(self.reader, return_exceptions=True)
        self.process, self.reader, self.current_id = None, None, None

    async def _load(self, model_id):
        profile = self.profiles[model_id]
        command = build_command(profile["args"])
        # Never edit CUDA_VISIBLE_DEVICES: the launcher owns the Slurm allocation.
        self.logs.clear()
        self.process = await asyncio.create_subprocess_exec(*command, env=dict(os.environ),
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT, limit=512 * 1024)
        self.reader = asyncio.create_task(self._read_logs(self.process))
        base = f'http://127.0.0.1:{profile["args"].port}'
        deadline = time.monotonic() + self.timeout
        async with httpx.AsyncClient(trust_env=False, timeout=2) as client:
            while time.monotonic() < deadline:
                if self.process.returncode is not None:
                    raise RuntimeError("llama-server 加载失败：" + "\n".join(list(self.logs)[-4:]))
                try:
                    response = await client.get(base + "/props")
                    body = response.json() if response.status_code == 200 else {}
                    if body.get("model_path") and Path(body["model_path"]).resolve() == profile["path"]:
                        self.current_id = model_id
                        return
                except (httpx.HTTPError, ValueError):
                    pass
                await asyncio.sleep(.5)
        raise RuntimeError("模型加载超时，请检查模型服务终端和显存占用")

    async def _switch(self, model_id, previous):
        try:
            await self._stop()
            self.status = "loading"
            await self._load(model_id)
            self.status = "ready"
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.error = str(exc)
            await self._stop()
            if previous and previous != model_id:
                self.status = "recovering"
                try:
                    await self._load(previous)
                    self.status = "ready"
                    self.error = "切换失败，已恢复之前的模型。" + self.error
                except Exception as recovery:
                    await self._stop()
                    self.status = "failed"
                    self.error += "；原模型恢复失败：" + str(recovery)
            else:
                self.status = "failed"
        finally:
            self.target_id = None

    async def close(self):
        if self.task and not self.task.done():
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)
        await self._stop()
        self.status = "idle"


def create_control_app(args, initial_id, project=PROJECT):
    controller = ModelController(model_profiles(project, args))

    @asynccontextmanager
    async def lifespan(app):
        # An existing manual service must be stopped explicitly; never kill/adopt it.
        with socket.socket() as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind(("127.0.0.1", args.port))
            except OSError:
                raise RuntimeError(f"端口 {args.port} 已占用，请先在旧模型服务终端按 Ctrl+C，再启动 --managed") from None
        controller.switch(initial_id)
        try:
            yield
        finally:
            await controller.close()

    async def info(request):
        return JSONResponse(controller.snapshot())

    async def switch(request: Request):
        if request.headers.get("origin") or request.headers.get("sec-fetch-site") == "cross-site":
            raise HTTPException(403, "请从研究工作台切换模型")
        if "application/json" not in request.headers.get("content-type", ""):
            raise HTTPException(415, "请使用 application/json")
        body = await request.body()
        if len(body) > 2048:
            raise HTTPException(413, "请求内容过大")
        try:
            payload = json.loads(body)
        except (ValueError, UnicodeError):
            raise HTTPException(400, "JSON 格式不正确") from None
        if not isinstance(payload, dict) or not isinstance(payload.get("model_id"), str):
            raise HTTPException(400, "请提供 model_id")
        try:
            result = controller.switch(payload["model_id"])
        except (OSError, ValueError) as exc:
            raise HTTPException(400, str(exc)) from None
        return JSONResponse(result, status_code=202)

    async def error(request, exc):
        return JSONResponse({"error": str(exc.detail)}, status_code=exc.status_code)

    app = Starlette(lifespan=lifespan, exception_handlers={HTTPException: error},
                    routes=[Route("/models", info), Route("/switch", switch, methods=["POST"])])
    app.state.controller = controller
    return app


class ModelControlClient:
    """The web process does not own GPUs or inference subprocesses."""
    def __init__(self, project=PROJECT):
        self.project = project

    async def request(self, path, payload=None):
        base = os.getenv("MODEL_CONTROL_URL", "http://127.0.0.1:8001").rstrip("/")
        async with httpx.AsyncClient(trust_env=False, timeout=2) as client:
            response = await (client.get(base + path) if payload is None else
                              client.post(base + path, json=payload))
        data = response.json()
        if not isinstance(data, dict):
            raise ValueError("模型控制服务响应无效")
        if response.is_error:
            raise HTTPException(response.status_code, data.get("error", "模型控制请求失败"))
        if data.get("managed") is not True:
            raise ValueError("模型控制服务响应无效")
        return data

    async def catalog(self):
        try:
            return await self.request("/models")
        except (httpx.HTTPError, ValueError, HTTPException):
            return dict(managed=False, status="unmanaged", current_id=None, target_id=None,
                error=None, models=public_profiles(model_profiles(self.project)))

    async def switch(self, model_id):
        try:
            return await self.request("/switch", {"model_id": model_id})
        except (httpx.HTTPError, ValueError):
            raise HTTPException(503, "请先用 python deploy/serve_gguf.py --managed 启动可切换模型服务") from None
