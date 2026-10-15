"""Local web workspace for the existing CLI/MCP research pipeline."""
import argparse
import asyncio
from collections import deque
from contextlib import asynccontextmanager, suppress
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import signal
import sys
import time
from urllib.parse import urlsplit
import uuid

import httpx
from dotenv import dotenv_values, load_dotenv
from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles
from research.model_manager import ModelControlClient
from research.accounts import SESSION_TTL_SECONDS, UserStore


PROJECT = Path(__file__).resolve().parent
ID_PATTERN = re.compile(r"[a-f0-9]{12}|[a-f0-9]{32}")
PAPER_PATTERN = re.compile(r"P[0-9]{1,5}")
ACTIVE = {"queued", "running"}
FINAL = {"completed", "partial", "failed", "cancelled"}


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def read_json(path, default=None):
    try:
        if path.stat().st_size > 8 * 1024 * 1024:
            return default
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError):
        return default


def read_text(path, limit=2 * 1024 * 1024):
    try:
        with path.open(encoding="utf-8") as stream:
            return stream.read(limit)
    except (OSError, UnicodeError):
        return ""


def contained(path, root):
    """Reject symlinks escaping the designated artifact directory."""
    return path.resolve().is_relative_to(root.resolve())


class JobManager:
    def __init__(self, project=PROJECT, output_root=None):
        self.project = Path(project).resolve()
        self.output_root = Path(output_root or self.project / "outputs").resolve()
        self.manifests = self.output_root / "_web" / "jobs"
        self.runs = self.output_root / "_web" / "runs"
        self.jobs = {}
        self.tasks = {}
        self.processes = {}
        self.lock = asyncio.Lock()
        self.health_lock = asyncio.Lock()
        self.health_cache = (0, {})
        self.model_control = ModelControlClient(self.project)

    def clean(self, value):
        secrets = [secret for name, secret in os.environ.items()
                   if len(secret) >= 8 and re.search(r"(?:KEY|TOKEN|PASSWORD|SECRET)$", name, re.I)]

        def redact(item):
            if isinstance(item, dict):
                return {k: redact(v) for k, v in item.items()
                        if k not in {"output_dir", "markdown_path", "summary_path", "fixture_text"}}
            if isinstance(item, list):
                return [redact(v) for v in item]
            if not isinstance(item, str):
                return item
            if re.fullmatch(r"/api/jobs/[a-f0-9]{12,32}/papers/P[0-9]{1,5}/pdf", item):
                return item
            item = item.replace(str(self.project), "[project]")
            for secret in secrets:
                item = item.replace(secret, "[redacted]")
            item = re.sub(r"(https?://)[^\s/@]+:[^\s/@]+@", r"\1[redacted]@", item)
            item = re.sub(r"(?i)(authorization\s*[:=]\s*(?:bearer\s+)?)[^\s,\"}]+", r"\1[redacted]", item)
            return re.sub(r"(?<![\w:/])/(?:[^\s\"'<>/]+/)+[^\s\"'<>]*", "[local path]", item)

        return redact(value)

    def persist(self, job):
        self.manifests.mkdir(parents=True, exist_ok=True)
        target = self.manifests / (job["id"] + ".json")
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(job, ensure_ascii=False), encoding="utf-8")
        temporary.replace(target)

    async def startup(self):
        self.manifests.mkdir(parents=True, exist_ok=True)
        for path in self.manifests.glob("*.json"):
            if not ID_PATTERN.fullmatch(path.stem) or not contained(path, self.manifests):
                continue
            job = read_json(path)
            if not isinstance(job, dict) or job.get("id") != path.stem:
                continue
            if job.get("mock"):
                continue  # Archived test runs are not research history.
            if job.get("status") in ACTIVE:
                job.update(status="failed", updated_at=now_iso(), finished_at=now_iso(),
                           error="网页服务已重启，之前的任务已中断；请重新运行。")
                self.persist(job)
            self.jobs[job["id"]] = job

    async def shutdown(self):
        for job_id in list(self.tasks):
            if self.jobs[job_id]["status"] in ACTIVE:
                await self.cancel(job_id)
        if self.tasks:
            await asyncio.gather(*list(self.tasks.values()), return_exceptions=True)

    @property
    def active_id(self):
        return next((key for key, job in self.jobs.items() if job["status"] in ACTIVE or
                     (key in self.tasks and not self.tasks[key].done())), None)

    def run_directory(self, job):
        if job.get("legacy"):
            path = self.output_root / job["id"]
            return path if contained(path, self.output_root) else None
        root = self.runs / job["id"]
        if not contained(root, self.runs) or not root.exists():
            return None
        children = [p for p in root.iterdir() if p.is_dir() and ID_PATTERN.fullmatch(p.name)
                    and contained(p, root)]
        # The main Agent may retry searches. A report-bearing run takes priority.
        return max(children, key=lambda p: ((p / "selection.json").exists(), p.stat().st_mtime),
                   default=None)

    def artifact(self, run, name):
        path = run / name
        return path if contained(path, run) else run / "__unavailable__"

    def discover_history(self):
        if not self.output_root.exists():
            return
        for path in self.output_root.iterdir():
            if (not path.is_dir() or not ID_PATTERN.fullmatch(path.name)
                    or path.name in self.jobs or not contained(path, self.output_root)):
                continue
            metrics = read_json(self.artifact(path, "metrics.json"), {})
            meta = read_json(self.artifact(path, "run_meta.json"), {})
            selections = read_json(self.artifact(path, "selection.json"), [])
            if not metrics or not isinstance(metrics, dict) or metrics.get("mode") == "mock":
                continue
            trace = read_json(self.artifact(path, "main_trace.json"), [])
            build_args = next((item.get("arguments", {}) for item in trace
                               if isinstance(item, dict) and item.get("tool") == "build_report"), {})
            idea = (meta.get("idea") or build_args.get("idea") or
                    next((p.get("research_idea") for p in selections if p.get("research_idea")),
                         "历史研究任务"))
            requested = meta.get("requested_top_k") or build_args.get("top_k") or metrics.get("requested_top_k")
            requested = requested or metrics.get("selected_count", 3)
            timestamp = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()
            completed = metrics.get("completed_count", 0)
            status = "completed" if metrics.get("task_success") and completed >= requested else "partial"
            if not completed:
                status = "failed"
            self.jobs[path.name] = dict(id=path.name, idea=idea, top_k=requested,
                concurrent=bool(metrics.get("concurrent")),
                status=status, created_at=timestamp, updated_at=timestamp, finished_at=timestamp,
                logs=[], legacy=True, model_id=meta.get("model_id"), model_label=meta.get("model_label"))

    def get(self, job_id):
        if not ID_PATTERN.fullmatch(job_id):
            raise HTTPException(404, "任务不存在")
        self.discover_history()
        if job_id not in self.jobs:
            raise HTTPException(404, "任务不存在")
        return self.jobs[job_id]

    def snapshot(self, job):
        result = {key: job.get(key) for key in
                  ("id", "idea", "top_k", "concurrent", "status", "created_at", "updated_at", "model_id", "model_label")}
        result.update(stage="search", run_id=None, candidates=[], papers=[], metrics={},
                      report="", events=[], completed_count=0, selected_count=0,
                      logs=list(job.get("logs", []))[-100:])
        if job.get("error"):
            result["error"] = job["error"]
        end = datetime.fromisoformat(job.get("finished_at") or now_iso())
        result["elapsed_seconds"] = max(0, round((end - datetime.fromisoformat(job["created_at"])).total_seconds(), 1))
        run = self.run_directory(job)
        if run:
            result["run_id"] = run.name
            candidates = read_json(self.artifact(run, "candidates.json"), [])
            result["candidates"] = [{key: p.get(key) for key in
                ("id", "title", "abstract", "year", "authors", "url", "sources", "basic_score")}
                for p in candidates if isinstance(p, dict)]
            selected = read_json(self.artifact(run, "selection.json"), [])
            topk = read_json(self.artifact(run, "topk.json"), [])
            metrics = read_json(self.artifact(run, "metrics.json"), {})
            events = []
            for line in read_text(self.artifact(run, "trace.jsonl")).splitlines():
                try:
                    row = json.loads(line)
                    if isinstance(row, dict):
                        events.append(row)
                except ValueError:
                    continue  # A concurrent trace append may not yet be complete.
            result["events"] = events[-250:]
            if events:
                result["updated_at"] = datetime.fromtimestamp(events[-1].get("time", time.time()), timezone.utc).isoformat()
            by_id = {p.get("id"): dict(p) for p in selected if isinstance(p, dict)}
            for event in events:
                pid = event.get("id")
                if pid not in by_id:
                    continue
                if event.get("event") == "paper_started":
                    by_id[pid]["status"] = "running"
                elif event.get("event") == "paper_processed":
                    by_id[pid].update(event)
            for paper in topk:
                if paper.get("id") in by_id:
                    by_id[paper["id"]].update(paper)
            for paper in by_id.values():
                pid = paper["id"]
                if not PAPER_PATTERN.fullmatch(pid):
                    continue
                item = {key: paper.get(key) for key in
                    ("id", "title", "url", "year", "authors", "sources", "selection_reason", "error")}
                item["status"] = paper.get("status", "pending")
                if job["status"] in {"cancelled", "failed"} and item["status"] in {"pending", "running"}:
                    item["status"] = job["status"]
                item["summary"] = paper.get("summary") or read_text(self.artifact(run, f"summaries/{pid}.md"))
                pdf = self.artifact(run, f"fulltext/{pid}.pdf")
                item["pdf_available"] = pdf.is_file()
                item["pdf_url"] = (f'/api/jobs/{job["id"]}/papers/{pid}/pdf'
                                   if pdf.is_file() else None)
                result["papers"].append(item)
            result["metrics"] = metrics
            result["report"] = read_text(self.artifact(run, "report.md"))
            result["selected_count"] = len(result["papers"])
            result["completed_count"] = sum(p["status"] == "ok" for p in result["papers"])
            names = {event.get("event") for event in events}
            if "report_completed" in names:
                result["stage"] = "done"
            elif "report_started" in names or topk:
                result["stage"] = "report"
            elif "task_agent_selected" in names or selected:
                result["stage"] = "read"
            elif "sources_searched" in names:
                result["stage"] = "select"
            if job.get("legacy"):
                result["elapsed_seconds"] = metrics.get("server_total_seconds", 0)
        if job["status"] in {"completed", "partial"}:
            result["stage"] = "done"
        return self.clean(result)

    async def create(self, payload, owner_id=None):
        if not isinstance(payload, dict):
            raise HTTPException(400, "请求内容必须是 JSON 对象")
        idea, top_k = payload.get("idea"), payload.get("top_k", 3)
        if not isinstance(idea, str) or not 1 <= len(idea.strip()) <= 2000:
            raise HTTPException(400, "研究想法需为 1–2000 个字符")
        if type(top_k) is not int or not 1 <= top_k <= 5:
            raise HTTPException(400, "Top-K 需为 1–5 的整数")
        if set(payload) - {"idea", "top_k", "concurrent"}:
            raise HTTPException(400, "请求包含不支持的参数")
        if type(payload.get("concurrent", False)) is not bool:
            raise HTTPException(400, "concurrent 需为布尔值")
        async with self.lock:
            if self.active_id:
                raise HTTPException(409, "已有任务正在运行，请等待完成或停止后再试")
            selected_model = {}
            catalog = await self.model_control.catalog()
            if catalog["managed"]:
                if catalog["status"] != "ready":
                    raise HTTPException(409, "模型尚未就绪，请等待加载完成后再开始研究")
                selected_model = next((m for m in catalog["models"] if m["id"] == catalog["current_id"]), {})
            if not (await self.health())["model"]["connected"]:
                raise HTTPException(409, "模型未连接，请先启动模型服务后再开始研究")
            stamp = now_iso()
            job = dict(id=uuid.uuid4().hex, idea=idea.strip(), top_k=top_k,
                       owner_id=owner_id,
                       concurrent=payload.get("concurrent", False),
                       status="queued", created_at=stamp, updated_at=stamp, logs=[],
                       model_id=selected_model.get("id"), model_label=selected_model.get("label"))
            self.jobs[job["id"]] = job
            self.persist(job)
            self.tasks[job["id"]] = asyncio.create_task(self.execute(job))
            return self.snapshot(job)

    async def execute(self, job):
        process = None
        try:
            if job["status"] == "cancelled":
                return
            job.update(status="running", updated_at=now_iso())
            self.persist(job)
            env = dict(os.environ)  # Keep the scheduler's GPU allocation unchanged.
            env.update(PAPER_OUTPUT_ROOT=str(self.runs / job["id"]), PAPER_WEB_JOB_ID=job["id"],
                       PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
            if job.get("model_id"):
                env.update(PAPER_MODEL_ID=job["model_id"], PAPER_MODEL_LABEL=job["model_label"])
            command = [sys.executable, "-u", str(self.project / "main_agent.py"),
                       "--idea", job["idea"], "--top-k", str(job["top_k"])]
            if job["concurrent"]:
                command.append("--concurrent")
            process = await asyncio.create_subprocess_exec(*command, cwd=self.project, env=env,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
                start_new_session=os.name == "posix", limit=512 * 1024)
            self.processes[job["id"]] = process
            if job["status"] == "cancelled":
                await self.stop_process(process)
            async for line in process.stdout:
                text = self.clean(line.decode("utf-8", errors="replace").rstrip())[:4000]
                if text:
                    job["logs"].append(text)
                    job["logs"] = job["logs"][-100:]
                    job["updated_at"] = now_iso()
            code = await process.wait()
            if job["status"] != "cancelled":
                snapshot = self.snapshot(job)
                metrics = snapshot["metrics"]
                if code == 0 and metrics.get("task_success") and snapshot["completed_count"] >= job["top_k"]:
                    job["status"] = "completed"
                elif snapshot["report"] or snapshot["completed_count"]:
                    job["status"] = "partial"
                    job["error"] = "部分环节未完成，请查看来源状态、论文详情和日志。"
                else:
                    job["status"] = "failed"
                    job["error"] = "任务未完成，请查看运行日志并检查模型服务及论文来源连接。"
        except asyncio.CancelledError:
            job.update(status="cancelled", error="任务已停止")
            if process:
                await self.stop_process(process)
        except Exception as exc:
            job.update(status="failed", error="启动或执行失败：" + self.clean(str(exc)))
            if process and process.returncode is None:
                await self.stop_process(process)
        finally:
            self.processes.pop(job["id"], None)
            job.update(updated_at=now_iso(), finished_at=now_iso())
            self.persist(job)
            self.tasks.pop(job["id"], None)

    async def stop_process(self, process):
        if process.returncode is not None:
            return
        if os.name == "posix":
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=8)
        except asyncio.TimeoutError:
            if os.name == "posix":
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
            await process.wait()

    async def cancel(self, job_id):
        job = self.get(job_id)
        if job["status"] in ACTIVE:
            job.update(status="cancelled", error="任务已停止", updated_at=now_iso(), finished_at=now_iso())
            self.persist(job)
            if job_id in self.processes:
                await self.stop_process(self.processes[job_id])
            task = self.tasks.get(job_id)
            if task:
                await asyncio.shield(task)
        return self.snapshot(job)

    async def health(self):
        async with self.health_lock:
            if time.monotonic() - self.health_cache[0] < 5:
                return {"model": self.health_cache[1], "active_job_id": self.active_id}
            name = os.getenv("LLM_MODEL", "未配置模型")
            base = os.getenv("LLM_BASE_URL", "").rstrip("/")
            model = dict(name=name, connected=False, detail="请配置并启动模型服务")
            if base:
                try:
                    local = urlsplit(base).hostname in {"127.0.0.1", "localhost", "::1"}
                    key = os.getenv("LLM_API_KEY")
                    async with asyncio.timeout(1.8):
                        async with httpx.AsyncClient(timeout=1.5, trust_env=not local and os.getenv("LLM_TRUST_ENV", "1") == "1") as client:
                            response = await client.get(base + "/models", headers={"Authorization": f"Bearer {key}"} if key else {})
                    if response.status_code == 200:
                        model.update(connected=True, detail="模型服务已连接")
                    else:
                        model["detail"] = f"模型服务返回 HTTP {response.status_code}"
                except (httpx.HTTPError, ValueError, TimeoutError):
                    model["detail"] = "模型服务未连接；请确认服务已启动"
            self.health_cache = (time.monotonic(), self.clean(model))
            return {"model": self.health_cache[1], "active_job_id": self.active_id}


def create_app(project=PROJECT, output_root=None, data_dir=None):
    manager = JobManager(project, output_root)
    configured_data = os.getenv("PAPER_DATA_DIR") or dotenv_values(manager.project / ".env").get("PAPER_DATA_DIR")
    if data_dir is None:
        # Explicit test/output workspaces never read or write the real account database.
        data_dir = (manager.output_root / "_accounts" if output_root is not None else
                    Path(configured_data) if configured_data else manager.project / "data")
    account_path = Path(data_dir).expanduser()
    if not account_path.is_absolute():
        account_path = manager.project / account_path
    accounts = UserStore(account_path)
    cookie_name = "paper_session"
    auth_attempts = {}

    @asynccontextmanager
    async def lifespan(app):
        load_dotenv(manager.project / ".env")
        await run_in_threadpool(accounts.initialize)
        await manager.startup()
        try:
            yield
        finally:
            await manager.shutdown()

    async def error_response(request, exc):
        return JSONResponse({"error": manager.clean(str(exc.detail))}, status_code=exc.status_code)

    async def home(request):
        path = manager.project / "web" / "index.html"
        if not path.is_file():
            raise HTTPException(503, "前端文件未安装")
        return FileResponse(path)

    async def status(request):
        result = dict(await manager.health())
        active_id = manager.active_id
        result["resource_busy"] = bool(active_id)
        result["active_job_id"] = (active_id if active_id and
                                   visible(manager.jobs[active_id], request.state.user) else None)
        return JSONResponse(result)

    def visible(job, user):
        owner = job.get("owner_id")
        return owner == user["id"] or (not owner and user["role"] == "admin")

    def owned_job(request):
        job = manager.get(request.path_params["job_id"])
        if not visible(job, request.state.user):
            raise HTTPException(404, "任务不存在")
        return job

    def admin_only(request):
        if request.state.user["role"] != "admin":
            raise HTTPException(403, "仅管理员可执行此操作")

    async def bounded_body(request, limit):
        body = bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > limit:
                raise HTTPException(413, "请求内容过大")
        return bytes(body)

    async def payload(request, limit=20000):
        if "application/json" not in request.headers.get("content-type", ""):
            raise HTTPException(415, "请使用 application/json")
        body = await bounded_body(request, limit)
        try:
            return json.loads(body)
        except (ValueError, UnicodeError):
            raise HTTPException(400, "JSON 格式不正确") from None

    async def object_payload(request, allowed, required=()):
        data = await payload(request)
        if not isinstance(data, dict) or set(data) - set(allowed) or set(required) - set(data):
            raise HTTPException(400, "请求参数不正确")
        return data

    def throttle(request, category):
        # Bounded, process-local limits. Never trust a caller-supplied forwarding header.
        current = time.monotonic()
        expired = [key for key, values in auth_attempts.items() if not values or values[-1] < current - 300]
        for key in expired:
            auth_attempts.pop(key, None)
        key = (category, request.client.host if request.client else "unknown")
        if key not in auth_attempts and len(auth_attempts) >= 4096:
            raise HTTPException(429, "请求过于频繁，请稍后重试")
        attempts = auth_attempts.setdefault(key, deque())
        while attempts and attempts[0] < current - 300:
            attempts.popleft()
        if len(attempts) >= (10 if category == "register" else 30):
            raise HTTPException(429, "尝试次数过多，请五分钟后重试")
        attempts.append(current)

    def with_session(user, token, status_code=200):
        response = JSONResponse({"user": user}, status_code=status_code)
        response.set_cookie(cookie_name, token, max_age=SESSION_TTL_SECONDS,
                            httponly=True, samesite="lax", path="/",
                            secure=os.getenv("PAPER_COOKIE_SECURE", "0") == "1")
        return response

    async def me(request):
        return JSONResponse({"user": request.state.user})

    async def register(request):
        throttle(request, "register")
        data = await object_payload(request, {"username", "password", "display_name"}, {"username", "password"})
        user = await run_in_threadpool(accounts.create_user, data["username"], data["password"],
                                       display_name=data.get("display_name"), role="user")
        token = await run_in_threadpool(accounts.create_session, user["id"])
        return with_session(user, token, 201)

    async def login(request):
        throttle(request, "login")
        data = await object_payload(request, {"username", "password"}, {"username", "password"})
        user = await run_in_threadpool(accounts.authenticate, data["username"], data["password"])
        token = await run_in_threadpool(accounts.create_session, user["id"])
        return with_session(user, token)

    async def logout(request):
        token = request.cookies.get(cookie_name)
        if token:
            await run_in_threadpool(accounts.revoke_session, token)
        response = JSONResponse({"ok": True})
        response.delete_cookie(cookie_name, path="/")
        return response

    async def profile(request):
        data = await object_payload(request, {"username", "display_name", "bio"})
        user = await run_in_threadpool(accounts.update_profile, request.state.user["id"], data)
        return JSONResponse({"user": user})

    async def password(request):
        throttle(request, "password")
        data = await object_payload(request, {"current_password", "new_password"}, {"current_password", "new_password"})
        user_id = request.state.user["id"]
        await run_in_threadpool(accounts.change_password, user_id, data["current_password"], data["new_password"])
        token = await run_in_threadpool(accounts.create_session, user_id)
        return with_session(await run_in_threadpool(accounts.profile, user_id), token)

    async def avatar(request):
        if request.method == "DELETE":
            user = await run_in_threadpool(accounts.delete_avatar, request.state.user["id"])
        else:
            raw = await bounded_body(request, 3 * 1024 * 1024)
            user = await run_in_threadpool(accounts.set_avatar, request.state.user["id"], raw,
                                           request.headers.get("content-type", ""))
        return JSONResponse({"user": user})

    async def avatar_file(request):
        path = await run_in_threadpool(accounts.avatar_path, request.path_params["user_id"])
        return FileResponse(path, media_type="image/png")

    async def admin_users(request):
        admin_only(request)
        try:
            page = int(request.query_params.get("page", "1"))
            page_size = int(request.query_params.get("page_size", "20"))
        except ValueError:
            raise HTTPException(400, "分页参数需为整数") from None
        result = await run_in_threadpool(accounts.admin_list, search=request.query_params.get("search", ""),
                                         page=page, page_size=page_size)
        return JSONResponse(result)

    async def admin_update(request):
        admin_only(request)
        data = await object_payload(request, {"username", "display_name", "bio", "role", "is_active"})
        user = await run_in_threadpool(accounts.admin_update, request.state.user["id"],
                                       request.path_params["user_id"], data)
        return JSONResponse({"user": user})

    async def admin_password(request):
        admin_only(request)
        data = await object_payload(request, {"new_password"}, {"new_password"})
        await run_in_threadpool(accounts.reset_password, request.state.user["id"],
                                 request.path_params["user_id"], data["new_password"])
        return JSONResponse({"ok": True})

    async def models(request):
        result = await manager.model_control.catalog()
        result["blocked_by_job"] = bool(manager.active_id)
        return JSONResponse(manager.clean(result))

    async def switch_model(request):
        admin_only(request)
        data = await payload(request, 2048)
        if (not isinstance(data, dict) or not isinstance(data.get("model_id"), str)
                or data["model_id"] not in {"q4_k_m", "q5_k_m"}):
            raise HTTPException(400, "请选择 Q4_K_M 或 Q5_K_M")
        async with manager.lock:
            if manager.active_id:
                raise HTTPException(409, "研究任务正在运行，请等待完成或停止任务后再切换模型")
            result = await manager.model_control.switch(data["model_id"])
            manager.health_cache = (0, {})
        return JSONResponse(manager.clean(result), status_code=202)

    async def jobs(request):
        if request.method == "POST":
            return JSONResponse(await manager.create(await payload(request), request.state.user["id"]), status_code=202)
        manager.discover_history()
        summaries = []
        for job in sorted(manager.jobs.values(), key=lambda j: j["created_at"], reverse=True):
            if not visible(job, request.state.user):
                continue
            snap = manager.snapshot(job)
            summaries.append({key: snap[key] for key in
                ("id", "idea", "top_k", "status", "created_at", "updated_at", "completed_count", "selected_count")})
        return JSONResponse({"jobs": summaries})

    async def job_detail(request):
        return JSONResponse(manager.snapshot(owned_job(request)))

    async def cancel_job(request):
        return JSONResponse(await manager.cancel(owned_job(request)["id"]))

    async def artifact_response(request):
        job = owned_job(request)
        run = manager.run_directory(job)
        if not run:
            raise HTTPException(404, "文件尚未生成")
        paper_id = request.path_params.get("paper_id")
        if paper_id:
            if not PAPER_PATTERN.fullmatch(paper_id):
                raise HTTPException(404, "论文不存在")
            snapshot = manager.snapshot(job)
            if paper_id not in {p["id"] for p in snapshot["papers"]}:
                raise HTTPException(404, "论文不存在")
            path = manager.artifact(run, f"fulltext/{paper_id}.pdf")
            media, filename = "application/pdf", f"{paper_id}.pdf"
        else:
            path = manager.artifact(run, "report.md")
            media, filename = "text/markdown; charset=utf-8", f"research-{job['id'][:8]}.md"
        if not path.is_file():
            raise HTTPException(404, "文件尚未生成")
        return FileResponse(path, media_type=media, filename=filename)

    app = Starlette(lifespan=lifespan, exception_handlers={HTTPException: error_response}, routes=[
        Route("/api/auth/me", me),
        Route("/api/auth/register", register, methods=["POST"]),
        Route("/api/auth/login", login, methods=["POST"]),
        Route("/api/auth/logout", logout, methods=["POST"]),
        Route("/api/auth/profile", profile, methods=["PATCH"]),
        Route("/api/auth/password", password, methods=["POST"]),
        Route("/api/auth/avatar", avatar, methods=["POST", "DELETE"]),
        Route("/api/users/{user_id}/avatar", avatar_file),
        Route("/api/admin/users", admin_users),
        Route("/api/admin/users/{user_id}", admin_update, methods=["PATCH"]),
        Route("/api/admin/users/{user_id}/password", admin_password, methods=["POST"]),
        Route("/", home), Route("/api/status", status), Route("/api/jobs", jobs, methods=["GET", "POST"]),
        Route("/api/models", models), Route("/api/models/switch", switch_model, methods=["POST"]),
        Route("/api/jobs/{job_id}", job_detail),
        Route("/api/jobs/{job_id}/cancel", cancel_job, methods=["POST"]),
        Route("/api/jobs/{job_id}/report", artifact_response),
        Route("/api/jobs/{job_id}/papers/{paper_id}/pdf", artifact_response),
        Mount("/static", StaticFiles(directory=manager.project / "web", check_dir=False), name="static"),
    ])
    app.state.manager = manager
    app.state.accounts = accounts

    async def local_security(request: Request, call_next):
        rejection = None
        if request.method in {"POST", "PUT", "PATCH", "DELETE"}:
            origin = request.headers.get("origin")
            if origin and origin.rstrip("/") != str(request.base_url).rstrip("/"):
                rejection = JSONResponse({"error": "仅允许同源请求"}, status_code=403)
            if request.headers.get("sec-fetch-site") == "cross-site":
                rejection = JSONResponse({"error": "仅允许同源请求"}, status_code=403)
        request.state.user = None
        if not rejection and request.url.path.startswith("/api/"):
            token = request.cookies.get(cookie_name)
            if token:
                request.state.user = await run_in_threadpool(accounts.session_user, token)
            public = request.url.path in {"/api/auth/me", "/api/auth/login", "/api/auth/register", "/api/auth/logout"}
            if not public and request.state.user is None:
                rejection = JSONResponse({"error": "请先登录后继续"}, status_code=401)
        response = rejection if rejection is not None else await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; connect-src 'self'; font-src 'self'; frame-ancestors 'none'; base-uri 'self'"
        )
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        elif request.url.path == "/" or request.url.path.startswith("/static/"):
            # Revalidate the shell and assets together after local deployments.
            response.headers["Cache-Control"] = "no-cache"
        return response

    app.add_middleware(BaseHTTPMiddleware, dispatch=local_security)
    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    parser = argparse.ArgumentParser(description="Paper Research Agent 本地网页界面")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    options = parser.parse_args()
    uvicorn.run(app, host=options.host, port=options.port)
