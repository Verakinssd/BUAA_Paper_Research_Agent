"""HTTP account boundaries using isolated storage and no model/GPU services."""

import asyncio
from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

import httpx
from PIL import Image
from starlette.exceptions import HTTPException

from web_app import create_app, now_iso


class AccountAccessTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.app = create_app(output_root=self.root / "outputs", data_dir=self.root / "data")
        self.accounts = self.app.state.accounts
        self.accounts.initialize()
        self.manager = self.app.state.manager
        await self.manager.startup()
        self.manager.health = AsyncMock(return_value={
            "model": {"name": "test", "connected": True}, "active_job_id": None,
        })
        self.manager.model_control.catalog = AsyncMock(return_value={"managed": False, "models": []})
        self.manager.model_control.switch = AsyncMock(return_value={"status": "loading"})
        self.users = {}
        for role, name in (("admin", "administrator"), ("alice", "alice"), ("bob", "bob")):
            self.users[role] = await asyncio.to_thread(
                self.accounts.create_user, name, "test-pass-123", role=role if role == "admin" else "user")
        self.clients = {}
        for name, user in self.users.items():
            client = self.client()
            client.cookies.set("paper_session", self.accounts.create_session(user["id"]))
            self.clients[name] = client
        self.guest = self.client()

    def client(self):
        return httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url="http://testserver")

    async def asyncTearDown(self):
        await self.manager.shutdown()
        await self.guest.aclose()
        for client in self.clients.values():
            await client.aclose()
        self.temp.cleanup()

    def seed_job(self, char, owner=None, status="completed"):
        job_id = char * 32
        job = dict(id=job_id, idea=f"private topic {char}", top_k=1, concurrent=False,
                   status=status, created_at=now_iso(), updated_at=now_iso(), logs=[f"private log {char}"])
        if owner:
            job["owner_id"] = owner["id"]
        self.manager.jobs[job_id] = job
        self.manager.persist(job)
        run = self.manager.runs / job_id / (char * 12)
        (run / "fulltext").mkdir(parents=True)
        (run / "selection.json").write_text(json.dumps([{"id": "P1", "title": f"private paper {char}"}]))
        (run / "report.md").write_text(f"# private report {char}", encoding="utf-8")
        (run / "fulltext" / "P1.pdf").write_bytes(b"%PDF-1.4\nprivate test artifact")
        return job_id

    async def test_guest_can_load_shell_but_no_research_or_accounts(self):
        self.assertEqual((await self.guest.get("/")).status_code, 200)
        self.assertEqual((await self.guest.get("/api/auth/me")).json(), {"user": None})
        for path in ("/api/status", "/api/models", "/api/jobs", "/api/admin/users",
                     "/api/users/" + self.users["alice"]["id"] + "/avatar"):
            response = await self.guest.get(path)
            self.assertEqual(response.status_code, 401, path)
        response = await self.guest.post("/api/jobs", json={"idea": "guest cannot run"})
        self.assertEqual(response.status_code, 401)
        self.assertFalse(self.manager.jobs)

    async def test_registration_is_user_only_and_session_cookie_is_opaque(self):
        bad = await self.guest.post("/api/auth/register", json={
            "username": "new_account", "password": "test-pass-123", "role": "admin",
        })
        self.assertEqual(bad.status_code, 400)
        response = await self.guest.post("/api/auth/register", json={
            "username": "new_account", "password": "test-pass-123", "display_name": "Researcher",
        })
        self.assertIn(response.status_code, (200, 201))
        user = response.json()["user"]
        self.assertEqual(user["role"], "user")
        self.assertNotIn("password", json.dumps(user))
        self.assertEqual((await self.guest.get("/api/auth/me")).json()["user"]["id"], user["id"])
        cookie = response.headers["set-cookie"]
        self.assertIn("httponly", cookie.lower())
        self.assertIn("samesite=lax", cookie.lower())
        self.assertNotIn(user["username"], cookie)
        duplicate = await self.guest.post("/api/auth/register", json={
            "username": "NEW_ACCOUNT", "password": "test-pass-123",
        })
        self.assertEqual(duplicate.status_code, 409)

    async def test_invalid_login_is_generic_and_logout_revokes_copied_cookie(self):
        bodies = []
        for username in ("alice", "unknown_account"):
            response = await self.guest.post("/api/auth/login", json={"username": username, "password": "wrong-pass-123"})
            self.assertEqual(response.status_code, 401)
            bodies.append(response.json())
        self.assertEqual(bodies[0], bodies[1])
        response = await self.guest.post("/api/auth/login", json={"username": "ALICE", "password": "test-pass-123"})
        self.assertEqual(response.status_code, 200)
        token = self.guest.cookies.get("paper_session")
        self.assertTrue(self.accounts.session_user(token))
        self.assertEqual((await self.guest.post("/api/auth/logout", json={})).status_code, 200)
        self.assertIsNone(self.accounts.session_user(token))
        self.assertEqual((await self.guest.get("/api/auth/me")).json(), {"user": None})

    async def test_history_details_downloads_and_cancel_are_owner_scoped(self):
        alice_id = self.seed_job("a", self.users["alice"])
        bob_id = self.seed_job("b", self.users["bob"])
        archive_id = self.seed_job("c")
        admin_id = self.seed_job("d", self.users["admin"])
        expected = {"alice": {alice_id}, "bob": {bob_id}, "admin": {archive_id, admin_id}}
        for name, ids in expected.items():
            client = self.clients[name]
            listing = (await client.get("/api/jobs")).json()["jobs"]
            self.assertEqual({job["id"] for job in listing}, ids)
            for job_id in (alice_id, bob_id, archive_id, admin_id):
                for suffix in ("", "/report", "/papers/P1/pdf"):
                    response = await client.get(f"/api/jobs/{job_id}{suffix}")
                    self.assertEqual(response.status_code, 200 if job_id in ids else 404, (name, suffix, job_id))
                if job_id not in ids:
                    response = await client.post(f"/api/jobs/{job_id}/cancel", json={})
                    self.assertEqual(response.status_code, 404, (name, job_id))
        for job_id in (alice_id, bob_id):
            self.assertEqual(self.manager.jobs[job_id]["status"], "completed")

    async def test_status_does_not_reveal_another_users_active_job_id(self):
        job_id = self.seed_job("e", self.users["alice"], status="running")
        self.manager.health.return_value = {
            "model": {"connected": True}, "active_job_id": job_id,
        }
        for name in ("bob", "admin"):
            response = await self.clients[name].get("/api/status")
            self.assertEqual(response.status_code, 200)
            self.assertIsNone(response.json()["active_job_id"])
            self.assertTrue(response.json()["resource_busy"])
            self.assertNotIn(job_id, response.text)
        response = await self.clients["alice"].get("/api/status")
        self.assertEqual(response.json()["active_job_id"], job_id)

    async def test_new_jobs_persist_authenticated_owner(self):
        with patch.object(self.manager, "execute", new_callable=AsyncMock):
            response = await self.clients["alice"].post("/api/jobs", json={"idea": "owned research", "top_k": 1})
            self.assertEqual(response.status_code, 202, response.text)
            job_id = response.json()["id"]
            self.assertEqual(self.manager.jobs[job_id]["owner_id"], self.users["alice"]["id"])
            saved = json.loads((self.manager.manifests / (job_id + ".json")).read_text())
            self.assertEqual(saved["owner_id"], self.users["alice"]["id"])
            self.assertEqual((await self.clients["bob"].get(f"/api/jobs/{job_id}")).status_code, 404)
            await self.clients["alice"].post(f"/api/jobs/{job_id}/cancel", json={})

    async def test_only_admin_can_switch_shared_model(self):
        response = await self.clients["alice"].post("/api/models/switch", json={"model_id": "q5_k_m"})
        self.assertEqual(response.status_code, 403)
        self.manager.model_control.switch.assert_not_awaited()
        response = await self.clients["admin"].post("/api/models/switch", json={"model_id": "q5_k_m"})
        self.assertEqual(response.status_code, 202)
        self.manager.model_control.switch.assert_awaited_once_with("q5_k_m")

    async def test_profile_changes_do_not_change_role_or_other_users(self):
        alice = self.clients["alice"]
        response = await alice.patch("/api/auth/profile", json={
            "username": "alice_updated", "display_name": "<b>Alice</b>", "bio": "My research signature",
        })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["user"]["username"], "alice_updated")
        self.assertEqual(response.json()["user"]["display_name"], "<b>Alice</b>")
        self.assertEqual(self.accounts.profile(self.users["bob"]["id"])["username"], "bob")
        for body in ({"role": "admin"}, {"is_active": False}, {"username": "bob"}):
            response = await alice.patch("/api/auth/profile", json=body)
            self.assertIn(response.status_code, (400, 409))
        self.assertEqual(self.accounts.profile(self.users["alice"]["id"])["role"], "user")

    async def test_password_change_revokes_other_sessions_and_old_password(self):
        alice = self.clients["alice"]
        other_token = self.accounts.create_session(self.users["alice"]["id"])
        response = await alice.post("/api/auth/password", json={
            "current_password": "wrong-password", "new_password": "new-test-pass-123",
        })
        self.assertEqual(response.status_code, 400)
        self.assertTrue(self.accounts.session_user(other_token))
        response = await alice.post("/api/auth/password", json={
            "current_password": "test-pass-123", "new_password": "new-test-pass-123",
        })
        self.assertEqual(response.status_code, 200)
        self.assertIsNone(self.accounts.session_user(other_token))
        self.assertEqual((await alice.get("/api/auth/me")).json()["user"]["id"], self.users["alice"]["id"])
        self.assertEqual((await self.guest.post("/api/auth/login", json={"username": "alice", "password": "test-pass-123"})).status_code, 401)
        self.assertEqual((await self.guest.post("/api/auth/login", json={"username": "alice", "password": "new-test-pass-123"})).status_code, 200)

    async def test_avatars_are_reencoded_and_require_authentication(self):
        image = BytesIO()
        Image.new("RGB", (640, 300), "#237768").save(image, format="JPEG")
        response = await self.clients["alice"].post("/api/auth/avatar", content=image.getvalue(), headers={"content-type": "image/jpeg"})
        self.assertEqual(response.status_code, 200)
        url = response.json()["user"]["avatar_url"]
        self.assertTrue(url.startswith("/api/users/"))
        response = await self.clients["alice"].get(url)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "image/png")
        with Image.open(BytesIO(response.content)) as avatar:
            self.assertEqual(avatar.size, (256, 256))
        self.assertEqual((await self.guest.get(url)).status_code, 401)
        response = await self.clients["alice"].post("/api/auth/avatar", content=b"<svg onload='alert(1)'/>", headers={"content-type": "image/svg+xml"})
        self.assertEqual(response.status_code, 400)
        response = await self.clients["alice"].post("/api/auth/avatar", content=b"<html>not an image</html>", headers={"content-type": "image/png"})
        self.assertEqual(response.status_code, 400)
        response = await self.clients["alice"].delete("/api/auth/avatar")
        self.assertEqual(response.status_code, 200)
        self.assertIsNone(response.json()["user"]["avatar_url"])
        self.assertEqual((await self.clients["alice"].get(url)).status_code, 404)

    async def test_admin_can_manage_users_without_receiving_password_hashes(self):
        alice, admin = self.clients["alice"], self.clients["admin"]
        user_path = "/api/admin/users/" + self.users["bob"]["id"]
        self.assertEqual((await alice.get("/api/admin/users")).status_code, 403)
        self.assertEqual((await alice.patch(user_path, json={"is_active": False})).status_code, 403)
        self.assertEqual((await alice.post(user_path + "/password", json={"new_password": "other-pass-123"})).status_code, 403)
        listing = await admin.get("/api/admin/users?search=bob&page=1&page_size=20")
        self.assertEqual(listing.status_code, 200)
        self.assertEqual(listing.json()["total"], 1)
        self.assertEqual(listing.json()["users"][0]["id"], self.users["bob"]["id"])
        self.assertNotIn("password_hash", listing.text)
        response = await admin.patch(user_path, json={"display_name": "Managed Bob", "bio": "Updated by admin", "role": "admin"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["user"]["role"], "admin")
        response = await admin.patch(user_path, json={"role": "user", "is_active": False})
        self.assertEqual(response.status_code, 200)
        self.assertEqual((await self.clients["bob"].get("/api/status")).status_code, 401)
        self.assertEqual((await self.guest.post("/api/auth/login", json={"username": "bob", "password": "test-pass-123"})).status_code, 401)
        self.assertEqual((await admin.patch(user_path, json={"is_active": True})).status_code, 200)
        response = await admin.post(user_path + "/password", json={"new_password": "reset-test-pass-123"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual((await self.guest.post("/api/auth/login", json={"username": "bob", "password": "reset-test-pass-123"})).status_code, 200)

    async def test_last_admin_and_cross_origin_mutations_are_rejected(self):
        admin = self.clients["admin"]
        own_path = "/api/admin/users/" + self.users["admin"]["id"]
        for data in ({"role": "user"}, {"is_active": False}):
            self.assertEqual((await admin.patch(own_path, json=data)).status_code, 400)
        for method, path, kwargs in (
            ("patch", "/api/auth/profile", {"json": {"bio": "cross-site"}}),
            ("delete", "/api/auth/avatar", {}),
            ("post", "/api/auth/logout", {"json": {}}),
        ):
            response = await getattr(admin, method)(path, headers={"origin": "https://unrelated.test"}, **kwargs)
            self.assertEqual(response.status_code, 403, method)
        self.assertEqual(self.accounts.profile(self.users["admin"]["id"])["role"], "admin")

    async def test_secure_cookie_flag_can_be_enabled_for_https_deployments(self):
        with patch.dict("os.environ", {"PAPER_COOKIE_SECURE": "1"}):
            response = await self.guest.post("/api/auth/login", json={
                "username": "alice", "password": "test-pass-123",
            })
        self.assertEqual(response.status_code, 200)
        self.assertIn("secure", response.headers["set-cookie"].lower())

    async def test_auth_attempt_limits_stop_repeated_requests(self):
        with patch.object(self.accounts, "authenticate", side_effect=HTTPException(401, "invalid")):
            for _ in range(30):
                response = await self.guest.post("/api/auth/login", json={
                    "username": "alice", "password": "wrong-pass-123",
                })
                self.assertEqual(response.status_code, 401)
            self.assertEqual((await self.guest.post("/api/auth/login", json={
                "username": "alice", "password": "wrong-pass-123",
            })).status_code, 429)
        for _ in range(10):
            self.assertEqual((await self.guest.post("/api/auth/register", json={})).status_code, 400)
        self.assertEqual((await self.guest.post("/api/auth/register", json={})).status_code, 429)

    async def test_oversized_json_and_streamed_avatar_are_rejected(self):
        response = await self.clients["alice"].patch("/api/auth/profile", content=b"x" * 20_001,
                                                   headers={"content-type": "application/json"})
        self.assertEqual(response.status_code, 413)

        async def avatar_chunks():
            yield b"x" * (2 * 1024 * 1024)
            yield b"x" * (2 * 1024 * 1024)

        response = await self.clients["alice"].post("/api/auth/avatar", content=avatar_chunks(),
                                                  headers={"content-type": "image/png"})
        self.assertEqual(response.status_code, 413)
        self.assertIsNone(self.accounts.profile(self.users["alice"]["id"])["avatar_url"])


if __name__ == "__main__":
    unittest.main()
