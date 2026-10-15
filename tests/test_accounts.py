from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
import sqlite3
import tempfile
import time
import unittest
from unittest.mock import patch

from PIL import Image
from starlette.exceptions import HTTPException

from research.accounts import UserStore, MAX_AVATAR_BYTES, _verify_password


PASSWORD = "correct horse battery"


class AccountStoreTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temporary.name) / "data"
        self.store = UserStore(self.data_dir)

    def tearDown(self):
        self.temporary.cleanup()

    def assert_status(self, status, function, *args, **kwargs):
        with self.assertRaises(HTTPException) as caught:
            function(*args, **kwargs)
        self.assertEqual(caught.exception.status_code, status)

    def create(self, name="researcher", role="user"):
        return self.store.create_user(name, PASSWORD, role=role)

    def test_lazy_creation_password_storage_and_username_normalization(self):
        self.assertFalse(self.data_dir.exists())
        self.assertIsNone(self.store.session_user(None))
        self.assertFalse(self.data_dir.exists())
        user = self.create("Ｒｅｓｅａｒｃｈｅｒ")
        self.assertEqual(user["username"], "Researcher")
        self.assertEqual(user["role"], "user")
        self.assertEqual(self.store.authenticate("researcher", PASSWORD)["id"], user["id"])
        self.assertNotIn("password_hash", user)
        self.assert_status(409, self.create, "RESEARCHER")
        with sqlite3.connect(self.store.db_path) as connection:
            encoded = connection.execute("SELECT password_hash FROM users").fetchone()[0]
            self.assertEqual(connection.execute("PRAGMA journal_mode").fetchone()[0], "delete")
        self.assertTrue(encoded.startswith("pbkdf2_sha256$600000$"))
        self.assertNotIn(PASSWORD, encoded)
        self.assertTrue(_verify_password(PASSWORD, encoded))
        self.assertFalse(_verify_password("incorrect", encoded))
        self.assert_status(401, self.store.authenticate, "researcher", "incorrect")
        self.assert_status(401, self.store.authenticate, "does-not-exist", PASSWORD)
        self.assert_status(401, self.store.authenticate, [], PASSWORD)

    def test_unique_username_is_enforced_across_concurrent_stores(self):
        def register():
            try:
                return UserStore(self.data_dir).create_user("Concurrent", PASSWORD)["id"]
            except HTTPException as exc:
                return exc.status_code
        with ThreadPoolExecutor(max_workers=3) as executor:
            outcomes = list(executor.map(lambda _: register(), range(3)))
        self.assertEqual(sum(isinstance(result, str) for result in outcomes), 1)
        self.assertEqual(outcomes.count(409), 2)

    def test_profile_validation_forbids_privilege_escalation(self):
        user = self.create()
        updated = self.store.update_profile(user["id"], {"username": "new_user", "display_name": "新名字", "bio": "阅读论文\n探索新想法"})
        self.assertEqual(updated["display_name"], "新名字")
        self.assertEqual(updated["bio"], "阅读论文\n探索新想法")
        self.assertEqual(self.store.authenticate("new_user", PASSWORD)["id"], user["id"])
        for invalid in ({"role": "admin"}, {"is_active": True}, {"username": "../bad"},
                        {"display_name": ""}, {"bio": "x" * 201}, {"display_name": []}):
            self.assert_status(400, self.store.update_profile, user["id"], invalid)
        self.assert_status(400, self.store.create_user, "ok_user", "short")
        self.assert_status(400, self.store.create_user, "ok_user", "x" * 129)
        self.assert_status(400, self.store.create_user, "ok_user", PASSWORD, role=[])
        self.assert_status(404, self.store.profile, "../bad")

    def test_sessions_are_hashed_expire_and_password_change_revokes_others(self):
        user = self.create()
        first, second = (self.store.create_session(user["id"]) for _ in range(2))
        self.assertNotEqual(first, second)
        self.assertEqual(self.store.session_user(first)["id"], user["id"])
        with sqlite3.connect(self.store.db_path) as connection:
            hashes = [row[0] for row in connection.execute("SELECT token_hash FROM sessions")]
        self.assertNotIn(first, hashes)
        self.assertTrue(all(len(digest) == 64 for digest in hashes))
        self.assert_status(400, self.store.change_password, user["id"], "wrong password", "new password")
        self.assertIsNotNone(self.store.session_user(second))
        self.store.change_password(user["id"], PASSWORD, "new password", current_token=first)
        self.assertIsNotNone(self.store.session_user(first))
        self.assertIsNone(self.store.session_user(second))
        self.assert_status(401, self.store.authenticate, user["username"], PASSWORD)
        self.assertEqual(self.store.authenticate(user["username"], "new password")["id"], user["id"])
        with patch("research.accounts.time.time", return_value=time.time() + 8 * 86400):
            self.assertIsNone(self.store.session_user(first))
        self.store.revoke_session(first)
        self.assertIsNone(self.store.session_user(first))

    def test_admin_permissions_disable_reset_and_last_admin_invariants(self):
        admin = self.create("administrator", "admin")
        user = self.create()
        token = self.store.create_session(user["id"])
        self.assert_status(403, self.store.admin_update, user["id"], admin["id"], {"role": "user"})
        self.assert_status(400, self.store.admin_update, admin["id"], admin["id"], {"role": "user"})
        self.assert_status(400, self.store.admin_update, admin["id"], admin["id"], {"is_active": False})
        self.assert_status(400, self.store.admin_update, admin["id"], user["id"], {"role": []})
        self.assert_status(400, self.store.admin_update, admin["id"], user["id"], {"is_active": "false"})
        self.store.admin_update(admin["id"], user["id"], {"is_active": False})
        self.assertIsNone(self.store.session_user(token))
        self.assert_status(401, self.store.authenticate, user["username"], PASSWORD)
        self.assert_status(403, self.store.create_session, user["id"])
        self.store.admin_update(admin["id"], user["id"], {"is_active": True})
        token = self.store.create_session(user["id"])
        self.store.reset_password(admin["id"], user["id"], "reset password")
        self.assertIsNone(self.store.session_user(token))
        self.assertEqual(self.store.authenticate(user["username"], "reset password")["id"], user["id"])
        self.assert_status(400, self.store.reset_password, admin["id"], admin["id"], "reset password")

    def test_concurrent_admin_demotions_preserve_one_admin_and_revoke_session(self):
        first = self.create("first_admin", "admin")
        second = self.create("second_admin", "admin")
        first_token, second_token = (self.store.create_session(user["id"]) for user in (first, second))
        def demote(user):
            try:
                UserStore(self.data_dir).admin_update(user["id"], user["id"], {"role": "user"})
                return 200
            except HTTPException as exc:
                return exc.status_code
        with ThreadPoolExecutor(max_workers=2) as executor:
            outcomes = list(executor.map(demote, (first, second)))
        self.assertEqual(sorted(outcomes), [200, 400])
        remaining = self.store.admin_list()["users"]
        self.assertEqual(sum(user["role"] == "admin" for user in remaining), 1)
        self.assertEqual(sum(self.store.session_user(token) is not None for token in (first_token, second_token)), 1)

    def test_admin_search_and_pagination_do_not_treat_wildcards_as_patterns(self):
        self.create("first_user")
        self.create("second-user")
        self.assertEqual(self.store.admin_list(search="FIRST")["total"], 1)
        self.assertEqual(self.store.admin_list(search="_")["total"], 1)
        self.assertEqual(self.store.admin_list(search="%")["total"], 0)
        page = self.store.admin_list(page=2, page_size=1)
        self.assertEqual(page["total"], 2)
        self.assertEqual(len(page["users"]), 1)
        for options in ({"page": True}, {"page_size": 101}, {"search": []}):
            self.assert_status(400, self.store.admin_list, **options)

    @staticmethod
    def image_bytes(format="PNG"):
        stream = BytesIO()
        Image.new("RGB", (350, 200), "#0d8471").save(stream, format=format)
        return stream.getvalue()

    def test_avatar_reencoding_cache_revision_and_deletion(self):
        user = self.create()
        self.assertIsNone(user["avatar_url"])
        updated = self.store.set_avatar(user["id"], self.image_bytes("JPEG"), "image/jpeg")
        self.assertIn("?v=1", updated["avatar_url"])
        with Image.open(self.store.avatar_path(user["id"])) as image:
            self.assertEqual(image.format, "PNG")
            self.assertEqual(image.size, (256, 256))
            self.assertFalse(image.info)
        updated = self.store.set_avatar(user["id"], self.image_bytes("WEBP"), "image/webp")
        self.assertIn("?v=2", updated["avatar_url"])
        self.assertIsNone(self.store.delete_avatar(user["id"])["avatar_url"])
        self.assert_status(404, self.store.avatar_path, user["id"])

    def test_avatar_rejects_invalid_oversized_mismatched_and_traversal_files(self):
        user = self.create()
        self.assert_status(400, self.store.set_avatar, user["id"], b"<svg></svg>", "image/svg+xml")
        self.assert_status(400, self.store.set_avatar, user["id"], b"<html>bad</html>", "image/png")
        self.assert_status(400, self.store.set_avatar, user["id"], self.image_bytes(), "image/jpeg")
        self.assert_status(413, self.store.set_avatar, user["id"], b"x" * (MAX_AVATAR_BYTES + 1), "image/png")
        self.assert_status(404, self.store.avatar_path, "../accounts.sqlite3")
        outside = Path(self.temporary.name) / "outside.png"
        outside.write_bytes(self.image_bytes())
        target = self.store.avatars_dir / (user["id"] + ".png")
        target.symlink_to(outside)
        self.assert_status(404, self.store.set_avatar, user["id"], self.image_bytes(), "image/png")
        self.assertEqual(outside.read_bytes(), self.image_bytes())
        self.assertTrue(target.is_symlink())


if __name__ == "__main__":
    unittest.main()
