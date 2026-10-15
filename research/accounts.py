"""Local SQLite accounts, opaque sessions, and validated avatar storage.

Constructing a store does not create files. Call ``initialize`` at application
startup, or use one of its methods, to initialize the configured data directory.
"""

from __future__ import annotations

import base64
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import hmac
from io import BytesIO
import os
from pathlib import Path
import re
import secrets
import sqlite3
import tempfile
import threading
import time
import unicodedata
import uuid
import warnings

from starlette.exceptions import HTTPException


PASSWORD_ITERATIONS = 600_000
SESSION_TTL_SECONDS = 7 * 24 * 60 * 60
MAX_AVATAR_BYTES = 3 * 1024 * 1024
MAX_AVATAR_PIXELS = 16_000_000
USER_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
TOKEN_PATTERN = re.compile(r"^[A-Za-z0-9_-]{43}$")
PROFILE_FIELDS = frozenset({"username", "display_name", "bio"})


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _error(status: int, message: str):
    raise HTTPException(status_code=status, detail=message)


def _username(value) -> tuple[str, str]:
    if not isinstance(value, str):
        _error(400, "用户名必须是文本。")
    value = unicodedata.normalize("NFKC", value).strip()
    if not 3 <= len(value) <= 32 or any(
        char not in "._-" and unicodedata.category(char)[0] not in {"L", "N"}
        for char in value
    ):
        _error(400, "用户名需为 3–32 个字符，只能包含字母、数字、点、下划线或连字符。")
    return value, value.casefold()


def _text(value, name: str, minimum: int, maximum: int, multiline=False) -> str:
    if not isinstance(value, str):
        _error(400, f"{name}必须是文本。")
    value = value.strip()
    if not minimum <= len(value) <= maximum:
        _error(400, f"{name}长度需为 {minimum}–{maximum} 个字符。")
    if any(unicodedata.category(char)[0] == "C" and not (multiline and char in "\n\t")
           for char in value):
        _error(400, f"{name}包含不支持的字符。")
    return value


def _password(value) -> str:
    if not isinstance(value, str) or not 8 <= len(value) <= 128:
        _error(400, "密码长度需为 8–128 个字符。")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        _error(400, "密码包含不支持的字符。")
    return value


def _hash_password(password: str) -> str:
    salt = secrets.token_bytes(32)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PASSWORD_ITERATIONS)
    return "$".join(("pbkdf2_sha256", str(PASSWORD_ITERATIONS),
                     base64.b64encode(salt).decode("ascii"), base64.b64encode(digest).decode("ascii")))


def _verify_password(password: str, encoded: str | None) -> bool:
    """Unknown accounts still perform the expensive derivation before failing."""
    iterations, salt, expected = PASSWORD_ITERATIONS, b"paper-agent-invalid-user", bytes(32)
    valid_encoding = False
    if encoded:
        try:
            algorithm, rounds, salt_text, digest_text = encoded.split("$")
            parsed_rounds = int(rounds)
            parsed_salt = base64.b64decode(salt_text, validate=True)
            parsed_digest = base64.b64decode(digest_text, validate=True)
            if (algorithm == "pbkdf2_sha256" and PASSWORD_ITERATIONS <= parsed_rounds <= 2_000_000
                    and len(parsed_salt) >= 16 and len(parsed_digest) == 32):
                iterations, salt, expected = parsed_rounds, parsed_salt, parsed_digest
                valid_encoding = True
        except (ValueError, TypeError):
            pass
    try:
        raw_password = password.encode("utf-8")
    except (AttributeError, UnicodeEncodeError):
        raw_password = b"invalid-password"
    digest = hashlib.pbkdf2_hmac("sha256", raw_password, salt, iterations)
    return hmac.compare_digest(digest, expected) and valid_encoding


def _token_hash(token) -> str | None:
    if not isinstance(token, str) or not TOKEN_PATTERN.fullmatch(token):
        return None
    return hashlib.sha256(token.encode("ascii")).hexdigest()


def _profile_changes(payload) -> dict:
    if not isinstance(payload, dict) or set(payload) - PROFILE_FIELDS:
        _error(400, "个人资料包含不支持的字段。")
    changes = {}
    if "username" in payload:
        changes["username"], changes["username_key"] = _username(payload["username"])
    if "display_name" in payload:
        changes["display_name"] = _text(payload["display_name"], "显示名称", 1, 40)
        changes["display_name_key"] = changes["display_name"].casefold()
    if "bio" in payload:
        changes["bio"] = _text(payload["bio"], "个性签名", 0, 200, multiline=True)
    return changes


class UserStore:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir).expanduser().resolve()
        self.db_path = self.data_dir / "accounts.sqlite3"
        self.avatars_dir = self.data_dir / "avatars"
        self._initialized = False
        self._initialize_lock = threading.Lock()

    def initialize(self):
        if self._initialized:
            return
        with self._initialize_lock:
            if self._initialized:
                return
            self.data_dir.mkdir(parents=True, exist_ok=True)
            self.avatars_dir.mkdir(exist_ok=True)
            if self.avatars_dir.is_symlink():
                raise RuntimeError("头像目录不能是符号链接。")
            connection = sqlite3.connect(self.db_path, timeout=15)
            try:
                connection.execute("PRAGMA busy_timeout = 15000")
                connection.execute("PRAGMA journal_mode = DELETE")
                connection.execute("PRAGMA foreign_keys = ON")
                connection.executescript("""
                    BEGIN IMMEDIATE;
                    CREATE TABLE IF NOT EXISTS users (
                        id TEXT PRIMARY KEY,
                        username TEXT NOT NULL,
                        username_key TEXT NOT NULL UNIQUE,
                        display_name TEXT NOT NULL,
                        display_name_key TEXT NOT NULL,
                        bio TEXT NOT NULL DEFAULT '',
                        password_hash TEXT NOT NULL,
                        role TEXT NOT NULL CHECK(role IN ('user', 'admin')),
                        is_active INTEGER NOT NULL DEFAULT 1 CHECK(is_active IN (0, 1)),
                        avatar_enabled INTEGER NOT NULL DEFAULT 0,
                        avatar_revision INTEGER NOT NULL DEFAULT 0,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS sessions (
                        token_hash TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                        created_at INTEGER NOT NULL,
                        expires_at INTEGER NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS sessions_user_id ON sessions(user_id);
                    CREATE INDEX IF NOT EXISTS sessions_expires_at ON sessions(expires_at);
                    COMMIT;
                """)
            finally:
                connection.close()
            try:
                self.db_path.chmod(0o600)
            except OSError:
                pass
            self._initialized = True

    @contextmanager
    def _connection(self, write=False):
        self.initialize()
        connection = sqlite3.connect(self.db_path, timeout=15, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 15000")
        try:
            if write:
                connection.execute("BEGIN IMMEDIATE")
            yield connection
            if write:
                connection.commit()
        except BaseException:
            if write:
                connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _public(row) -> dict:
        return {
            "id": row["id"], "username": row["username"], "display_name": row["display_name"],
            "bio": row["bio"], "role": row["role"], "is_active": bool(row["is_active"]),
            "avatar_url": (f'/api/users/{row["id"]}/avatar?v={row["avatar_revision"]}'
                           if row["avatar_enabled"] else None),
            "created_at": row["created_at"], "updated_at": row["updated_at"],
        }

    @staticmethod
    def _user(connection, user_id):
        if not isinstance(user_id, str) or not USER_ID_PATTERN.fullmatch(user_id):
            _error(404, "用户不存在。")
        row = connection.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        if row is None:
            _error(404, "用户不存在。")
        return row

    @classmethod
    def _admin(cls, connection, actor_id):
        row = cls._user(connection, actor_id)
        if not row["is_active"] or row["role"] != "admin":
            _error(403, "需要管理员权限。")
        return row

    @staticmethod
    def _apply_changes(connection, user_id, changes):
        if not changes:
            return
        changes = {**changes, "updated_at": _now()}
        query = "UPDATE users SET " + ", ".join(f"{key} = ?" for key in changes) + " WHERE id = ?"
        try:
            connection.execute(query, (*changes.values(), user_id))
        except sqlite3.IntegrityError:
            _error(409, "这个用户名已被使用。")

    def create_user(self, username, password, display_name=None, role="user") -> dict:
        username, username_key = _username(username)
        password = _password(password)
        if not isinstance(role, str) or role not in {"user", "admin"}:
            _error(400, "不支持的用户身份。")
        display_name = _text(display_name if display_name is not None else username,
                             "显示名称", 1, 40)
        password_hash = _hash_password(password)
        user_id, timestamp = uuid.uuid4().hex, _now()
        with self._connection(write=True) as connection:
            try:
                connection.execute("""
                    INSERT INTO users (id, username, username_key, display_name, display_name_key,
                                       password_hash, role, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (user_id, username, username_key, display_name, display_name.casefold(),
                      password_hash, role, timestamp, timestamp))
            except sqlite3.IntegrityError:
                _error(409, "这个用户名已被使用。")
            return self._public(self._user(connection, user_id))

    def authenticate(self, username, password) -> dict:
        key = None
        try:
            _, key = _username(username)
            _password(password)
        except HTTPException:
            _verify_password("invalid-password", None)
            _error(401, "用户名或密码错误，或账户已被停用。")
        with self._connection() as connection:
            row = connection.execute("SELECT * FROM users WHERE username_key = ?", (key,)).fetchone()
        valid = _verify_password(password, row["password_hash"] if row is not None else None)
        if not valid or row is None or not row["is_active"]:
            _error(401, "用户名或密码错误，或账户已被停用。")
        return self._public(row)

    def create_session(self, user_id) -> str:
        token = secrets.token_urlsafe(32)
        timestamp = int(time.time())
        with self._connection(write=True) as connection:
            row = self._user(connection, user_id)
            if not row["is_active"]:
                _error(403, "账户已被停用。")
            connection.execute("DELETE FROM sessions WHERE expires_at <= ?", (timestamp,))
            connection.execute("INSERT INTO sessions VALUES (?, ?, ?, ?)",
                               (_token_hash(token), user_id, timestamp, timestamp + SESSION_TTL_SECONDS))
        return token

    def session_user(self, token) -> dict | None:
        digest = _token_hash(token)
        if digest is None:
            return None
        with self._connection() as connection:
            row = connection.execute("""
                SELECT users.* FROM sessions JOIN users ON users.id = sessions.user_id
                WHERE token_hash = ? AND expires_at > ? AND users.is_active = 1
            """, (digest, int(time.time()))).fetchone()
            return self._public(row) if row else None

    def revoke_session(self, token):
        digest = _token_hash(token)
        if digest is not None:
            with self._connection(write=True) as connection:
                connection.execute("DELETE FROM sessions WHERE token_hash = ?", (digest,))

    def profile(self, user_id) -> dict:
        with self._connection() as connection:
            return self._public(self._user(connection, user_id))

    def update_profile(self, user_id, payload) -> dict:
        changes = _profile_changes(payload)
        with self._connection(write=True) as connection:
            self._user(connection, user_id)
            self._apply_changes(connection, user_id, changes)
            return self._public(self._user(connection, user_id))

    def change_password(self, user_id, current_password, new_password, current_token=None):
        new_hash = _hash_password(_password(new_password))
        _password(current_password)
        with self._connection(write=True) as connection:
            row = self._user(connection, user_id)
            if not _verify_password(current_password, row["password_hash"]):
                _error(400, "当前密码不正确。")
            self._apply_changes(connection, user_id, {"password_hash": new_hash})
            keep_digest = _token_hash(current_token)
            if keep_digest:
                connection.execute("DELETE FROM sessions WHERE user_id = ? AND token_hash <> ?",
                                   (user_id, keep_digest))
            else:
                connection.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))

    def admin_list(self, search="", page=1, page_size=20) -> dict:
        if not isinstance(search, str) or len(search) > 100:
            _error(400, "搜索词最多为 100 个字符。")
        search = _text(search, "搜索词", 0, 100)
        if type(page) is not int or not 1 <= page <= 1_000_000 or type(page_size) is not int or not 1 <= page_size <= 100:
            _error(400, "分页参数不正确。")
        search = unicodedata.normalize("NFKC", search).strip().casefold()
        pattern = "%" + search.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
        condition = "WHERE username_key LIKE ? ESCAPE '\\' OR display_name_key LIKE ? ESCAPE '\\'"
        with self._connection() as connection:
            total = connection.execute("SELECT COUNT(*) FROM users " + condition, (pattern, pattern)).fetchone()[0]
            rows = connection.execute("SELECT * FROM users " + condition +
                                      " ORDER BY created_at DESC, id DESC LIMIT ? OFFSET ?",
                                      (pattern, pattern, page_size, (page - 1) * page_size)).fetchall()
        return {"users": [self._public(row) for row in rows], "total": total,
                "page": page, "page_size": page_size}

    def admin_update(self, actor_id, target_id, payload) -> dict:
        if not isinstance(payload, dict) or set(payload) - (PROFILE_FIELDS | {"role", "is_active"}):
            _error(400, "用户管理包含不支持的字段。")
        changes = _profile_changes({key: value for key, value in payload.items() if key in PROFILE_FIELDS})
        if "role" in payload:
            if not isinstance(payload["role"], str) or payload["role"] not in {"user", "admin"}:
                _error(400, "不支持的用户身份。")
            changes["role"] = payload["role"]
        if "is_active" in payload:
            if type(payload["is_active"]) is not bool:
                _error(400, "账户状态必须是布尔值。")
            changes["is_active"] = int(payload["is_active"])
        with self._connection(write=True) as connection:
            self._admin(connection, actor_id)
            target = self._user(connection, target_id)
            if actor_id == target_id and changes.get("is_active") == 0:
                _error(400, "不能停用当前登录的管理员账户。")
            removing_active_admin = target["role"] == "admin" and target["is_active"] and (
                changes.get("role", target["role"]) != "admin" or not changes.get("is_active", target["is_active"]))
            if removing_active_admin:
                count = connection.execute("SELECT COUNT(*) FROM users WHERE role = 'admin' AND is_active = 1").fetchone()[0]
                if count <= 1:
                    _error(400, "必须保留至少一个已启用的管理员账户。")
            self._apply_changes(connection, target_id, changes)
            if changes.get("is_active") == 0 or (target["role"] == "admin" and changes.get("role") == "user"):
                connection.execute("DELETE FROM sessions WHERE user_id = ?", (target_id,))
            return self._public(self._user(connection, target_id))

    def reset_password(self, actor_id, target_id, new_password):
        new_hash = _hash_password(_password(new_password))
        with self._connection(write=True) as connection:
            self._admin(connection, actor_id)
            self._user(connection, target_id)
            if actor_id == target_id:
                _error(400, "请在个人空间中修改自己的密码。")
            self._apply_changes(connection, target_id, {"password_hash": new_hash})
            connection.execute("DELETE FROM sessions WHERE user_id = ?", (target_id,))

    def _avatar_file(self, user_id) -> Path:
        if not isinstance(user_id, str) or not USER_ID_PATTERN.fullmatch(user_id):
            _error(404, "头像不存在。")
        target = self.avatars_dir / f"{user_id}.png"
        if self.avatars_dir.is_symlink() or target.is_symlink():
            _error(404, "头像不存在。")
        return target

    @staticmethod
    def _decode_avatar(rawbytes, content_type) -> bytes:
        if not isinstance(rawbytes, bytes) or not rawbytes:
            _error(400, "请选择图片文件。")
        if len(rawbytes) > MAX_AVATAR_BYTES:
            _error(413, "头像文件不能超过 3 MB。")
        mime = content_type.split(";", 1)[0].lower().strip() if isinstance(content_type, str) else ""
        formats = {"image/png": "PNG", "image/jpeg": "JPEG", "image/webp": "WEBP"}
        if mime not in formats:
            _error(400, "头像仅支持 PNG、JPEG 或 WebP 图片。")
        from PIL import Image, ImageOps, UnidentifiedImageError
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(BytesIO(rawbytes)) as image:
                    if image.format != formats[mime]:
                        _error(400, "图片内容与文件格式不匹配。")
                    if image.width * image.height > MAX_AVATAR_PIXELS:
                        _error(400, "头像图片分辨率过大。")
                    image.load()
                    image = ImageOps.exif_transpose(image).convert("RGBA")
                    image = ImageOps.fit(image, (256, 256), method=Image.Resampling.LANCZOS)
                    result = BytesIO()
                    image.save(result, format="PNG", optimize=True)
                    return result.getvalue()
        except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError,
                Image.DecompressionBombWarning):
            _error(400, "无法读取这张图片，请选择有效的 PNG、JPEG 或 WebP 文件。")

    def set_avatar(self, user_id, rawbytes, content_type) -> dict:
        encoded = self._decode_avatar(rawbytes, content_type)
        with self._connection(write=True) as connection:
            row = self._user(connection, user_id)
            target = self._avatar_file(user_id)
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(dir=self.avatars_dir, prefix=".avatar-", suffix=".png", delete=False) as output:
                    temporary = Path(output.name)
                    output.write(encoded)
                os.replace(temporary, target)
            finally:
                if temporary and temporary.exists():
                    temporary.unlink()
            self._apply_changes(connection, user_id, {"avatar_enabled": 1, "avatar_revision": row["avatar_revision"] + 1})
            return self._public(self._user(connection, user_id))

    def delete_avatar(self, user_id) -> dict:
        with self._connection(write=True) as connection:
            row = self._user(connection, user_id)
            target = self._avatar_file(user_id)
            target.unlink(missing_ok=True)
            self._apply_changes(connection, user_id, {"avatar_enabled": 0, "avatar_revision": row["avatar_revision"] + 1})
            return self._public(self._user(connection, user_id))

    def avatar_path(self, user_id) -> Path:
        with self._connection() as connection:
            row = self._user(connection, user_id)
            target = self._avatar_file(user_id)
            if not row["avatar_enabled"] or not target.is_file():
                _error(404, "头像不存在。")
            return target
