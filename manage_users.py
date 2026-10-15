"""Create an explicit administrator account without putting passwords in argv."""

import argparse
from getpass import getpass
import os
from pathlib import Path

from dotenv import load_dotenv
from starlette.exceptions import HTTPException

from research.accounts import UserStore


PROJECT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description="管理网页用户账户")
    parser.add_argument("command", choices=["create-admin"], help="交互式创建管理员账户")
    args = parser.parse_args()
    load_dotenv(PROJECT / ".env")
    configured_dir = Path(os.getenv("PAPER_DATA_DIR") or PROJECT / "data").expanduser()
    if not configured_dir.is_absolute():
        configured_dir = PROJECT / configured_dir
    if args.command == "create-admin":
        try:
            username = input("管理员用户名: ").strip()
            password = getpass("密码（8–128 个字符）: ")
            confirmation = getpass("再次输入密码: ")
            if password != confirmation:
                parser.exit(1, "两次输入的密码不一致，未创建账户。\n")
            user = UserStore(configured_dir).create_user(username, password, role="admin")
        except HTTPException as exc:
            parser.exit(1, f"创建失败：{exc.detail}\n")
        except (EOFError, KeyboardInterrupt):
            parser.exit(1, "\n操作已取消。\n")
        print(f'管理员账户 {user["username"]} 已创建，可在网页中登录。')


if __name__ == "__main__":
    main()
