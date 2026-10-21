"""Loopback-only origin for an HTTPS-terminating development tunnel.

Do not expose this HTTP origin directly to a LAN or the internet. Visitors use
the tunnel's HTTPS URL; this wrapper preserves same-origin checks and enables
Secure cookies even when the tunnel omits X-Forwarded-Proto.
"""
import argparse
import os
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=7862)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("port 必须为 1–65535")

    os.environ["PAPER_COOKIE_SECURE"] = "1"
    from web_app import create_app
    import uvicorn

    application = create_app()

    async def https_origin(scope, receive, send):
        if scope["type"] == "http":
            scope = {**scope, "scheme": "https"}
        await application(scope, receive, send)

    uvicorn.run(https_origin, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
