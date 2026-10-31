"""Serve the web phone page on this computer only.

    python -m cli.phone_page [--port 8080]

Then open http://127.0.0.1:8080 in Chrome or Edge, and call a robot, for
example the echo robot (python -m cli.dummy_robot). Browsers only allow the
microphone on https or on this computer's own address, which is why it
listens on 127.0.0.1.
"""

from __future__ import annotations

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PAGE_DIR = Path(__file__).resolve().parents[1] / "lab" / "web_phone"


class _PhoneFiles(SimpleHTTPRequestHandler):
    # Browsers refuse a JavaScript module sent with the wrong type, and on
    # Windows the registry can map .js to text/plain. So the types are pinned.
    extensions_map = {
        **SimpleHTTPRequestHandler.extensions_map,
        ".js": "text/javascript",
        ".mjs": "text/javascript",
        ".html": "text/html; charset=utf-8",
        ".json": "application/json",
    }

    def end_headers(self) -> None:
        # Always fetch the latest file while the phone is being worked on.
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


def make_server(port: int = 8080, host: str = "127.0.0.1") -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), partial(_PhoneFiles, directory=str(PAGE_DIR)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Serve the web phone page on 127.0.0.1.")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    server = make_server(args.port)
    print(f"web phone: http://127.0.0.1:{server.server_address[1]}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
