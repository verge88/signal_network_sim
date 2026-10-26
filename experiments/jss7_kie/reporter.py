"""Separate KIE agent: signs the node's own report, including configured lies."""
from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .evidence import canonical, sign


def serve(events: Path, key: bytes, run_id: str, cap: int | None, port: int):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            query = parse_qs(urlparse(self.path).query)
            nonce = query.get("nonce", [""])[0]
            if urlparse(self.path).path != "/kie" or not 16 <= len(nonce) <= 128:
                self.send_error(400, "nonce required")
                return
            # Only this agent reads the local node journal. The detector uses PCAP.
            actual = len(set(events.read_text().splitlines())) if events.exists() else 0
            reported = actual if cap is None else min(actual, cap)
            payload = {"schema": 1, "node": "jss7-server", "run_id": run_id,
                       "nonce": nonce, "received_dialogs": reported}
            body = canonical(sign(payload, key))
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    HTTPServer(("127.0.0.1", port), Handler).serve_forever()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", type=Path, required=True)
    ap.add_argument("--key", type=Path, required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--cap", type=int)
    ap.add_argument("--port", type=int, default=18080)
    a = ap.parse_args()
    if a.cap is not None and a.cap < 0:
        ap.error("cap must be nonnegative")
    serve(a.events, a.key.read_bytes(), a.run_id, a.cap, a.port)


if __name__ == "__main__":
    main()
