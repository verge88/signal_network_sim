"""Separate KIE agent: signs the node's own report, including configured lies."""
from __future__ import annotations

import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .evidence import canonical, sign


def serve(events: Path, key: bytes, run_id: str, cap: int | None, port: int, faults: Path | None = None):
    # Faults belong to the test fixture, never to the detector or PCAP observer.
    schedule = json.loads(faults.read_text()) if faults else []
    previous = None
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            nonlocal previous
            query = parse_qs(urlparse(self.path).query)
            nonce = query.get("nonce", [""])[0]
            if urlparse(self.path).path != "/kie" or not 16 <= len(nonce) <= 128:
                self.send_error(400, "nonce required")
                return
            # Only this agent reads the local node journal. The detector uses PCAP.
            actual = len(set(events.read_text().splitlines())) if events.exists() else 0
            reported = actual if cap is None else min(actual, cap)
            sequence = query.get("sequence", [None])[0]
            if sequence is not None:
                try:
                    sequence = int(sequence)
                    if sequence < 0:
                        raise ValueError()
                except ValueError:
                    self.send_error(400, "invalid sequence")
                    return
            fault = schedule[sequence] if sequence is not None and sequence < len(schedule) else "honest"
            if fault == "conceal":
                reported = max(0, actual - 8)
            payload = {"schema": 1, "node": "jss7-server", "run_id": run_id,
                       "nonce": nonce, "received_dialogs": reported}
            if sequence is not None:
                payload["sequence"] = sequence
            report = sign(payload, key)
            if fault == "replay" and previous is not None:
                report = previous
            elif fault == "tamper":
                report["mac"] = "0" * 64
            elif fault == "timeout":
                time.sleep(1.0)
            else:
                previous = report
            body = canonical(report)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass  # Expected when the deadline fault outlives the HTTP client.

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
    ap.add_argument("--faults", type=Path)
    a = ap.parse_args()
    if a.cap is not None and a.cap < 0:
        ap.error("cap must be nonnegative")
    serve(a.events, a.key.read_bytes(), a.run_id, a.cap, a.port, a.faults)


if __name__ == "__main__":
    main()
