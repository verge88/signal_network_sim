"""Label-free KIE consistency check. Ground truth is used only by evaluation."""
from __future__ import annotations

import hashlib
import hmac
import json


def canonical(payload: dict) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def sign(payload: dict, key: bytes) -> dict:
    return {"payload": payload, "mac": hmac.new(key, canonical(payload), hashlib.sha256).hexdigest()}


def verify(report: dict, key: bytes, run_id: str, nonce: str) -> dict:
    payload = report["payload"]
    expected = hmac.new(key, canonical(payload), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, report["mac"]):
        raise ValueError("invalid KIE MAC")
    if payload.get("run_id") != run_id or payload.get("nonce") != nonce:
        raise ValueError("stale or foreign KIE report")
    if payload.get("schema") != 1 or payload.get("node") != "jss7-server":
        raise ValueError("unexpected KIE identity/schema")
    count = payload.get("received_dialogs")
    if type(count) is not int or count < 0:
        raise ValueError("invalid dialog counter")
    return payload


def transaction_ids(text: str) -> set[str]:
    """Deduplicate TCAP IDs, including multiple messages bundled in one frame.

    Valid only for one fresh SCTP association and one direction per capture.
    IDs must not be reused during the short experiment.
    """
    ids = set()
    for line in text.splitlines():
        for value in line.split(","):
            value = value.strip().replace(":", "").lower()
            if not value:
                continue
            if value.startswith("0x"):
                value = value[2:]
            if not value or any(c not in "0123456789abcdef" for c in value):
                raise ValueError("invalid TCAP transaction identifier")
            ids.add(value)
    return ids


def detect(observed_dialogs: int, payload: dict) -> dict:
    if type(observed_dialogs) is not int or observed_dialogs <= 0:
        raise ValueError("empty or invalid independent observation")
    reported = payload["received_dialogs"]
    residual = observed_dialogs - reported
    # Exact accounting after traffic has drained; no trained threshold.
    return {"observed_dialogs": observed_dialogs, "reported_dialogs": reported,
            "residual": residual, "alarm": residual != 0}


def evaluate(rows: list[dict]) -> dict:
    if not rows or not any(r["label"] == 0 for r in rows) or not any(r["label"] == 1 for r in rows):
        raise ValueError("both benign and concealment runs are required")
    tp = sum(r["label"] == 1 and r["alarm"] for r in rows)
    fn = sum(r["label"] == 1 and not r["alarm"] for r in rows)
    fp = sum(r["label"] == 0 and r["alarm"] for r in rows)
    tn = sum(r["label"] == 0 and not r["alarm"] for r in rows)
    return {"tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "episode_recall": tp / (tp + fn), "episode_fpr": fp / (fp + tn),
            "scope": "deterministic drained-window smoke test, not a statistical quality estimate"}
