"""Live KIE polling with causal counter bounds and explicit report failures.

Observer markers bracket the HTTP request in capture order. Wall-clock jumps
cannot move a packet into another interval. All processes use one host/interface.
"""
from __future__ import annotations

import json
from pathlib import Path
import secrets
import socket
import subprocess
import time
from urllib.error import URLError
from urllib.request import urlopen

from .evidence import detect_interval, transaction_ids, verify
from .run_lab import capture_views, command

# This fixture is passed to the reporter and evaluator, never to the detector.
FAULT_SCHEDULE = ["honest", "honest", "conceal", "conceal", "honest",
                  "replay", "honest", "tamper", "honest", "timeout", "honest"]
POLL_INTERVAL_SECONDS = 1.0
REPORT_DEADLINE_SECONDS = 0.3
MARKER_PORT = 18081


def first_frames(text):
    result = {}
    for line in text.splitlines():
        frame, timestamp, ids = line.split("\t", 2)
        position = int(frame)
        if position <= 0:
            raise ValueError("invalid capture frame number")
        for identity in transaction_ids(ids):
            result[identity] = min(result.get(identity, position), position)
    return result


def extract_trace(pcap, directory, topology):
    trace = {}
    for name, filt, field in capture_views(topology):
        raw = command(["tshark", "-r", str(pcap), "-Y", filt, "-T", "fields",
                       "-e", "frame.number", "-e", "frame.time_epoch", "-e", field,
                       "-E", "occurrence=a", "-E", "aggregator=,"])
        (directory / f"{name}_frame_ids.tsv").write_text(raw)
        trace[name] = first_frames(raw)
    return trace


def observation_bounds(trace, request_frame, response_frame):
    if not 0 < request_frame < response_frame:
        raise ValueError("invalid capture marker order")
    ends = [sum(t < request_frame for t in ids.values())
            for name, ids in trace.items() if name.endswith("end")]
    begins = [sum(t < response_frame for t in ids.values())
              for name, ids in trace.items() if name.endswith("begin")]
    if not ends or not begins:
        raise ValueError("missing capture directions")
    # Use both legs conservatively while traffic is in flight. The full trace
    # must additionally pass exact route/dialog completion checks at shutdown.
    return min(ends), max(begins)


def wait_marker(pcap, capture, token, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if capture.poll() is not None:
            raise RuntimeError("capture exited during live polling")
        try:
            raw = command(["tshark", "-r", str(pcap), "-Y", f"udp.dstport == {MARKER_PORT}",
                           "-T", "fields", "-e", "frame.number", "-e", "udp.payload"])
            for line in raw.splitlines():
                number, payload = line.split("\t", 1)
                if payload.replace(":", "") == token.hex():
                    return int(number)
        except subprocess.CalledProcessError:
            pass  # A concurrently written PCAP may have a partial last record.
        time.sleep(0.05)
    raise TimeoutError("capture did not reach the post-report marker")


def emit_marker():
    token = secrets.token_bytes(16)
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as udp:
        udp.sendto(token, ("127.0.0.1", MARKER_PORT))
    return token


def snapshot(pcap, directory, topology, capture, before_token):
    after_token = emit_marker()
    after_frame = wait_marker(pcap, capture, after_token)
    before_frame = wait_marker(pcap, capture, before_token)
    (directory / "markers.json").write_text(json.dumps({
        "before_payload_hex": before_token.hex(), "before_frame": before_frame,
        "after_payload_hex": after_token.hex(), "after_frame": after_frame}, indent=2))
    # Marker receipt proves capture consumed traffic after the report. Retry
    # only transient PCAP truncation, never missing protocol evidence.
    deadline = time.monotonic() + 10
    while True:
        try:
            return extract_trace(pcap, directory, topology), before_frame, after_frame
        except subprocess.CalledProcessError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.05)


def check_report(report, key, run_id, nonce, sequence, lower, upper):
    try:
        payload = verify(report, key, run_id, nonce)
        if type(payload.get("sequence")) is not int or payload["sequence"] != sequence:
            raise ValueError("wrong report sequence")
    except (KeyError, TypeError, ValueError) as exc:
        return {"status": "invalid_report", "alarm": None, "error": str(exc)}
    return detect_interval(lower, upper, payload)


def evaluate_windows(windows, schedule):
    if len(windows) != len(schedule):
        raise ValueError("incomplete polling schedule")
    expected = {"honest": "consistent", "conceal": "mismatch",
                "replay": "invalid_report", "tamper": "invalid_report", "timeout": "unavailable"}
    checks = [{"sequence": i, "fault": fault, "expected": expected[fault],
               "actual": window["status"], "passed": window["status"] == expected[fault]}
              for i, (window, fault) in enumerate(zip(windows, schedule))]
    counted = [(w, f) for w, f in zip(windows, schedule) if f in ("honest", "conceal")]
    metrics = {"tp": sum(f == "conceal" and w["status"] == "mismatch" for w, f in counted),
               "fn": sum(f == "conceal" and w["status"] != "mismatch" for w, f in counted),
               "fp": sum(f == "honest" and w["status"] == "mismatch" for w, f in counted),
               "tn": sum(f == "honest" and w["status"] == "consistent" for w, f in counted),
               "report_faults_detected": sum(c["passed"] for c in checks if c["fault"] in ("replay", "tamper", "timeout"))}
    return {"status": "passed" if all(c["passed"] for c in checks) else "failed",
            "metrics": metrics, "checks": checks,
            "scope": "controlled live polls, not a statistical accuracy estimate"}


def monitor(directory, topology, capture, client, key, run_id):
    out = directory / "periodic"
    out.mkdir()
    windows = []
    previous_upper = 0
    next_poll = time.monotonic() + POLL_INTERVAL_SECONDS
    for sequence in range(len(FAULT_SCHEDULE)):
        time.sleep(max(0, next_poll - time.monotonic()))
        next_poll = time.monotonic() + POLL_INTERVAL_SECONDS
        if client.poll() is not None:
            raise RuntimeError("traffic ended before the polling schedule")
        nonce = secrets.token_hex(16)
        target = out / f"{sequence:02d}"
        target.mkdir()
        before_token = emit_marker()
        request_ns = time.time_ns()
        request_monotonic = time.monotonic_ns()
        report = None
        try:
            with urlopen(f"http://127.0.0.1:18080/kie?nonce={nonce}&sequence={sequence}",
                         timeout=REPORT_DEADLINE_SECONDS) as response:
                report = json.load(response)
        except (TimeoutError, URLError) as exc:
            if not isinstance(exc, TimeoutError) and not isinstance(exc.reason, TimeoutError):
                raise
        response_ns = time.time_ns()
        latency_ms = (time.monotonic_ns() - request_monotonic) / 1e6
        if client.poll() is not None:
            raise RuntimeError("traffic stopped during a report request")
        trace, first, last = snapshot(directory / "traffic.pcap", target, topology, capture, before_token)
        lower, upper = observation_bounds(trace, first, last)
        if upper <= previous_upper or lower == 0:
            raise ValueError("no independently observed traffic progress between live polls")
        previous_upper = upper
        if report is None:
            decision = {"status": "unavailable", "alarm": None}
        else:
            (target / "report.json").write_text(json.dumps(report, indent=2))
            decision = check_report(report, key, run_id, nonce, sequence, lower, upper)
        window = {"sequence": sequence, "request_ns": request_ns, "response_ns": response_ns,
                  "request_nonce": nonce,
                  "request_frame": first, "response_frame": last,
                  "report_latency_ms": latency_ms,
                  "observed_lower": lower, "observed_upper": upper, **decision}
        windows.append(window)
        (out / "windows.json").write_text(json.dumps(windows, indent=2))
        print(f"LIVE topology={topology} sequence={sequence} status={window['status']}", flush=True)
    return windows


def audit_final(directory, topology, windows):
    final = directory / "periodic" / "final"
    final.mkdir()
    trace = extract_trace(directory / "traffic.pcap", final, topology)
    for window in windows:
        lower, upper = observation_bounds(trace, window["request_frame"], window["response_frame"])
        if (lower, upper) != (window["observed_lower"], window["observed_upper"]):
            raise ValueError("live observation disagrees with final capture; discard live results")
    result = evaluate_windows(windows, FAULT_SCHEDULE)
    result["final_capture_audit"] = "passed"
    (directory / "periodic" / "evaluation.json").write_text(json.dumps(result, indent=2))
    return result
