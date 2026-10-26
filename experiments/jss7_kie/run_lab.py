"""Linux integration runner. Missing SCTP, traffic or evidence is a failure."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import secrets
import signal
import socket
import subprocess
import sys
import tempfile
import time
from urllib.request import urlopen

from .evidence import detect, evaluate, transaction_ids, verify

ROOT = Path(__file__).resolve().parents[2]
MODULE = Path(__file__).resolve().parent
SCENARIOS = (("normal", 20, None, 0), ("legitimate_surge", 40, None, 0),
             ("matched_honest", 30, None, 0), ("concealed_activity", 30, 20, 1))


def command(args, **kwargs):
    return subprocess.run(args, check=True, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, timeout=kwargs.pop("timeout", 30), **kwargs).stdout


def stop(proc, sig=signal.SIGTERM):
    if proc is None or proc.poll() is not None:
        return
    proc.send_signal(sig)
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def wait_text(path, marker, proc, timeout=30):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if proc.poll() is not None:
            raise RuntimeError(f"process exited before {marker}; see {path}")
        if path.exists() and marker in path.read_text(errors="replace"):
            return
        time.sleep(0.1)
    raise TimeoutError(f"missing {marker}; see {path}")


def poll_report(url):
    end = time.monotonic() + 10
    while True:
        try:
            with urlopen(url, timeout=2) as response:
                return json.load(response)
        except OSError:
            if time.monotonic() >= end:
                raise
            time.sleep(0.1)


def observe(pcap: Path, out: Path):
    views = (("begin", "sctp.dstport == 3434 && tcap.begin_element", "tcap.otid"),
             ("end", "sctp.srcport == 3434 && tcap.end_element", "tcap.dtid"))
    evidence = {}
    for name, filt, field in views:
        text = command(["tshark", "-r", str(pcap), "-Y", filt, "-T", "fields",
                        "-e", field, "-E", "occurrence=a", "-E", "aggregator=,"])
        (out / f"{name}_ids.txt").write_text(text)
        evidence[name] = transaction_ids(text)
    if not evidence["begin"] or evidence["begin"] != evidence["end"]:
        raise ValueError("capture is empty or TCAP dialogs did not drain completely")
    return len(evidence["begin"])


def wait_capture(pcap, directory, expected, capture, timeout=10):
    """Wait for the capture process to consume the final completed response.

    Java completion does not imply tcpdump has drained its kernel buffer.
    Never turn incomplete evidence into a valid measurement by sleeping once.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if capture.poll() is not None:
            raise RuntimeError("packet capture exited while draining")
        try:
            if observe(pcap, directory) == expected:
                return
        except (ValueError, subprocess.CalledProcessError):
            pass  # A PCAP still being written can be incomplete.
        time.sleep(0.1)
    raise TimeoutError("capture did not contain all completed dialogs before deadline")


def run_scenario(out, name, count, cap, label, key_path, key, repetition):
    directory = out / f"{repetition:02d}_{name}"
    directory.mkdir()  # Refuse stale evidence.
    events = directory / "node_events.txt"
    run_id, nonce = secrets.token_hex(16), secrets.token_hex(16)
    cp = str(MODULE / "target/classes") + ":" + str(MODULE / "target/dependency/*")
    server = reporter = capture = client = None
    handles = []

    def start(args, log, cwd):
        handle = log.open("w")
        handles.append(handle)
        return subprocess.Popen(args, cwd=cwd, stdout=handle, stderr=subprocess.STDOUT)

    try:
        server = start(["java", "-cp", cp, "lab.StackPeer", "server", str(events)],
                       directory / "server.log", directory)
        wait_text(directory / "server.log", "SERVER_READY", server)
        prefix = [] if os.geteuid() == 0 else ["sudo", "-n"]
        capture = start(prefix + ["tcpdump", "--immediate-mode", "-i", "lo", "-nn", "-s", "0", "-U",
                                  "-w", str(directory / "traffic.pcap"),
                                  "sctp and (port 2345 or port 3434)"],
                        directory / "capture.log", directory)
        wait_text(directory / "capture.log", "listening on", capture)
        agent_args = [sys.executable, "-m", "experiments.jss7_kie.reporter",
                      "--events", str(events), "--key", str(key_path), "--run-id", run_id]
        if cap is not None:
            agent_args += ["--cap", str(cap)]
        reporter = start(agent_args, directory / "reporter.log", ROOT)
        client = start(["java", "-cp", cp, "lab.StackPeer", "client", str(count)],
                       directory / "client.log", directory)
        if client.wait(timeout=120) != 0:
            raise RuntimeError(f"jSS7 client failed: {directory / 'client.log'}")
        if f"COMPLETED={count}" not in (directory / "client.log").read_text():
            raise RuntimeError("missing completed MAP response evidence")
        wait_capture(directory / "traffic.pcap", directory, count, capture)
        # All responses have completed; KIE describes this drained window.
        report = poll_report(f"http://127.0.0.1:18080/kie?nonce={nonce}")
        payload = verify(report, key, run_id, nonce)
        (directory / "kie_report.json").write_text(json.dumps(report, indent=2))
        stop(capture, signal.SIGINT)
        if capture.returncode != 0:
            raise RuntimeError("packet capture failed")
        if "0 packets dropped by kernel" not in (directory / "capture.log").read_text():
            raise RuntimeError("packet capture loss or missing capture statistics")
        observed = observe(directory / "traffic.pcap", directory)
        if observed != count:
            raise ValueError(f"independent capture count {observed} != completed requests {count}")
        # Detector gets no scenario name, label, injected cap or node journal.
        decision = detect(observed, payload)
        row = dict(scenario=name, repetition=repetition, label=label,
                   kie_mac_verified=True, **decision)
        (directory / "result.json").write_text(json.dumps(row, indent=2))
        return row
    finally:
        stop(client)
        stop(capture, signal.SIGINT)
        stop(reporter)
        stop(server)
        for handle in handles:
            handle.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--repetitions", type=int, default=1)
    a = ap.parse_args()
    if platform.system() != "Linux":
        ap.error("real SCTP experiment requires Linux")
    if not 1 <= a.repetitions <= 20:
        ap.error("repetitions must be 1..20")
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    rows = []
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_SCTP):
            pass
        # Refuse occupation by an unrelated lab process before launching peers.
        for port in (2345, 3434):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_SCTP) as sock:
                sock.bind(("127.0.0.1", port))
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 18080))
        git = ["git", "-c", f"safe.directory={ROOT}"]  # This known checkout, including root in WSL.
        provenance = {"git_commit": command(git + ["rev-parse", "HEAD"], cwd=ROOT).strip(),
                      "git_status": command(git + ["status", "--porcelain"], cwd=ROOT),
                      "platform": platform.platform(), "python": sys.version,
                      "jss7_version": "8.0.0-179",
                      "tshark": command(["tshark", "--version"]).splitlines()[0],
                      "java": subprocess.run(["java", "-version"], capture_output=True, text=True, check=True).stderr,
                      "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                        for p in sorted(MODULE.rglob("*"))
                                        if p.is_file() and not any(x in p.parts for x in ("target", ".m2", "__pycache__", "artifacts"))},
                      "jars": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in (MODULE / "target/dependency").glob("*.jar")}}
        if not provenance["jars"]:
            raise ValueError("build the Maven module first")
        (out / "provenance.json").write_text(json.dumps(provenance, indent=2))
        with tempfile.TemporaryDirectory(prefix="jss7-kie-key-") as tmp:
            key = secrets.token_bytes(32)
            key_path = Path(tmp) / "key"
            key_path.write_bytes(key)
            key_path.chmod(0o600)
            for rep in range(a.repetitions):
                for args in SCENARIOS:
                    rows.append(run_scenario(out, *args, key_path, key, rep))
        metrics = evaluate(rows)
        summary = {"status": "passed" if metrics["fp"] == 0 and metrics["fn"] == 0 else "failed",
                   "metrics": metrics, "runs": rows,
                   "claim_boundary": "Two direct jSS7 peers; side-channel KIE; no STP, ML or operator-traffic validation."}
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))
        if summary["status"] != "passed":
            raise SystemExit(1)
    except Exception as exc:
        (out / "summary.json").write_text(json.dumps({"status": "invalid", "error": str(exc), "runs": rows}, indent=2))
        raise


if __name__ == "__main__":
    main()
