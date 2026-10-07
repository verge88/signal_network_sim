from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import secrets
import subprocess
import time
from pathlib import Path
from typing import Any


DEFAULT_UNITS = {
    "udm": "open5gs-udmd.service",
    "nrf": "open5gs-nrfd.service",
}


def _systemctl_show(unit: str) -> dict[str, str]:
    props = [
        "MainPID",
        "ActiveState",
        "SubState",
        "NRestarts",
        "InvocationID",
        "ActiveEnterTimestampMonotonic",
        "ExecMainStartTimestampMonotonic",
    ]
    cmd = ["systemctl", "show", unit]
    for prop in props:
        cmd.extend(["-p", prop])
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    out: dict[str, str] = {}
    for line in proc.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            out[key] = value
    return out


def _read_proc(pid: int) -> dict[str, float | int]:
    if pid <= 0:
        return {
            "cpu_ticks": 0,
            "io_chars": 0,
            "ctx_switches": 0,
            "fd_count": 0,
            "start_ticks": 0,
        }

    cpu_ticks = 0
    start_ticks = 0
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().split()
        cpu_ticks = int(fields[13]) + int(fields[14])
        start_ticks = int(fields[21])
    except Exception:
        pass

    io_chars = 0
    try:
        io = {}
        for line in Path(f"/proc/{pid}/io").read_text().splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                io[key.strip()] = int(value.strip())
        io_chars = int(io.get("rchar", 0)) + int(io.get("wchar", 0))
    except Exception:
        pass

    ctx_switches = 0
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("voluntary_ctxt_switches:") or line.startswith(
                "nonvoluntary_ctxt_switches:"
            ):
                ctx_switches += int(line.split(":", 1)[1].strip())
    except Exception:
        pass

    fd_count = 0
    try:
        fd_count = len(list(Path(f"/proc/{pid}/fd").iterdir()))
    except Exception:
        pass

    return {
        "cpu_ticks": cpu_ticks,
        "io_chars": io_chars,
        "ctx_switches": ctx_switches,
        "fd_count": fd_count,
        "start_ticks": start_ticks,
    }


def collect_local_state(nf_id: str, unit: str) -> dict[str, Any]:
    props = _systemctl_show(unit)
    try:
        pid = int(props.get("MainPID", "0") or 0)
    except ValueError:
        pid = 0
    proc = _read_proc(pid)

    now_mono_us = int(time.monotonic() * 1_000_000)
    try:
        active_enter = int(props.get("ActiveEnterTimestampMonotonic", "0") or 0)
    except ValueError:
        active_enter = 0

    return {
        "nf_id": nf_id,
        "unit": unit,
        "pid": pid,
        "active_state": props.get("ActiveState", "unknown"),
        "sub_state": props.get("SubState", "unknown"),
        "systemd_nrestarts": int(props.get("NRestarts", "0") or 0),
        "invocation_id": props.get("InvocationID", ""),
        "service_uptime_s": (
            max(0.0, (now_mono_us - active_enter) / 1_000_000.0)
            if active_enter > 0
            else 0.0
        ),
        **proc,
    }


def _read_control(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {"mask_nfs": []}
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"mask_nfs": []}
    if not isinstance(doc, dict):
        return {"mask_nfs": []}
    return doc


def _canonical_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def sign_report(payload: dict[str, Any], secret: bytes) -> str:
    return hmac.new(secret, _canonical_bytes(payload), hashlib.sha256).hexdigest()


def verify_report(report: dict[str, Any], secret: bytes) -> bool:
    signature = str(report.get("signature", ""))
    payload = {k: v for k, v in report.items() if k != "signature"}
    expected = sign_report(payload, secret)
    return hmac.compare_digest(signature, expected)


def _reported_fields(
    actual: dict[str, Any],
    previous_actual: dict[str, Any] | None,
    generation: int,
) -> dict[str, Any]:
    prev = previous_actual or {}
    return {
        "pid": actual["pid"],
        "invocation_id": actual["invocation_id"],
        "restart_generation": generation,
        "service_uptime_s": float(actual["service_uptime_s"]),
        "active_state": actual["active_state"],
        "cpu_ticks_delta": max(
            0, int(actual["cpu_ticks"]) - int(prev.get("cpu_ticks", actual["cpu_ticks"]))
        ),
        "io_chars_delta": max(
            0, int(actual["io_chars"]) - int(prev.get("io_chars", actual["io_chars"]))
        ),
        "ctx_switch_delta": max(
            0,
            int(actual["ctx_switches"])
            - int(prev.get("ctx_switches", actual["ctx_switches"])),
        ),
        "fd_count": int(actual["fd_count"]),
    }


def _write_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
        fh.flush()


def run_sidecar(
    reports_path: Path,
    ground_truth_path: Path,
    secret: bytes,
    units: dict[str, str],
    interval_s: float,
    control_file: Path | None,
) -> None:
    previous: dict[str, dict[str, Any]] = {}
    generation = {nf: 0 for nf in units}
    frozen_truthful: dict[str, dict[str, Any]] = {}
    sequence = 0

    while True:
        started = time.time()
        control = _read_control(control_file)
        mask_nfs = {str(x).lower() for x in control.get("mask_nfs", [])}
        phase = str(control.get("phase", "unspecified"))

        for nf_id, unit in units.items():
            actual = collect_local_state(nf_id, unit)
            prev = previous.get(nf_id)

            if prev is not None:
                if (
                    actual["pid"] != prev.get("pid")
                    or actual["invocation_id"] != prev.get("invocation_id")
                ):
                    generation[nf_id] += 1

            truthful = _reported_fields(actual, prev, generation[nf_id])
            if nf_id not in mask_nfs:
                frozen_truthful[nf_id] = dict(truthful)

            masking = nf_id in mask_nfs and nf_id in frozen_truthful
            reported = dict(frozen_truthful[nf_id]) if masking else dict(truthful)

            ts = time.time()
            nonce = secrets.token_hex(12)
            payload = {
                "schema": "kie-open5gs-v1",
                "ts": ts,
                "sequence": sequence,
                "nonce": nonce,
                "nf_id": nf_id,
                "unit": unit,
                "phase_hint": phase,
                "fault_injection": "baseline_report_mask" if masking else "none",
                "reported": reported,
            }
            report = dict(payload)
            report["signature"] = sign_report(payload, secret)
            _write_jsonl(reports_path, report)

            truth_row = {
                "ts": ts,
                "sequence": sequence,
                "nf_id": nf_id,
                "unit": unit,
                "phase_hint": phase,
                "masking_active": bool(masking),
                "actual": actual,
                "truthful_report": truthful,
            }
            _write_jsonl(ground_truth_path, truth_row)
            previous[nf_id] = actual

        sequence += 1
        elapsed = time.time() - started
        time.sleep(max(0.05, interval_s - elapsed))


def _parse_units(values: list[str]) -> dict[str, str]:
    units = dict(DEFAULT_UNITS)
    for item in values:
        if "=" not in item:
            raise ValueError(f"unit must be nf=systemd.service, got {item!r}")
        nf, unit = item.split("=", 1)
        units[nf.strip().lower()] = unit.strip()
    return units


def main() -> None:
    p = argparse.ArgumentParser(description="Signed local Open5GS KIE sidecar")
    p.add_argument("--reports", type=Path, required=True)
    p.add_argument("--ground-truth", type=Path, required=True)
    p.add_argument("--control-file", type=Path)
    p.add_argument("--interval-s", type=float, default=2.0)
    p.add_argument("--unit", action="append", default=[])
    p.add_argument(
        "--secret-env",
        default="OPEN5GS_KIE_SECRET",
        help="environment variable containing the HMAC key",
    )
    args = p.parse_args()

    secret_text = os.environ.get(args.secret_env, "")
    if not secret_text:
        raise SystemExit(f"missing HMAC secret in environment variable {args.secret_env}")
    run_sidecar(
        reports_path=args.reports,
        ground_truth_path=args.ground_truth,
        secret=secret_text.encode("utf-8"),
        units=_parse_units(args.unit),
        interval_s=args.interval_s,
        control_file=args.control_file,
    )


if __name__ == "__main__":
    main()
