from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from open5gs_kie_sidecar import sign_report, verify_report
from run_open5gs_kie_analysis import load_reports, validate_reports


def test_kie_report_signature_roundtrip():
    payload = {
        "schema": "kie-open5gs-v1",
        "ts": 1.0,
        "sequence": 1,
        "nonce": "abc",
        "nf_id": "udm",
        "unit": "open5gs-udmd.service",
        "phase_hint": "baseline",
        "fault_injection": "none",
        "reported": {
            "pid": 10,
            "invocation_id": "x",
            "restart_generation": 0,
            "service_uptime_s": 12.0,
            "active_state": "active",
            "cpu_ticks_delta": 1,
            "io_chars_delta": 2,
            "ctx_switch_delta": 3,
            "fd_count": 4,
        },
    }
    secret = b"secret"
    report = dict(payload)
    report["signature"] = sign_report(payload, secret)
    assert verify_report(report, secret)
    report["reported"]["pid"] = 11
    assert not verify_report(report, secret)


def test_validation_detects_controlled_mask_divergence(tmp_path: Path):
    secret = b"secret"
    reports_path = tmp_path / "reports.jsonl"
    truth_path = tmp_path / "truth.jsonl"

    payload = {
        "schema": "kie-open5gs-v1",
        "ts": 1.0,
        "sequence": 5,
        "nonce": "abc",
        "nf_id": "udm",
        "unit": "open5gs-udmd.service",
        "phase_hint": "hidden_udm_restart",
        "fault_injection": "baseline_report_mask",
        "reported": {
            "pid": 10,
            "invocation_id": "old",
            "restart_generation": 0,
            "service_uptime_s": 100.0,
            "active_state": "active",
            "cpu_ticks_delta": 1,
            "io_chars_delta": 1,
            "ctx_switch_delta": 1,
            "fd_count": 5,
        },
    }
    report = dict(payload)
    report["signature"] = sign_report(payload, secret)
    reports_path.write_text(json.dumps(report) + "\n", encoding="utf-8")

    truth = {
        "ts": 1.0,
        "sequence": 5,
        "nf_id": "udm",
        "unit": "open5gs-udmd.service",
        "phase_hint": "hidden_udm_restart",
        "masking_active": True,
        "actual": {
            "pid": 11,
            "invocation_id": "new",
            "cpu_ticks": 50,
            "io_chars": 50,
        },
        "truthful_report": {
            "restart_generation": 1,
            "cpu_ticks_delta": 10,
            "io_chars_delta": 10,
            "ctx_switch_delta": 10,
        },
    }
    truth_path.write_text(json.dumps(truth) + "\n", encoding="utf-8")

    reports = load_reports(reports_path, secret)
    truth_df = pd.DataFrame([{
        "ts": 1.0,
        "sequence": 5,
        "nf_id": "udm",
        "masking_active": True,
        "actual_pid": 11,
        "actual_invocation_id": "new",
        "actual_cpu_ticks": 50.0,
        "actual_io_chars": 50.0,
        "truth_generation": 1,
        "truth_cpu_delta": 10.0,
        "truth_io_delta": 10.0,
        "truth_ctx_delta": 10.0,
    }])
    result = validate_reports(reports, truth_df)
    assert result["all_signatures_valid"]
    assert result["masked_ground_truth_pairs"] == 1
    assert result["masked_activity_divergence_fraction"] == 1.0
