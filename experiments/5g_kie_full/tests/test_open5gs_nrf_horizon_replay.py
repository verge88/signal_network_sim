from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from replay_open5gs_nrf_horizons import (
    HORIZONS, pair_signed_truth, score_nrf_event,
)


def sample_reports():
    return pd.DataFrame([
        {"ts": 3.0, "nf": "nrf", "cpu_delta": 0.0, "valid_signature": True},
        {"ts": 5.0, "nf": "nrf", "cpu_delta": 0.0, "valid_signature": True},
        {"ts": 10.0, "nf": "nrf", "cpu_delta": 0.0, "valid_signature": True},
        {"ts": 13.5, "nf": "nrf", "cpu_delta": 2.0, "valid_signature": True},
    ])


def test_fixed_reference_and_only_horizon_change():
    reports = sample_reports()
    assert HORIZONS == (2., 3., 4., 5., 6., 8.)
    at_3 = score_nrf_event(10., reports, 3.)
    at_4 = score_nrf_event(10., reports, 4.)
    assert at_3 == {"state": "unacknowledged", "alarm": True, "positive_offset_s": None}
    assert at_4["state"] == "acknowledged"
    assert at_4["alarm"] is False
    assert at_4["positive_offset_s"] == 3.5


def test_no_report_is_unknown_not_an_alarm():
    reports = sample_reports()
    assert score_nrf_event(10., reports[reports.ts > 8.], 4.)["state"] == "insufficient_reports"
    assert score_nrf_event(10., reports[reports.ts > 8.], 4.)["alarm"] is False


def test_signature_invalid_is_not_false_concealment():
    reports = sample_reports()
    reports.loc[reports.ts == 13.5, "valid_signature"] = False
    assert score_nrf_event(10., reports, 4.)["state"] == "invalid_signature"
    assert score_nrf_event(10., reports, 4.)["alarm"] is False


def test_no_oracle_labels_used_by_score():
    reports = sample_reports()
    expected = score_nrf_event(10., reports, 4.)
    for value in ("hidden_nrf_burst", "truthful_nrf_burst"):
        annotated = reports.assign(audit_phase=value, masking_active=True,
                                   fault_injection="baseline_report_mask")
        assert score_nrf_event(10., annotated, 4.) == expected


def pair_rows():
    report = [
        {"nf_id": "nrf", "sequence": 1, "ts": 1.25,
         "fault_injection": "baseline_report_mask"},
        {"nf_id": "nrf", "sequence": 2, "ts": 2.25,
         "fault_injection": "none"},
    ]
    truth = [
        {"nf_id": "nrf", "sequence": 1, "ts": 1.25, "masking_active": True},
        {"nf_id": "nrf", "sequence": 2, "ts": 2.25, "masking_active": False},
    ]
    return report, truth


def test_signed_metadata_and_truth_are_paired_by_exact_sample():
    reports, truth = pair_rows()
    paired = pair_signed_truth(reports, truth)
    assert len(paired) == 2
    assert int(paired.masking_active.sum()) == 1


def test_missing_signed_sample_fails_closed():
    reports, truth = pair_rows()
    with pytest.raises(ValueError, match="missing paired"):
        pair_signed_truth(reports[:1], truth)


def test_false_mask_provenance_fails_closed():
    reports, truth = pair_rows()
    reports[0]["fault_injection"] = "none"
    with pytest.raises(ValueError, match="injection flag"):
        pair_signed_truth(reports, truth)


def test_duplicate_sample_fails_closed():
    reports, truth = pair_rows()
    with pytest.raises(ValueError, match="duplicate sample"):
        pair_signed_truth(reports + [reports[0]], truth)
