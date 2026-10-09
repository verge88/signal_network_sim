from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import run_open5gs_phase_trial as phase_trial
from run_open5gs_phase_trial import (
    FROZEN_DESIGN, PHASE_FRACTIONS, latest_nrf_sample,
    paired_factorial_plan, align_event_to_sample
)
from open5gs_kie_sidecar import sign_report
from verify_live_kie_phase_trial import verify
from aggregate_open5gs_phase_trial import summarize, RUNNERS


@pytest.mark.parametrize("seed", sorted(RUNNERS))
def test_balanced_independent_phase_profile_trial(seed):
    plan = paired_factorial_plan(seed)
    assert plan == paired_factorial_plan(seed)
    assert len(plan) == 12
    assert FROZEN_DESIGN == "prospective-nrf-sampler-phase-v1"
    assert all(p["phase"] in ("truthful_nrf_burst", "hidden_nrf_burst") for p in plan)
    assert len({(p["cycle"], p["phase"]) for p in plan}) == 12
    for profile in ("short", "standard"):
        for fraction in PHASE_FRACTIONS:
            sub = [p for p in plan if p["profile"] == profile
                   and p["assigned_phase_fraction"] == fraction]
            assert len(sub) == 2
            assert {p["phase"] for p in sub} == {
                "truthful_nrf_burst", "hidden_nrf_burst"}


def test_incomplete_or_changed_grid_fails_before_capture():
    with pytest.raises(ValueError, match="six cycles"):
        paired_factorial_plan(42, 3)


def test_sample_timestamp_without_oracle_values(tmp_path):
    path = tmp_path / "kie_reports.jsonl"
    path.write_text(
        json.dumps({"nf_id": "nrf", "ts": 5.0, "fault_injection": "MASK",
                    "phase_hint": "hidden", "reported": {"cpu_ticks_delta": 500}})
        + "\n" + json.dumps({"nf_id": "udm", "ts": 5.2}) + "\n"
    )
    assert latest_nrf_sample(path) == 5.0


def test_sampler_alignment_is_from_actual_signed_sample(monkeypatch, tmp_path):
    clock = [100.0]
    monkeypatch.setattr(phase_trial.time, "time", lambda: clock[0])
    def advance(sec):
        clock[0] += sec
    monkeypatch.setattr(phase_trial.time, "sleep", advance)
    monkeypatch.setattr(
        phase_trial, "wait_for_new_nrf_sample",
        lambda reports, after, period: 104.0
    )
    sample, planned, actual = align_event_to_sample(
        tmp_path / "signed.jsonl", 4.0, 0.5
    )
    assert (sample, planned, actual) == (104.0, 106.0, 106.0)


def test_trial_signature_validation_rejects_tampering(tmp_path):
    reports = tmp_path / "reports.jsonl"
    truth = tmp_path / "truth.jsonl"
    out = tmp_path / "validation.json"
    secret = b"ephemeral-key"
    row = {"ts": 100.1, "nf_id": "nrf", "sequence": 2,
           "reported": {"cpu_ticks_delta": 1}}
    row["signature"] = sign_report(row, secret)
    reports.write_text(json.dumps(row) + "\n")
    truth.write_text(json.dumps({"ts": 100.1, "nf_id": "nrf", "sequence": 2}) + "\n")
    result = verify(reports, truth, secret, out)
    assert result["all_signatures_valid"]
    assert result["report_count"] == result["paired_collector_samples"] == 1
    assert not result["hmac_key_exported"]
    assert "ephemeral-key" not in out.read_text()
    row["reported"]["cpu_ticks_delta"] = 2
    reports.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="invalid signed"):
        verify(reports, truth, secret, out)


def test_trial_signature_validation_rejects_unpaired_samples(tmp_path):
    reports = tmp_path / "reports.jsonl"
    truth = tmp_path / "truth.jsonl"
    row = {"ts": 1, "nf_id": "nrf", "sequence": 1}
    row["signature"] = sign_report(row, b"test")
    reports.write_text(json.dumps(row) + "\n")
    truth.write_text(json.dumps({"ts": 2, "nf_id": "nrf", "sequence": 1}) + "\n")
    with pytest.raises(ValueError, match="pair exactly"):
        verify(reports, truth, b"test", tmp_path / "out.json")


def test_missing_trial_artifacts_cannot_be_pooled(tmp_path):
    with pytest.raises(ValueError, match="expected exactly one artifact"):
        summarize(tmp_path, tmp_path / "out")
