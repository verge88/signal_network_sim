"""NF-level causal KIE corroboration with sample-cadence-aware timing.

Version 1's NRF acknowledgment cutoff t+3 s could exclude the first valid
2-second sidecar sample that integrates an observed HTTP/2 burst. In v2, use
t+4 s, derived from the 2-second SBI bucket plus one 2-second sidecar
sampling period. All thresholds and event definitions remain unchanged.

The detector NEVER reads phase, fault_injection, cycle, mask or ground truth.
Archived audit labels are merged only after every event has been scored.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from run_open5gs_concealment import (
    evaluate_one_event,
    load_signed_reports,
)

NRF_ACK_LAG_S = 4.0
NRF_REFERENCE_START_S = 8.0
NRF_REFERENCE_STOP_S = 2.0
NRF_PRE_ACK_S = 1.0
NRF_CPU_TICK_MIN_RISE = 1.0
KNOWN_PHASES = (
    "truthful_udm_restart",
    "hidden_udm_restart",
    "truthful_nrf_burst",
    "hidden_nrf_burst",
)


def score_event_v2(event: dict, kie: pd.DataFrame) -> dict:
    """HMAC check precedes decision; UDM semantic rule is unchanged."""
    nf = str(event["nf"])
    if nf != "nrf":
        old = evaluate_one_event(event, kie)
        old["v2_window_end_offset_s"] = float(old["report_lag_s"])
        return old

    t = float(event["ts"])
    sidecar = kie[kie["nf"] == "nrf"]
    pre = sidecar[
        (sidecar["ts"] >= t - NRF_REFERENCE_START_S)
        & (sidecar["ts"] < t - NRF_REFERENCE_STOP_S)
    ]
    post = sidecar[
        (sidecar["ts"] >= t - NRF_PRE_ACK_S)
        & (sidecar["ts"] <= t + NRF_ACK_LAG_S)
    ]
    output = dict(event)
    output.update({
        "state": "unknown",
        "concealment_alarm": False,
        "integrity_alarm": False,
        "report_response": None,
        "pre_reports": int(len(pre)),
        "post_reports": int(len(post)),
        "report_lag_s": NRF_ACK_LAG_S,
        "v2_window_end_offset_s": NRF_ACK_LAG_S,
        "cpu_response_ticks": None,
        "first_positive_kie_offset_s": None,
    })
    if pre.empty or post.empty:
        output["state"] = "insufficient_reports"
        return output
    if not bool(pd.concat([pre, post])["valid_signature"].all()):
        output["state"] = "invalid_signature"
        output["integrity_alarm"] = True
        return output

    reference = float(pre["cpu_delta"].median())
    gain = post["cpu_delta"].astype(float) - reference
    response = gain >= NRF_CPU_TICK_MIN_RISE
    responded = bool(response.any())
    output["cpu_response_ticks"] = float(gain.max())
    if responded:
        first = float(post.loc[response, "ts"].min())
        output["first_positive_kie_offset_s"] = first - t
    output["report_response"] = responded
    output["state"] = "acknowledged" if responded else "unacknowledged"
    output["concealment_alarm"] = not responded
    return output


def run(
    original_events_path: Path,
    boundary_audit_path: Path,
    reports_path: Path,
    secret: bytes | None,
    output_dir: Path,
    seed: int,
) -> dict:
    original = pd.read_csv(original_events_path)
    audited = pd.read_csv(boundary_audit_path)
    kie = load_signed_reports(reports_path, secret)
    # Intentionally only select NF/timestamp/external evidence from archived
    # events. No experiment label is ever passed into score_event_v2.
    basic = original[["ts", "nf", "external_event", "nf_instance_id"]]
    evaluated = pd.DataFrame([
        score_event_v2(record, kie) for record in basic.to_dict("records")
    ])
    if len(evaluated) != len(original):
        raise AssertionError("event count changed between detector variants")
    if not np.array_equal(
        evaluated["ts"].to_numpy(float), original["ts"].to_numpy(float)
    ):
        raise AssertionError("event selection/timestamp changed")

    key = ["ts", "nf", "external_event"]
    if audited.duplicated(key).any():
        raise ValueError("nonunique archived audit event identity")
    labels = audited[key + [
        "audit_phase", "audit_cycle", "audit_status",
        "eligible_as_confirmed_exposure", "legacy_unlabeled",
    ]]
    evaluated = evaluated.merge(labels, on=key, how="left", validate="one_to_one")
    if evaluated["audit_phase"].isna().any():
        raise ValueError("event was not represented in the phase audit")
    evaluated["seed"] = seed
    evaluated["v1_alarm"] = original["concealment_alarm"].astype(bool).to_numpy()
    evaluated["v2_alarm"] = evaluated["concealment_alarm"].astype(bool)
    evaluated["alarm_changed"] = (
        evaluated["v1_alarm"] != evaluated["v2_alarm"]
    )

    metrics = []
    for phase in KNOWN_PHASES:
        g = evaluated[
            (evaluated["audit_phase"] == phase)
            & (evaluated["eligible_as_confirmed_exposure"])
        ]
        n = len(g)
        v1 = int(g["v1_alarm"].sum())
        v2 = int(g["v2_alarm"].sum())
        metrics.append({
            "seed": seed,
            "phase": phase,
            "evaluable_events": int(n),
            "v1_concealment_alarms": v1,
            "v2_concealment_alarms": v2,
            "v1_rate": v1 / n if n else None,
            "v2_rate": v2 / n if n else None,
            "alarm_changes": int(g["alarm_changed"].sum()),
            "unknown_events": int(
                (~g["state"].isin(["acknowledged", "unacknowledged"])).sum()
            ),
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    evaluated.to_csv(output_dir / "v2_concealment_events.csv", index=False)
    pd.DataFrame(metrics).to_csv(
        output_dir / "v2_concealment_phase_metrics.csv", index=False
    )
    changes = evaluated[evaluated["alarm_changed"]]
    changes.to_csv(output_dir / "v2_alarm_deltas.csv", index=False)

    summary = {
        "seed": seed,
        "signature_check": ("LIVE_HMAC_VERIFIED" if secret is not None else "NOT_REVERIFIED_OFFLINE"),
        "detector_provenance": {
            "nrF_cutoff_previous_s": 3.0,
            "nrF_cutoff_v2_s": NRF_ACK_LAG_S,
            "sidecar_interval_s": 2.0,
            "http2_window_s": 2.0,
            "cpu_tick_min_rise": NRF_CPU_TICK_MIN_RISE,
            "udm_lifecycle_rule_unchanged": True,
            "event_witness_definitions_unchanged": True,
        },
        "total_external_events": int(len(evaluated)),
        "alarm_change_events": int(len(changes)),
        "phase_metrics": metrics,
        "limitations": [
            "The lag adjustment was motivated by an error in the earlier prospective run; performance on that run is exploratory only.",
            "Offline replay checks no HMAC: the unknown sidecar key was not archived, and report-level valid_signature is a placeholder only.",
            "No phase/fault-injection tags or ground truth are inputs to the detector.",
            "A 4-second window can overlap report-mask recovery for very short NRF bursts; such a regression must be measured in new randomized testbeds.",
            "Observed CPU ticks are not a request accounting counter; silence does not cryptographically prove concealment.",
            "HMAC verifies signed bytes, not that the signing agent is trustworthy.",
        ],
    }
    (output_dir / "v2_concealment_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="Compare legacy and lag-aware NRF concealment on same real observed events")
    p.add_argument("--original-events", type=Path, required=True)
    p.add_argument("--boundary-audit", type=Path, required=True)
    p.add_argument("--reports", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument(
        "--offline-no-verify", action="store_true",
        help="Explicit offline artifact replay; HMAC secret unavailable and NOT re-verified",
    )
    args = p.parse_args()
    value = os.environ.get("OPEN5GS_KIE_SECRET", "")
    if not args.offline_no_verify and not value:
        raise SystemExit("runner-only OPEN5GS_KIE_SECRET is required for live verification")
    if args.offline_no_verify and value:
        raise SystemExit("refusing offline bypass with a live verification key available")
    result = run(
        args.original_events, args.boundary_audit, args.reports,
        None if args.offline_no_verify else value.encode(), args.output_dir, args.seed
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
