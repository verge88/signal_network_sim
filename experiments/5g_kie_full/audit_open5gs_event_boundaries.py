"""Replay-only audit of event phase attribution in real Open5GS SBI captures.

This NEVER changes a detector verdict. It uses the original, already-scored
concealment_events.csv, raw HTTP/2 request timestamps, and the orchestration
phase intervals solely to classify experimental labels and exposure.

No signed KIE fields, masking flags, ground truth, labels or phase intervals
enter the detector's decision. Old HMAC keys are unavailable: the saved run's
successful in-run signature validation is retained as provenance, not repeated.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from open5gs_real_features import load_endpoint_map, load_tshark_events

MIN_RAW_REQUESTS = 5
MIN_DOMINANCE = 0.8
PHASES = ("truthful_udm_restart", "hidden_udm_restart",
          "truthful_nrf_burst", "hidden_nrf_burst")


def raw_nrf_get_timestamps(events_path: Path, endpoints_path: Path) -> np.ndarray:
    """The GET packet's own timestamp, not the center of an aggregated bin."""
    df = load_tshark_events(events_path)
    nrf_addrs = {
        addr for addr, nf in load_endpoint_map(endpoints_path).items() if nf == "nrf"
    }
    selected = df[
        df["method"].str.upper().eq("GET")
        & df["path"].str.contains("/nnrf-", regex=False)
        & df["ip_dst"].isin(nrf_addrs)
    ]
    return np.sort(selected["ts"].to_numpy(float))


def _phase_name(row: pd.Series) -> str:
    return str(row["phase"])


def assign_using_packet_evidence(
    event: pd.Series,
    intervals: pd.DataFrame,
    request_ts: np.ndarray,
) -> dict[str, Any]:
    """Associate an *already detected* burst with the dominant physical requests.

    First NRF hot window is [center-1s, center+1s]. Assign only if >=80%
    of first-bin GET packets are in exactly one known phase. Otherwise label
    unknown, never automatically as a negative/truthful control.
    """
    t = float(event["ts"])
    legacy = str(event.get("phase", "unlabeled"))
    if event["nf"] != "nrf":
        return {
            "audit_phase": legacy,
            "audit_cycle": int(event.get("cycle", -1)),
            "audit_status": "journal_event_exact_timestamp",
            "raw_requests": None,
            "assigned_requests": None,
            "phase_fraction": None,
            "first_request_ts": None,
            "last_request_ts": None,
            "timestamp_delta_to_phase_start_s": None,
        }

    bucket = request_ts[(request_ts >= t - 1.000001) & (request_ts < t + 1.000001)]
    count = len(bucket)
    out: dict[str, Any] = {
        "audit_phase": "unresolved",
        "audit_cycle": -1,
        "audit_status": "no_raw_http2_requests" if count == 0 else "ambiguous_boundary",
        "raw_requests": int(count),
        "assigned_requests": 0,
        "phase_fraction": None,
        "first_request_ts": float(bucket[0]) if count else None,
        "last_request_ts": float(bucket[-1]) if count else None,
        "timestamp_delta_to_phase_start_s": None,
    }
    if count < MIN_RAW_REQUESTS:
        out["audit_status"] = "insufficient_raw_requests"
        return out

    candidates = []
    for _, row in intervals.iterrows():
        inside = int(
            ((bucket >= float(row["start_ts"]))
             & (bucket < float(row["end_ts"]))).sum()
        )
        if inside:
            candidates.append((inside, row))

    if not candidates:
        out["audit_status"] = "unassigned_outside_intervals"
        return out

    candidates.sort(key=lambda pair: pair[0], reverse=True)
    best, best_row = candidates[0]
    fraction = best / count
    out["assigned_requests"] = int(best)
    out["phase_fraction"] = float(fraction)
    out["timestamp_delta_to_phase_start_s"] = t - float(best_row["start_ts"])
    if fraction < MIN_DOMINANCE or (
        len(candidates) > 1 and candidates[1][0] == best
    ):
        out["audit_status"] = "ambiguous_multi_phase_window"
        return out

    out["audit_phase"] = _phase_name(best_row)
    out["audit_cycle"] = int(best_row["cycle"])
    if out["audit_phase"] in ("baseline", "washout", "stable_recovery"):
        out["audit_status"] = "independent_background_control"
    elif out["audit_phase"] in ("truthful_nrf_burst", "hidden_nrf_burst"):
        out["audit_status"] = (
            "packet_verified_boundary_relabel"
            if legacy != out["audit_phase"] else "packet_verified_in_phase"
        )
    else:
        # NRF traffic during UDM operations is not equivalent to an NRF
        # burst intervention; do not score this as an NRF truthful/hidden case.
        out["audit_status"] = "cross_nf_event_not_classified"
        out["audit_phase"] = "unresolved"
        out["audit_cycle"] = -1
    return out


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def audit_artifact(artifact_dir: Path, output_dir: Path, seed: int, event_window_only_replay: bool = False) -> dict[str, Any]:
    events = _read_csv(artifact_dir / "concealment_events.csv")
    phases = _read_csv(artifact_dir / "phase_intervals.csv")
    timestamps = raw_nrf_get_timestamps(
        artifact_dir / "sbi_http2_events.tsv",
        artifact_dir / "open5gs_endpoints.json",
    )
    old_summary = json.loads(
        (artifact_dir / "concealment_summary.json").read_text(encoding="utf-8")
    )
    if old_summary.get("signature_check") != "HMAC_verified":
        raise ValueError("Original event scorer did not enable live HMAC checks")

    validation_path = artifact_dir / "kie_report_validation.json"
    if validation_path.exists():
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        if not validation.get("all_signatures_valid", False):
            raise ValueError("Original run had invalid KIE signatures")
        signature_provenance = "live_all_reports_valid_original_run_not_rechecked"
    else:
        # The v2 collection workflow wrote event-level v1 decisions but
        # omitted the separate all-reports validator. The old event scorer
        # required HMAC-valid pre/post reports to make a decision, so only
        # those event windows have prior integrity evidence.
        if not event_window_only_replay:
            raise FileNotFoundError(validation_path)
        if "integrity_alarm" not in events or bool(events["integrity_alarm"].any()):
            raise ValueError("Archived event-level integrity is insufficient")
        signature_provenance = (
            "original_live_event_windows_only_not_all_reports_or_rechecked"
        )

    additions = [
        assign_using_packet_evidence(row, phases, timestamps)
        for _, row in events.iterrows()
    ]
    audited = pd.concat(
        [events.reset_index(drop=True), pd.DataFrame(additions)], axis=1
    )
    audited["seed"] = int(seed)
    audited["eligible_as_confirmed_exposure"] = (
        audited["audit_phase"].isin(PHASES)
        & audited["state"].isin(["acknowledged", "unacknowledged"])
    )
    audited["legacy_unlabeled"] = events["phase"].eq("unlabeled").to_numpy()
    # Important: an HMAC decision remains exactly as stored; we only reassign
    # EVALUATION labels using timestamps of actual packets.
    if not np.array_equal(audited["concealment_alarm"], events["concealment_alarm"]):
        raise AssertionError("replay altered a detector decision")
    output_dir.mkdir(parents=True, exist_ok=True)
    audited.to_csv(output_dir / "boundary_audit_events.csv", index=False)

    expected = phases.loc[phases["phase"].isin(PHASES)].copy()
    coverage = []
    for _, target in expected.iterrows():
        linked = audited[
            (audited["audit_phase"] == target["phase"])
            & (audited["audit_cycle"] == target["cycle"])
        ]
        coverage.append({
            "seed":seed,
            "phase":target["phase"],
            "cycle":int(target["cycle"]),
            "planned_start":float(target["start_ts"]),
            "planned_end":float(target["end_ts"]),
            "witnessed_events":int(len(linked)),
            "fully_evaluable_events":int(linked["eligible_as_confirmed_exposure"].sum()),
            "missing_external_witness":bool(linked.empty),
            "fragmented_duplicate_witness":bool(len(linked)>1),
            "concealment_alarm":bool(linked["concealment_alarm"].any()),
        })
    cov_df = pd.DataFrame(coverage)
    cov_df.to_csv(output_dir / "boundary_intervention_coverage.csv", index=False)

    phase_rows = []
    for phase in PHASES:
        group = audited[
            (audited["audit_phase"] == phase)
            & audited["eligible_as_confirmed_exposure"]
        ]
        n=int(len(group))
        alarms=int(group["concealment_alarm"].sum())
        phase_rows.append({
            "seed":seed, "phase":phase,"evaluable_events":n,
            "concealment_alarms":alarms,"rate":alarms/n if n else None,
        })
    pd.DataFrame(phase_rows).to_csv(
        output_dir / "boundary_audit_phase_metrics.csv", index=False
    )

    original_unlabeled=audited[audited["legacy_unlabeled"]].copy()
    background=audited[
        audited["audit_status"].eq("independent_background_control")
    ]
    unresolved=audited[audited["audit_phase"].eq("unresolved")]
    summary = {
        "seed":seed,
        "replay_of_stored_decisions":True,
        "signature_verification":signature_provenance,
        "raw_nrf_get_packets":len(timestamps),
        "event_count":len(audited),
        "nrf_event_count":int(audited["nf"].eq("nrf").sum()),
        "legacy_unlabeled_count":len(original_unlabeled),
        "legacy_unlabeled_alarms":int(original_unlabeled["concealment_alarm"].sum()),
        "legacy_unlabeled_reassigned":int(
            original_unlabeled["audit_phase"].isin(PHASES).sum()
        ),
        "legacy_unlabeled_unresolved":int(
            original_unlabeled["audit_phase"].eq("unresolved").sum()
        ),
        "background_events":len(background),
        "background_alarms":int(background["concealment_alarm"].sum()),
        "unresolved_events":len(unresolved),
        "unresolved_alarms":int(unresolved["concealment_alarm"].sum()),
        "planned_interventions":len(expected),
        "planned_missing_external":int(cov_df["missing_external_witness"].sum()),
        "planned_duplicate_fragmented":int(cov_df["fragmented_duplicate_witness"].sum()),
        "phase_metrics":phase_rows,
        "notes":[
            "New phase labels use raw GET packet times only, after detector scoring.",
            "An event not uniquely attributable with >=80% packet dominance remains unresolved.",
            "Neither replay nor event matching re-verifies HMAC signatures (key was ephemeral).",
            "A background alarm is an observed negative-control alert but not necessarily a proven false accusation.",
            "This replay is exploratory/post-hoc; a future run must prospectively confirm packet-based attribution.",
        ],
    }
    (output_dir / "boundary_audit_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser=argparse.ArgumentParser(description="Audit phase boundaries without changing KIE detector decisions")
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--event-window-only-replay", action="store_true",
        help="Allow a missing all-reports validator only when live event-window checks are recorded",
    )
    args=parser.parse_args()
    summary=audit_artifact(
        args.artifact_dir,args.output_dir,args.seed,args.event_window_only_replay
    )
    print(json.dumps(summary,indent=2,ensure_ascii=False))


if __name__=="__main__":
    main()
