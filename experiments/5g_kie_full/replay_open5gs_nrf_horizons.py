"""Exploratory, locked-grid replay of NRF report-window sensitivity.

Uses archived Open5GS observations. All detector outcomes are computed from
external timestamps and signed payload fields BEFORE experiment labels or
mask/exposure metadata are joined. No offline HMAC re-verification is claimed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from run_open5gs_concealment import load_signed_reports

HORIZONS = (2.0, 3.0, 4.0, 5.0, 6.0, 8.0)
CONDITIONS = {
    13007: (1.0, 0.0),
    14009: (1.0, 2.0),
    15013: (4.0, 0.0),
    16001: (4.0, 2.0),
}
KNOWN = {"acknowledged", "unacknowledged"}


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def score_nrf_event(ts: float, signed: pd.DataFrame, horizon_s: float) -> dict:
    """Frozen v1/v2 NRF decision, changing ONLY the post-event cutoff.

    No phase, masking, fault-injection flags, labels or true state enter.
    """
    pre = signed[(signed.ts >= ts - 8.0) & (signed.ts < ts - 2.0)]
    post = signed[(signed.ts >= ts - 1.0) & (signed.ts <= ts + horizon_s)]
    if pre.empty or post.empty:
        return {"state": "insufficient_reports", "alarm": False, "positive_offset_s": None}
    if not bool(pd.concat([pre, post]).valid_signature.all()):
        return {"state": "invalid_signature", "alarm": False, "positive_offset_s": None}
    gain = post.cpu_delta.astype(float) - float(pre.cpu_delta.median())
    positive = post[gain >= 1.0]
    acknowledged = not positive.empty
    return {
        "state": "acknowledged" if acknowledged else "unacknowledged",
        "alarm": not acknowledged,
        "positive_offset_s": float(positive.ts.min() - ts) if acknowledged else None,
    }


def pair_signed_truth(report_rows: list[dict], truth_rows: list[dict]) -> pd.DataFrame:
    """Join evaluation-only mask state to a signed report by exact sample identity."""
    reports = pd.DataFrame([
        {"nf": str(r["nf_id"]).lower(), "sequence": int(r["sequence"]),
         "ts": float(r["ts"]), "report_flag": str(r.get("fault_injection", ""))}
        for r in report_rows
    ])
    truth = pd.DataFrame([
        {"nf": str(r["nf_id"]).lower(), "sequence": int(r["sequence"]),
         "ts": float(r["ts"]), "masking_active": bool(r["masking_active"])}
        for r in truth_rows
    ])
    key = ["nf", "sequence", "ts"]
    if reports.empty or truth.empty:
        raise ValueError("empty signed report or collector truth file")
    if reports.duplicated(key).any() or truth.duplicated(key).any():
        raise ValueError("duplicate sample identity")
    joined = reports.merge(truth, on=key, how="outer", validate="one_to_one", indicator=True)
    if not bool(joined._merge.eq("both").all()):
        raise ValueError("missing paired signed sample or truth record")
    # The experiment-specific flag is signed metadata but MUST NOT be a detector
    # input. Confirm only after decisions that it agrees with sidecar truth.
    flagged = joined.report_flag.eq("baseline_report_mask")
    if not bool((flagged == joined.masking_active).all()):
        raise ValueError("signed injection flag does not match collector truth")
    return joined.drop(columns=["_merge"])


def live_verified_prefix(
    reports: list[dict], truth: list[dict], live_count: int
) -> tuple[list[dict], list[dict]]:
    """Retain only samples covered by live HMAC validation, never silently
    treat later appended reports as authenticated offline.
    """
    if live_count < 1 or live_count > len(reports):
        raise ValueError("invalid HMAC-validated report count")
    verified = reports[:live_count]
    ids = {(str(r["nf_id"]).lower(), int(r["sequence"]), float(r["ts"]))
           for r in verified}
    kept_truth = [r for r in truth if
                  (str(r["nf_id"]).lower(), int(r["sequence"]), float(r["ts"])) in ids]
    if len(kept_truth) != live_count:
        raise ValueError("no collector truth for all HMAC-validated samples")
    return verified, kept_truth


def replay(input_root: Path, output_dir: Path) -> dict:
    event_rows = []
    crosschecks = []
    provenance = []
    for seed, (period, load) in CONDITIONS.items():
        found = list(input_root.rglob(f"5g-open5gs-stress-{seed}"))
        if len(found) != 1:
            raise ValueError(f"missing or duplicate original artifact for seed {seed}: {len(found)}")
        root = found[0]
        manifest = json.loads((root / "stress_manifest.json").read_text())
        if (int(manifest["seed"]) != seed or
            float(manifest["kie_period_s"]) != period or
            float(manifest["background_rps"]) != load):
            raise ValueError(f"frozen condition mismatch: {seed}")
        live = json.loads((root / "kie_report_validation.json").read_text())
        if (not live.get("all_signatures_valid", False) or
            int(live.get("invalid_signature_count", -1)) != 0):
            raise ValueError(f"live HMAC provenance failed: {seed}")
        report_rows = jsonl(root / "kie_reports.jsonl")
        truth_rows = jsonl(root / "kie_ground_truth.jsonl")
        verified_rows, verified_truth = live_verified_prefix(
            report_rows, truth_rows, int(live["report_count"])
        )
        samples = pair_signed_truth(verified_rows, verified_truth)
        # load_signed_reports parses bytes, but cannot reverify HMAC offline.
        # Exclude any appended reports beyond the source-run validated prefix.
        signed_all = load_signed_reports(root / "kie_reports.jsonl", None)
        signed = signed_all.merge(
            samples[["nf", "ts"]], on=["nf", "ts"], how="inner", validate="one_to_one"
        )
        signed_nrf = signed[signed.nf == "nrf"].reset_index(drop=True)
        masked_nrf = samples[samples.nf == "nrf"]
        original = pd.read_csv(root / "concealment_events.csv")
        archived = pd.read_csv(root / "v2_concealment_events.csv")
        key = ["ts", "nf", "external_event"]
        if original.duplicated(key).any() or archived.duplicated(key).any():
            raise ValueError(f"duplicate external witness: {seed}")
        events = archived.merge(
            original[key + ["state", "concealment_alarm"]].rename(columns={
                "state": "archived_v1_state", "concealment_alarm": "archived_v1_alarm"
            }), on=key, validate="one_to_one", how="outer", indicator=True
        )
        if not bool(events._merge.eq("both").all()):
            raise ValueError(f"original and v2 event identities diverged: {seed}")
        events = events[events.nf == "nrf"]
        if not bool(events.audit_phase.isin(("truthful_nrf_burst", "hidden_nrf_burst")).all()):
            raise ValueError(f"unresolved NRF witness in seed {seed}")

        validated = 0
        for event in events.to_dict("records"):
            t = float(event["ts"])
            # All horizon decisions are frozen independently of labels.
            decisions = {h: score_nrf_event(t, signed_nrf, h) for h in HORIZONS}
            v1 = decisions[3.0]
            v2 = decisions[4.0]
            if (v1["state"] != str(event["archived_v1_state"]) or
                bool(v1["alarm"]) != bool(event["archived_v1_alarm"]) or
                v2["state"] != str(event["state"]) or
                bool(v2["alarm"]) != bool(event["v2_alarm"])):
                raise ValueError(f"3/4s frozen decision crosscheck failed: seed={seed} t={t}")
            validated += 1

            # Ground truth is joined ONLY here, after all decisions.
            own_samples = masked_nrf
            for horizon in HORIZONS:
                decision = decisions[horizon]
                window = own_samples[
                    (own_samples.ts >= t - 1.0) & (own_samples.ts <= t + horizon)
                ]
                positive_at = decision["positive_offset_s"]
                first_positive_masked = None
                if positive_at is not None:
                    p = window[window.ts.sub(t + positive_at).abs() < 1e-6]
                    if len(p) == 1:
                        first_positive_masked = bool(p.iloc[0].masking_active)
                    elif len(p) != 1:
                        raise ValueError("acknowledging report is absent from paired samples")
                event_rows.append({
                    "seed": seed, "period_s": period, "background_rps": load,
                    "ts": t, "phase": str(event["audit_phase"]),
                    "cycle": int(event["audit_cycle"]),
                    "horizon_s": horizon, "state": decision["state"],
                    "known": decision["state"] in KNOWN,
                    "alarm": bool(decision["alarm"]),
                    "first_positive_offset_s": positive_at,
                    "first_positive_is_masked": first_positive_masked,
                    "signed_reports_in_window": int(len(window)),
                    "masked_signed_reports_in_window": int(window.masking_active.sum()),
                    "exposed_to_masked_report": bool(window.masking_active.any()),
                })
        crosschecks.append({"seed": seed, "nrf_events_reproduced": validated,
                            "h3_matches_v1": True, "h4_matches_v2": True})
        provenance.append({
            "seed": seed, "live_hmac_verified": bool(live["all_signatures_valid"]),
            "offline_hmac_reverified": False,
            "archived_reports": len(report_rows),
            "live_validated_reports": len(verified_rows),
            "excluded_postvalidation_reports": len(report_rows) - len(verified_rows),
            "paired_samples": len(samples),
        })

    table = pd.DataFrame(event_rows)
    if table.empty:
        raise ValueError("no NRF witness event to replay")
    summary = []
    for (phase, horizon, period, load), group in table.groupby(
        ["phase", "horizon_s", "period_s", "background_rps"], sort=True
    ):
        known = group[group.known]
        summary.append({
            "phase": phase, "horizon_s": horizon,
            "period_s": period, "background_rps": load,
            "witnessed_events": len(group),
            "evaluable": len(known), "unknown": len(group) - len(known),
            "alarms": int(known.alarm.sum()),
            "alarm_fraction_evaluable": (float(known.alarm.mean()) if len(known) else None),
            "mask_exposed_events": int(group.exposed_to_masked_report.sum()),
            "unexposed_events": int((~group.exposed_to_masked_report).sum()),
            "masked_report_count": int(group.masked_signed_reports_in_window.sum()),
            "ack_with_masked_first_positive": int((
                group.first_positive_is_masked == True).sum()),
        })
    output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_dir / "nrf_horizon_events.csv", index=False)
    pd.DataFrame(summary).to_csv(output_dir / "nrf_horizon_by_condition.csv", index=False)
    pd.DataFrame(crosschecks).to_csv(output_dir / "frozen_detector_crosschecks.csv", index=False)
    pd.DataFrame(provenance).to_csv(output_dir / "signed_report_provenance.csv", index=False)

    pooled = []
    for (phase, horizon), group in table.groupby(["phase", "horizon_s"], sort=True):
        known = group[group.known]
        exposed = group[group.exposed_to_masked_report & group.known]
        pooled.append({
            "phase": phase, "horizon_s": horizon,
            "witnessed": len(group), "evaluable": len(known),
            "unknown": len(group) - len(known),
            "alarms": int(known.alarm.sum()),
            "alarm_rate": float(known.alarm.mean()) if len(known) else None,
            "exposed": int(group.exposed_to_masked_report.sum()),
            "exposed_evaluable": len(exposed),
            "exposed_alarms": int(exposed.alarm.sum()),
            "exposed_alarm_rate": float(exposed.alarm.mean()) if len(exposed) else None,
        })
    pd.DataFrame(pooled).to_csv(output_dir / "nrf_horizon_pooled_exploratory.csv", index=False)
    doc = {
        "classification": "EXPLORATORY retrospective offline sensitivity analysis",
        "source_runs": [37908173468, 37820113809],
        "horizons_s": list(HORIZONS),
        "hmac": "verified live in source runs; original keys unavailable offline",
        "detector": "fixed NRF CPU reference t-8..t-2, signed report rise >=1 tick",
        "mask_metadata_used_for_detector": False,
        "notes": [
            "These are NOT independent new Open5GS interventions.",
            "The grid was specified after seeing previous 3s and 4s outcomes.",
            "Report-window exposure is post-hoc descriptive, not a causal subgroup.",
            "Fault-injection metadata in signed payloads is evaluation-only.",
            "Four runners mean rates and Wilson intervals are descriptive; no significance claim.",
            "HMAC verifies source bytes, not a trustworthy or uncompromised NF."
        ],
        "crosschecks": crosschecks,
        "pooled": pooled,
    }
    (output_dir / "nrf_horizon_summary.json").write_text(
        json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return doc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    a = parser.parse_args()
    print(json.dumps(replay(a.input_root, a.output_dir), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
