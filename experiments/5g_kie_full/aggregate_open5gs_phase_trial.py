"""Prospective NRF sampler-phase trial: independent-run descriptive audit.

Model decisions already frozen in original v1/v2 files; randomized phase,
scenario truth, profile, and collector mask state enter ONLY this evaluator.
No pooling of within-run observations as independent sampling units.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from run_open5gs_phase_trial import (
    FROZEN_DESIGN, PHASE_FRACTIONS, paired_factorial_plan
)

RUNNERS = {
    # Independent original prospective runners.
    21011: 1.0, 21013: 1.0,
    22011: 2.0, 22013: 2.0,
    24011: 4.0, 24013: 4.0,
    # Fresh, independently randomized replication runners.
    31011: 1.0, 31013: 1.0,
    32011: 2.0, 32013: 2.0,
    34011: 4.0, 34013: 4.0,
}
CATEGORIES = ("truthful_nrf_burst", "hidden_nrf_burst")


def _truth(path: Path) -> pd.DataFrame:
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()
            if x.strip()]
    if not rows:
        raise ValueError("no collector samples")
    return pd.DataFrame([{
        "nf": str(r["nf_id"]).lower(), "ts": float(r["ts"]),
        "masking_active": bool(r["masking_active"])
    } for r in rows])


def summarize(input_root: Path, output_dir: Path) -> dict:
    exposures = []
    coverage = []
    provenance = []
    for seed, period in sorted(RUNNERS.items()):
        folders = list(input_root.rglob(f"5g-open5gs-phase-{seed}"))
        if len(folders) != 1:
            raise ValueError(f"expected exactly one artifact for {seed}, got {len(folders)}")
        directory = folders[0]
        manifest = json.loads((directory / "phase_trial_manifest.json").read_text())
        if (manifest["design"] != FROZEN_DESIGN or manifest["seed"] != seed or
            float(manifest["kie_period_s"]) != period or
            manifest["ordered_interventions"] != paired_factorial_plan(seed)):
            raise ValueError(f"randomization protocol mismatch for seed {seed}")
        verify = json.loads((directory / "kie_report_validation.json").read_text())
        if (not verify["all_signatures_valid"] or verify["invalid_signature_count"] != 0 or
            not verify["all_reports_truth_paired"]):
            raise ValueError(f"live HMAC or collector pairing failed in seed {seed}")
        provenance.append({
            "seed": seed, "period_s": period, "hmac_checked_live": True,
            "signed_reports": int(verify["report_count"]),
            "matched_collector_samples": int(verify["paired_collector_samples"]),
            "independent_runner": True,
        })
        planned = pd.DataFrame(manifest["ordered_interventions"])
        observed_plan = pd.read_csv(directory / "randomized_alignment.csv")
        keys = ["cycle", "phase"]
        if (planned.duplicated(keys).any() or observed_plan.duplicated(keys).any() or
            len(planned) != 12 or len(observed_plan) != 12):
            raise ValueError(f"incomplete/duplicate intervention plan: {seed}")
        joined = observed_plan.merge(
            planned, on=keys, how="outer", validate="one_to_one",
            suffixes=("", "_planned"), indicator=True
        )
        if (not joined._merge.eq("both").all() or
            not joined.profile.eq(joined.profile_planned).all() or
            not joined.assigned_phase_fraction.eq(joined.assigned_phase_fraction_planned).all()):
            raise ValueError(f"realized trial mismatch: {seed}")

        original = pd.read_csv(directory / "concealment_events.csv")
        v2 = pd.read_csv(directory / "v2_concealment_events.csv")
        key = ["ts", "nf", "external_event"]
        if original.duplicated(key).any() or v2.duplicated(key).any():
            raise ValueError(f"duplicate independently witnessed events: {seed}")
        merged = v2.merge(
            original[key + ["state", "concealment_alarm"]].rename(
                columns={"state": "v1_state", "concealment_alarm": "v1_alarm_original"}
            ), how="outer", on=key, validate="one_to_one", indicator=True
        )
        if not merged._merge.eq("both").all():
            raise ValueError(f"v1/v2 witness mismatch: {seed}")
        if not merged.v1_alarm.eq(merged.v1_alarm_original).all():
            raise ValueError(f"v1 decisions altered: {seed}")
        merged = merged[(merged.nf == "nrf") & merged.audit_phase.isin(CATEGORIES)].copy()
        truth = _truth(directory / "kie_ground_truth.jsonl")
        truth = truth[truth.nf.eq("nrf")]
        intervals = pd.read_csv(directory / "phase_intervals.csv")
        burst = pd.read_csv(directory / "burst_profiles.csv")
        if len(burst) != 12:
            raise ValueError(f"incomplete NRF stimulus profile records: {seed}")
        if not burst.requested.eq(burst.successful).all():
            raise ValueError(f"incomplete delivered NRF stimulus in {seed}")
        if len(intervals[intervals.phase.isin(CATEGORIES)]) != 12:
            raise ValueError(f"incomplete intervention timing manifest: {seed}")

        for trial in joined.to_dict("records"):
            selected = merged[
                (merged.audit_phase == trial["phase"]) &
                (merged.audit_cycle == trial["cycle"])
            ]
            alignment_overrun = bool(trial["phase_alignment_overrun"])
            assigned_fraction = float(trial["assigned_phase_fraction"])
            realized_fraction = float(trial["actual_phase_fraction"])
            if abs(realized_fraction - (
                (float(trial["actual_start_ts"]) - float(trial["reference_signed_nrf_ts"])) / period
            )) > 0.00001:
                raise ValueError(f"tampered realized phase in seed {seed}")
            # Availability and exact packet evidence remain separate from trial allocation.
            coverage.append({
                "seed": seed, "period_s": period, "cycle": int(trial["cycle"]),
                "phase": trial["phase"], "profile": trial["profile"],
                "assigned_phase_fraction": assigned_fraction,
                "actual_phase_fraction": realized_fraction,
                "alignment_overrun": alignment_overrun,
                "external_witness_count": int(len(selected)),
                "complete_single_witness": len(selected) == 1,
            })
            for row in selected.to_dict("records"):
                t = float(row["ts"])
                samples = truth[(truth.ts >= t - 1) & (truth.ts <= t + 4)]
                exposed = bool(samples.masking_active.any())
                exposures.append({
                    "seed": seed, "period_s": period,
                    "cycle": int(trial["cycle"]), "profile": trial["profile"],
                    "phase": trial["phase"],
                    "assigned_phase_fraction": assigned_fraction,
                    "actual_phase_fraction": realized_fraction,
                    "alignment_overrun": alignment_overrun,
                    "event_ts": t,
                    "v1_state": row["v1_state"], "v2_state": row["state"],
                    "v1_alarm": bool(row["v1_alarm"]),
                    "v2_alarm": bool(row["v2_alarm"]),
                    "v1_known": row["v1_state"] in ("acknowledged", "unacknowledged"),
                    "v2_known": row["state"] in ("acknowledged", "unacknowledged"),
                    "report_samples_in_window": len(samples),
                    "masked_samples_in_window": int(samples.masking_active.sum()),
                    "masked_sample_exposure": exposed,
                })
    out = pd.DataFrame(exposures)
    cov = pd.DataFrame(coverage)
    if len(cov) != 12*len(RUNNERS):
        raise ValueError(f"expected {12*len(RUNNERS)} interventions, got {len(cov)}")
    output_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_dir / "trial_events.csv", index=False)
    cov.to_csv(output_dir / "trial_intervention_coverage.csv", index=False)
    pd.DataFrame(provenance).to_csv(output_dir / "runner_signature_provenance.csv", index=False)

    metrics = []
    if not out.empty:
        for (period, profile, fraction, phase), group in out.groupby(
            ["period_s", "profile", "assigned_phase_fraction", "phase"]
        ):
            # Note: if duplicate witnesses existed, coverage table retains that
            # condition; rates per witness still descriptive, not ITT rates.
            for detector in ("v1", "v2"):
                evaluable = group[group[detector + "_known"]]
                metrics.append({
                    "period_s": period, "profile": profile,
                    "assigned_phase_fraction": fraction, "phase": phase,
                    "detector": detector,
                    "witnessed": len(group),
                    "evaluable": len(evaluable), "unknown": len(group)-len(evaluable),
                    "alarms": int(evaluable[detector + "_alarm"].sum()),
                    "alarm_rate_on_evaluable": (
                        float(evaluable[detector + "_alarm"].mean())
                        if len(evaluable) else None
                    ),
                    "mask_exposed_events": int(group.masked_sample_exposure.sum()),
                    "phase_alignment_overruns": int(group.alignment_overrun.sum()),
                    "independent_runner_count": int(group.seed.nunique()),
                })
    pd.DataFrame(metrics).to_csv(output_dir / "descriptive_cells.csv", index=False)

    runner = []
    for (seed, period, profile, phase), group in out.groupby(
        ["seed", "period_s", "profile", "phase"]
    ):
        for detector in ("v1", "v2"):
            valid = group[group[detector + "_known"]]
            runner.append({
                "seed": seed, "period_s": period, "profile": profile,
                "phase": phase, "detector": detector,
                "witnessed": len(group), "known": len(valid),
                "alarms": int(valid[detector + "_alarm"].sum()),
                "rate": float(valid[detector + "_alarm"].mean()) if len(valid) else None,
            })
    pd.DataFrame(runner).to_csv(output_dir / "runner_level_metrics.csv", index=False)
    summary = {
        "classification": "independent replication of prospective randomized phase trial",
        "design": FROZEN_DESIGN, "runners_per_cadence": 4,
        "cadences_s": [1, 2, 4], "phase_fractions": list(PHASE_FRACTIONS),
        "profiles": ["short", "standard"],
        "planned_interventions": len(cov),
        "independent_runners": len(RUNNERS),
        "missing_independent_witnesses": int(cov.external_witness_count.eq(0).sum()),
        "duplicate_or_fragmented_witnesses": int(cov.external_witness_count.gt(1).sum()),
        "alignment_overrun_trials": int(cov.alignment_overrun.sum()),
        "raw_witness_events": len(out), "report_validation": provenance,
        "interpretation": [
            "Assignments randomized prospectively inside each independent Open5GS runner.",
            "Four independent environments per cadence; repeated interventions within a runner are correlated, and no p-value claim is made.",
            "Truthful/hidden and sampling offset are not detector inputs.",
            "Any missing/ambiguous witness remains visible instead of becoming a model negative.",
            "NRF CPU ticks are proxy, and local signed sidecar masking is NOT a real compromised NF.",
            "This laboratory protocol is not a production false-positive rate estimate.",
        ],
    }
    (output_dir / "trial_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(summarize(a.input_root, a.output_dir), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
