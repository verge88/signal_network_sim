"""Prospective Open5GS NRF KIE sampler-phase randomized interventions.

The orchestrator assigns NRF burst profile and sampler offset before capture.
Neither schedule nor mask/fault/phase labels enter frozen v1/v2 scoring.
Only signed-report sample times are read to schedule stimuli (not truth data).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random
import subprocess
import time

from run_open5gs_randomized_scenarios import atomic_control, endpoint, run_phase
from run_open5gs_stress import PROFILES

PHASE_FRACTIONS = (0.15, 0.50, 0.85)
SCENARIOS = ("truthful_nrf_burst", "hidden_nrf_burst")
FROZEN_DESIGN = "prospective-nrf-sampler-phase-v1"


def paired_factorial_plan(seed: int, cycles: int = 6) -> list[dict]:
    if cycles != 6:
        raise ValueError("six cycles required for balanced profile x offset grid")
    cells = [(p, phi) for p in ("short", "standard") for phi in PHASE_FRACTIONS]
    rng = random.Random(seed ^ 0x5A19)
    rng.shuffle(cells)
    plan = []
    for cycle, (profile, phi) in enumerate(cells, 1):
        phases = list(SCENARIOS)
        rng.shuffle(phases)
        for phase in phases:
            plan.append({
                "cycle": cycle, "profile": profile,
                "assigned_phase_fraction": phi, "phase": phase
            })
    return plan


def latest_nrf_sample(report_path: Path) -> float | None:
    """Read only signed report timestamp; fault marker/report values unused.

    Last few KB suffice for both NF reports at the current sequence; a
    partial first line from a tail read is ignored.
    """
    if not report_path.exists():
        return None
    with report_path.open("rb") as handle:
        size = handle.seek(0, 2)
        offset = max(0, size - 16384)
        handle.seek(offset)
        data = handle.read().splitlines()
    if offset:
        data = data[1:]
    for raw in reversed(data):
        try:
            row = json.loads(raw)
        except (ValueError, UnicodeDecodeError):
            continue
        if str(row.get("nf_id", "")).lower() == "nrf":
            return float(row["ts"])
    return None


def wait_for_new_nrf_sample(report_path: Path, after: float, period: float) -> float:
    deadline = time.monotonic() + max(15., 4. * period)
    while time.monotonic() < deadline:
        ts = latest_nrf_sample(report_path)
        if ts is not None and ts >= after:
            return ts
        time.sleep(0.05)
    raise TimeoutError("NRF signed sample did not arrive in preregistered interval")


def align_event_to_sample(
    reports: Path, period: float, phase_fraction: float,
    *, lead_in_s: float = 3.0,
) -> tuple[float, float, float]:
    """Return (signed sample time, planned event time, actual event time).

    Both truth and concealed phases use the same sampler alignment rule.
    Overrun is recorded rather than corrected post hoc.
    """
    if period not in (1., 2., 4.):
        raise ValueError("unexpected period")
    if phase_fraction not in PHASE_FRACTIONS:
        raise ValueError("unexpected randomized offset")
    time.sleep(lead_in_s)
    signed_ts = wait_for_new_nrf_sample(reports, time.time(), period)
    planned = signed_ts + period * phase_fraction
    time.sleep(max(0., planned - time.time()))
    actual = time.time()
    return signed_ts, planned, actual


def run(args: argparse.Namespace) -> None:
    if args.baseline_seconds < 120:
        raise ValueError("baseline must be at least 120 seconds")
    root = args.output_dir
    root.mkdir(parents=True, exist_ok=True)
    nrf_addr, nrf_port = endpoint(root / "open5gs_endpoints.json")
    reports = root / "kie_reports.jsonl"
    control = root / "kie_control.json"
    ordered = paired_factorial_plan(args.seed, args.cycles)
    manifest = {
        "design": FROZEN_DESIGN, "seed": args.seed,
        "cycles": args.cycles, "kie_period_s": args.kie_period_s,
        "background_rps": 0.0, "phase_fractions": list(PHASE_FRACTIONS),
        "baseline_seconds": args.baseline_seconds,
        "washout_seconds": args.washout_seconds,
        "stable_seconds": args.stable_seconds,
        "ordered_interventions": ordered,
        "profiles": PROFILES, "detector_labels_used": False,
        "sampling_alignment_source": "KIE report nf/ts only, not collector ground truth",
    }
    (root / "phase_trial_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8"
    )
    with (root / "phase_intervals.csv").open("w", newline="") as ff, \
         (root / "burst_profiles.csv").open("w", newline="") as pf, \
         (root / "randomized_alignment.csv").open("w", newline="") as af:
        phases = csv.writer(ff)
        phases.writerow(["start_ts", "end_ts", "phase", "cycle"])
        profiles = csv.writer(pf)
        profiles.writerow(["phase", "cycle", "profile", "requested", "successful",
                           "post_hold_s", "start_ts", "end_ts"])
        alignment = csv.writer(af)
        alignment.writerow(["phase", "cycle", "profile", "assigned_phase_fraction",
                            "reference_signed_nrf_ts", "planned_start_ts",
                            "actual_start_ts", "actual_offset_s",
                            "actual_phase_fraction", "phase_alignment_overrun",
                            "period_s"])
        run_phase(phases, control, "baseline", 0, args.baseline_seconds)
        ff.flush()
        url = f"http://{nrf_addr}:{nrf_port}/nnrf-nfm/v1/nf-instances"
        for item in ordered:
            cycle = item["cycle"]
            phase = item["phase"]
            profile = item["profile"]
            conf = PROFILES[profile]
            atomic_control(control, phase, "nrf" if phase.startswith("hidden_") else None)
            sample_ts, planned, started = align_event_to_sample(
                reports, args.kie_period_s, item["assigned_phase_fraction"]
            )
            attempted, successful = int(conf["requests"]), 0
            for _ in range(attempted):
                proc = subprocess.run(
                    ["curl", "--http2-prior-knowledge", "-sS", "--max-time", "4",
                     "-o", "/dev/null", url], check=False, timeout=6
                )
                successful += (proc.returncode == 0)
                time.sleep(0.03)
            time.sleep(float(conf["post_hold_s"]))
            ended = time.time()
            # Fail closed to incomplete stimuli rather than classify a failure
            # of curl as a missed detector.
            if successful != attempted:
                raise RuntimeError(f"NRF stimulus incomplete: {successful}/{attempted}")
            actual_fraction = (started - sample_ts) / args.kie_period_s
            alignment.writerow([
                phase, cycle, profile, item["assigned_phase_fraction"],
                f"{sample_ts:.6f}", f"{planned:.6f}", f"{started:.6f}",
                f"{started-sample_ts:.6f}", f"{actual_fraction:.6f}",
                int(actual_fraction >= 1.0 or abs(started-planned) > 0.25),
                args.kie_period_s,
            ])
            profiles.writerow([
                phase, cycle, profile, attempted, successful,
                conf["post_hold_s"], f"{started:.6f}", f"{ended:.6f}"
            ])
            phases.writerow([f"{started:.6f}", f"{ended:.6f}", phase, cycle])
            ff.flush()
            pf.flush()
            af.flush()
            print(f"cycle={cycle} {phase} {profile} offset={actual_fraction:.3f} "
                  f"delivered={successful}/{attempted}", flush=True)
            run_phase(phases, control, "washout", cycle, args.washout_seconds)
            run_phase(phases, control, "stable_recovery", cycle, args.stable_seconds)
            ff.flush()
        atomic_control(control, "complete", None)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--kie-period-s", type=float, choices=[1., 2., 4.], required=True)
    p.add_argument("--cycles", type=int, default=6)
    p.add_argument("--baseline-seconds", type=int, default=180)
    p.add_argument("--washout-seconds", type=int, default=12)
    p.add_argument("--stable-seconds", type=int, default=10)
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
