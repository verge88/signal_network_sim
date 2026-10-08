"""Prespecified, balanced Open5GS KIE cadence/burst stress protocol.

Stress variables are external stimulus only; the KIE v1/v2 detectors read no
profile names, labels, randomized plan, masked flags, or test ground truth.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random
import subprocess
import time

from run_open5gs_randomized_scenarios import (
    SCENARIOS, atomic_control, endpoint, run_phase, randomized_plan
)

PROFILES = {
    "short": {"requests": 30, "post_hold_s": 0.25},
    "standard": {"requests": 60, "post_hold_s": 3.0},
}


def profile_plan(seed: int, cycles: int) -> dict[int, str]:
    if cycles < 2 or cycles % 2:
        raise ValueError("balanced stress profiles require a positive even cycle count")
    sequence = ["short"] * (cycles // 2) + ["standard"] * (cycles // 2)
    random.Random(seed ^ 0xCADA).shuffle(sequence)
    return dict(zip(range(1, cycles + 1), sequence))


def stress_intervention(
    writer: csv.writer,
    profile_writer: csv.writer,
    control: Path,
    phase: str,
    cycle: int,
    addr: str,
    port: int,
    profile: str,
) -> None:
    if phase not in SCENARIOS:
        raise ValueError(f"unsupported intervention {phase!r}")
    params = PROFILES[profile]
    hidden = phase.startswith("hidden_")
    nf = "udm" if phase.endswith("udm_restart") else "nrf"
    atomic_control(control, phase, nf if hidden else None)
    # Apply identical masking lead-in to each profile: 3 seconds.
    time.sleep(3)
    started = time.time()
    successes = 0
    attempted = 0
    if nf == "udm":
        subprocess.run(
            ["sudo", "systemctl", "restart", "open5gs-udmd.service"],
            check=True, timeout=30
        )
        time.sleep(10)
    else:
        url = f"http://{addr}:{port}/nnrf-nfm/v1/nf-instances"
        for _ in range(int(params["requests"])):
            attempted += 1
            result = subprocess.run(
                ["curl", "--http2-prior-knowledge", "-sS",
                 "--max-time", "4", "-o", "/dev/null", url],
                check=False, timeout=6
            )
            successes += int(result.returncode == 0)
            time.sleep(0.03)
        time.sleep(float(params["post_hold_s"]))
    ended = time.time()
    if nf == "nrf" and successes != attempted:
        raise RuntimeError(
            f"NRF workload incomplete: {successes}/{attempted} requests succeeded"
        )
    writer.writerow([f"{started:.6f}", f"{ended:.6f}", phase, cycle])
    profile_writer.writerow([
        phase,cycle,profile,attempted,successes,params["post_hold_s"],
        f"{started:.6f}",f"{ended:.6f}"
    ])
    print(
        f"stress cycle={cycle} {phase} profile={profile} "
        f"requests={successes}/{attempted} duration={ended-started:.2f}s",
        flush=True
    )


def run(args: argparse.Namespace) -> None:
    root: Path = args.output_dir
    root.mkdir(parents=True, exist_ok=True)
    control = root / "kie_control.json"
    addr, port = endpoint(root / "open5gs_endpoints.json")
    profiles = profile_plan(args.seed, args.cycles)
    plan = randomized_plan(args.seed, args.cycles)
    metadata = {
        "seed": args.seed,
        "cycles": args.cycles,
        "kie_period_s": args.kie_period_s,
        "background_rps": args.background_rps,
        "baseline_seconds": args.baseline_seconds,
        "washout_seconds": args.washout_seconds,
        "stable_seconds": args.stable_seconds,
        "post_intervention_controls": {
            "short": PROFILES["short"],
            "standard": PROFILES["standard"]
        },
        "profile_per_cycle": profiles,
        "ordered_interventions":plan,
        "ground_truth_only":True,
    }
    (root / "stress_manifest.json").write_text(
        json.dumps(metadata,indent=2,ensure_ascii=False)+"\n",encoding="utf-8"
    )

    with (root/"phase_intervals.csv").open("w",newline="",encoding="utf-8") as phases_file, \
         (root/"burst_profiles.csv").open("w",newline="",encoding="utf-8") as prof_file:
        writer=csv.writer(phases_file)
        writer.writerow(["start_ts","end_ts","phase","cycle"])
        prof=csv.writer(prof_file)
        prof.writerow(["phase","cycle","profile","requested","successful",
                       "post_hold_s","start_ts","end_ts"])
        run_phase(writer,control,"baseline",0,args.baseline_seconds)
        phases_file.flush()
        for item in plan:
            cycle=int(item["cycle"]); phase=str(item["phase"])
            stress_intervention(
                writer,prof,control,phase,cycle,addr,port,profiles[cycle]
            )
            phases_file.flush();prof_file.flush()
            run_phase(writer,control,"washout",cycle,args.washout_seconds)
            run_phase(writer,control,"stable_recovery",cycle,args.stable_seconds)
            phases_file.flush()
        atomic_control(control,"complete",None)


def main()->None:
    p=argparse.ArgumentParser(description="Balanced Open5GS KIE cadence and burst profile experiment")
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--seed",type=int,required=True)
    p.add_argument("--kie-period-s",type=float,choices=[1.0,4.0],required=True)
    p.add_argument("--background-rps",type=float,choices=[0.0,2.0],required=True)
    p.add_argument("--cycles",type=int,default=6)
    p.add_argument("--baseline-seconds",type=int,default=1200)
    p.add_argument("--washout-seconds",type=int,default=25)
    p.add_argument("--stable-seconds",type=int,default=20)
    args=p.parse_args()
    run(args)


if __name__=="__main__":
    main()
