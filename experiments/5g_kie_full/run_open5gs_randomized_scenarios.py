"""Balanced, randomized Open5GS truthful/hidden experiment orchestration.

The orchestration manifest and phase labels are used for evaluation only.
No orchestration metadata enters event detection or signed-KIE consistency
decisions.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import random
import subprocess
import time


SCENARIOS = (
    "truthful_udm_restart",
    "hidden_udm_restart",
    "truthful_nrf_burst",
    "hidden_nrf_burst",
)


def randomized_plan(seed: int, cycles: int) -> list[dict[str, object]]:
    if cycles < 1:
        raise ValueError("cycles must be positive")
    rng = random.Random(seed)
    plan = []
    for cycle in range(1, cycles + 1):
        perm = list(SCENARIOS)
        rng.shuffle(perm)
        for position, phase in enumerate(perm):
            plan.append({"cycle": cycle, "phase": phase, "position": position})
    return plan


def atomic_control(path: Path, phase: str, masked_nf: str | None) -> None:
    content = {
        "phase": phase,
        "mask_nfs": [masked_nf] if masked_nf else [],
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(content) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def endpoint(path: Path) -> tuple[str, int]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    for row in doc["endpoints"]:
        if row["nf"].lower() == "nrf" and row["reachable"]:
            return str(row["address"]), int(row["port"])
    raise ValueError("no reachable NRF endpoint")


def run_phase(
    writer: csv.writer,
    control: Path,
    phase: str,
    cycle: int,
    duration_s: float,
) -> None:
    atomic_control(control, phase, None)
    start = time.time()
    time.sleep(duration_s)
    end = time.time()
    writer.writerow([f"{start:.6f}", f"{end:.6f}", phase, cycle])


def run_perturbation(
    writer: csv.writer,
    control: Path,
    phase: str,
    cycle: int,
    nrf_addr: str,
    nrf_port: int,
    burst_requests: int,
) -> None:
    is_hidden = phase.startswith("hidden_")
    target = "udm" if "udm_restart" in phase else "nrf"
    atomic_control(control, phase, target if is_hidden else None)
    # Symmetric lead-in (truthful and hidden); sidecar sampling aligns at 2s.
    time.sleep(3)
    start = time.time()

    if target == "udm":
        subprocess.run(
            ["sudo", "systemctl", "restart", "open5gs-udmd.service"],
            check=True,
            timeout=30,
        )
        time.sleep(10)
    else:
        url = f"http://{nrf_addr}:{nrf_port}/nnrf-nfm/v1/nf-instances"
        ok_count = 0
        for _ in range(burst_requests):
            p = subprocess.run(
                [
                    "curl", "--http2-prior-knowledge", "-sS",
                    "--max-time", "4", "-o", "/dev/null", url,
                ],
                check=False, timeout=6,
            )
            if p.returncode == 0:
                ok_count += 1
            time.sleep(0.03)
        print(
            f"NRF request control: {ok_count}/{burst_requests} HTTP2 GETs succeeded",
            flush=True,
        )
        time.sleep(3)

    end = time.time()
    writer.writerow([f"{start:.6f}", f"{end:.6f}", phase, cycle])


def run(args: argparse.Namespace) -> None:
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    control = out / "kie_control.json"
    host, port = endpoint(out / "open5gs_endpoints.json")
    plan = randomized_plan(args.seed, args.cycles)
    (out / "randomized_plan.json").write_text(
        json.dumps(
            {
                "seed": args.seed,
                "cycles": args.cycles,
                "ordered_phases": plan,
                "baseline_seconds": args.baseline_seconds,
                "washout_seconds": args.washout_seconds,
                "stable_seconds": args.stable_seconds,
                "burst_requests": args.burst_requests,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    with (out / "phase_intervals.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["start_ts", "end_ts", "phase", "cycle"])

        print(f"Baseline: {args.baseline_seconds} seconds", flush=True)
        run_phase(writer, control, "baseline", 0, args.baseline_seconds)
        f.flush()

        for item in plan:
            cycle = int(item["cycle"])
            phase = str(item["phase"])
            print(
                f"Randomized cycle={cycle} order={item['position']} phase={phase}",
                flush=True,
            )
            run_perturbation(
                writer, control, phase, cycle, host, port, args.burst_requests
            )
            f.flush()
            run_phase(writer, control, "washout", cycle, args.washout_seconds)
            f.flush()
            run_phase(writer, control, "stable_recovery", cycle, args.stable_seconds)
            f.flush()

        atomic_control(control, "complete", None)


def main() -> None:
    p = argparse.ArgumentParser(description="Randomized Open5GS KIE scenarios")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--cycles", type=int, default=6)
    p.add_argument("--baseline-seconds", type=int, default=1200)
    p.add_argument("--washout-seconds", type=int, default=25)
    p.add_argument("--stable-seconds", type=int, default=20)
    p.add_argument("--burst-requests", type=int, default=60)
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
