"""Paired v6 experiment: legacy median vs Byzantine-resistant 3-of-4 transport.

The semantic detector is frozen at the v5 candidate q3of5 with persistence off.
Both transport modes score exactly the same generated worlds. The runner adds
three stress classes beyond ordinary benign/attack traffic:

1) one physical transport origin INJECTs arbitrary-high telemetry;
2) one physical transport origin SUPPRESSes attack telemetry;
3) label-free legitimate semantic disturbances challenge precision.

Outputs are intentionally table-oriented for dissertation Chapter 4.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd

from sba_semantic_v6 import (
    AttackFamily,
    CompromiseMode,
    EvidenceOrigin,
    LegitimateDisturbance,
    SemanticQuorumDetector,
    SemanticSbaSimulator,
    Sophistication,
    TransportMode,
    empirical_auc,
)

LOG = logging.getLogger("sba-semantic-v6")
PHYSICAL_TRANSPORT_ORIGINS = (
    EvidenceOrigin.CONSUMER,
    EvidenceOrigin.SCP,
    EvidenceOrigin.PRODUCER,
    EvidenceOrigin.NWDAF,
)
MODES = (TransportMode.LEGACY_MEDIAN, TransportMode.BYZ_3OF4)


def setup_logging() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure:
            reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )


def progress(label: str, i: int, n: int, started: float) -> None:
    if n <= 0:
        return
    step = max(1, n // 5)
    if i not in (1, n) and i % step:
        return
    elapsed = max(time.perf_counter() - started, 1e-9)
    rate = i / elapsed
    eta = (n - i) / rate if rate > 0 else float("nan")
    LOG.info(
        "%-40s %4d/%-4d %5.1f%% %6.1f win/s ETA %6.1fs",
        label,
        i,
        n,
        100.0 * i / n,
        rate,
        eta,
    )


def wilson_interval(successes: int, n: int, z: float = 1.959963984540054):
    if n <= 0:
        return float("nan"), float("nan")
    p = successes / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n) / denom
    return max(0.0, center - half), min(1.0, center + half)


def attack_ids(window):
    return set(window.attack_event_ids)


def cell_metrics(windows, results) -> Dict[str, float]:
    n = len(windows)
    alerts = sum(int(r.alert) for r in results)
    semantic_alerts = sum(int(r.semantic_alert) for r in results)
    transport_alerts = sum(int(r.transport_alert) for r in results)
    hidden = 0
    detected = 0
    fired = 0
    false_ids = 0
    event_latencies: List[float] = []
    campaign_delays: List[float] = []

    for window, result in zip(windows, results):
        aids = attack_ids(window)
        fids = set(result.fired_event_ids)
        hidden += len(aids)
        detected += len(aids & fids)
        fired += len(fids)
        false_ids += len(fids - aids)
        by_id = {e.event_id: e for e in window.events}

        for eid in aids & fids:
            if eid in result.event_alert_times and eid in by_id:
                event_latencies.append(
                    max(0.0, result.event_alert_times[eid] - by_id[eid].timestamp)
                )
        if aids and result.alert and np.isfinite(result.first_alert_time):
            first_attack = min(by_id[eid].timestamp for eid in aids if eid in by_id)
            campaign_delays.append(max(0.0, result.first_alert_time - first_attack))

    lo, hi = wilson_interval(alerts, n)
    transport_lo, transport_hi = wilson_interval(transport_alerts, n)
    semantic_lo, semantic_hi = wilson_interval(semantic_alerts, n)
    return {
        "n_windows": n,
        "alerts": alerts,
        "window_rate": alerts / n if n else float("nan"),
        "window_rate_ci_lo": lo,
        "window_rate_ci_hi": hi,
        "semantic_rate": semantic_alerts / n if n else float("nan"),
        "semantic_rate_ci_lo": semantic_lo,
        "semantic_rate_ci_hi": semantic_hi,
        "transport_rate": transport_alerts / n if n else float("nan"),
        "transport_rate_ci_lo": transport_lo,
        "transport_rate_ci_hi": transport_hi,
        "event_recall": detected / hidden if hidden else float("nan"),
        "event_precision": detected / fired if fired else float("nan"),
        "false_attributed_events": false_ids,
        "event_latency_median_s": float(np.median(event_latencies)) if event_latencies else float("nan"),
        "event_latency_p95_s": float(np.percentile(event_latencies, 95)) if event_latencies else float("nan"),
        "campaign_ttd_median_s": float(np.median(campaign_delays)) if campaign_delays else float("nan"),
        "campaign_ttd_p95_s": float(np.percentile(campaign_delays, 95)) if campaign_delays else float("nan"),
        "unknown_fraction": float(np.nanmean([r.unknown_fraction for r in results])) if results else float("nan"),
    }


def generate_windows(sim, n: int, start_id: int, factory, label: str):
    windows = []
    started = time.perf_counter()
    for i in range(n):
        windows.append(factory(sim, start_id + i, i))
        progress(label, i + 1, n, started)
    return windows


def fit_detectors(calibration, target_fpr: float):
    detectors = {}
    for mode in MODES:
        LOG.info("Fitting detector transport=%s ...", mode.value)
        detectors[mode] = SemanticQuorumDetector(
            target_fpr,
            transport_mode=mode,
        ).fit(calibration)
        if mode == TransportMode.LEGACY_MEDIAN:
            LOG.info("  legacy thresholds=%s", detectors[mode].transport_thresholds)
        else:
            LOG.info("  robust thresholds=%s", detectors[mode].robust_thresholds)
    return detectors


def score_modes(windows, detectors):
    return {
        mode: [det.score(w) for w in windows]
        for mode, det in detectors.items()
    }


def evaluate_benign(seed: int, n: int, detectors):
    sim = SemanticSbaSimulator(seed=seed + 1000)
    contexts = list(sim.CONTROL_CONTEXTS)
    windows = generate_windows(
        sim,
        n,
        10_000,
        lambda s, wid, i: s.generate_window(wid, context=contexts[i % len(contexts)]),
        "benign paired world",
    )
    scored = score_modes(windows, detectors)
    rows = []
    rng = np.random.default_rng(seed + 515151)
    labels = rng.integers(0, 2, size=n)
    for mode in MODES:
        m = cell_metrics(windows, scored[mode])
        auc = empirical_auc(labels, [r.score for r in scored[mode]])
        rows.append({
            "mode": mode.value,
            "scenario": "benign",
            "null_auc": auc,
            **m,
        })
        LOG.info(
            "BENIGN %-13s FPR=%.4f [%.4f,%.4f] semantic=%.4f transport=%.4f nullAUC=%.4f",
            mode.value,
            m["window_rate"],
            m["window_rate_ci_lo"],
            m["window_rate_ci_hi"],
            m["semantic_rate"],
            m["transport_rate"],
            auc,
        )
    return rows


def evaluate_attacks(seed: int, n: int, hidden: int, detectors):
    rows = []
    families = [f for f in AttackFamily if f != AttackFamily.NONE]
    for j, family in enumerate(families):
        sim = SemanticSbaSimulator(seed=seed + 20_000 + 101 * j)
        windows = generate_windows(
            sim,
            n,
            20_000,
            lambda s, wid, _i, family=family: s.generate_window(
                wid,
                attack_family=family,
                hidden_calls=hidden,
                sophistication=Sophistication.ADAPTIVE,
            ),
            f"attack {family.value}",
        )
        scored = score_modes(windows, detectors)
        for mode in MODES:
            m = cell_metrics(windows, scored[mode])
            rows.append({
                "mode": mode.value,
                "family": family.value,
                "hidden_calls": hidden,
                **m,
            })
            LOG.info(
                "ATTACK %-13s %-20s winR=%.3f eventR=%.3f transport=%.3f eventLat=%.3fs",
                mode.value,
                family.value,
                m["window_rate"],
                m["event_recall"],
                m["transport_rate"],
                m["event_latency_median_s"],
            )
    return rows


def evaluate_transport_injection(seed: int, n: int, detectors):
    rows = []
    for j, origin in enumerate(PHYSICAL_TRANSPORT_ORIGINS):
        sim = SemanticSbaSimulator(seed=seed + 30_000 + j)
        windows = generate_windows(
            sim,
            n,
            30_000,
            lambda s, wid, _i, origin=origin: s.generate_window(
                wid,
                compromised_origins=[origin],
                compromise_mode=CompromiseMode.INJECT,
            ),
            f"INJECT {origin.value}",
        )
        scored = score_modes(windows, detectors)
        for mode in MODES:
            m = cell_metrics(windows, scored[mode])
            rows.append({
                "mode": mode.value,
                "origin": origin.value,
                "scenario": "one_origin_inject",
                "byzantine_fpr": m["window_rate"],
                "semantic_byzantine_fpr": m["semantic_rate"],
                "transport_byzantine_fpr": m["transport_rate"],
                "fpr_ci_lo": m["window_rate_ci_lo"],
                "fpr_ci_hi": m["window_rate_ci_hi"],
                "false_attributed_events": m["false_attributed_events"],
            })
            LOG.info(
                "INJECT %-13s %-8s total=%.4f semantic=%.4f transport=%.4f",
                mode.value,
                origin.value,
                m["window_rate"],
                m["semantic_rate"],
                m["transport_rate"],
            )
    return rows


def evaluate_transport_suppression(seed: int, n: int, hidden: int, detectors):
    rows = []
    families = [f for f in AttackFamily if f != AttackFamily.NONE]
    for origin_idx, origin in enumerate(PHYSICAL_TRANSPORT_ORIGINS):
        by_mode: Dict[TransportMode, List[Dict[str, float]]] = {m: [] for m in MODES}
        for j, family in enumerate(families):
            sim = SemanticSbaSimulator(
                seed=seed + 40_000 + 1000 * origin_idx + 101 * j
            )
            windows = generate_windows(
                sim,
                n,
                40_000,
                lambda s, wid, _i, origin=origin, family=family: s.generate_window(
                    wid,
                    attack_family=family,
                    hidden_calls=hidden,
                    sophistication=Sophistication.ADAPTIVE,
                    compromised_origins=[origin],
                    compromise_mode=CompromiseMode.SUPPRESS,
                ),
                f"SUPPRESS {origin.value} {family.value}",
            )
            scored = score_modes(windows, detectors)
            for mode in MODES:
                by_mode[mode].append(cell_metrics(windows, scored[mode]))

        for mode in MODES:
            metrics = by_mode[mode]
            row = {
                "mode": mode.value,
                "origin": origin.value,
                "scenario": "one_origin_suppress",
                "mean_window_recall": float(np.mean([m["window_rate"] for m in metrics])),
                "mean_event_recall": float(np.mean([m["event_recall"] for m in metrics])),
                "mean_transport_alert_rate": float(np.mean([m["transport_rate"] for m in metrics])),
                "mean_event_latency_s": float(np.nanmean([m["event_latency_median_s"] for m in metrics])),
            }
            rows.append(row)
            LOG.info(
                "SUPPRESS %-13s %-8s mean winR=%.3f eventR=%.3f transport=%.3f",
                mode.value,
                origin.value,
                row["mean_window_recall"],
                row["mean_event_recall"],
                row["mean_transport_alert_rate"],
            )
    return rows


def evaluate_disturbances(seed: int, n: int, detectors, disturbance_events: int):
    rows = []
    disturbances = [d for d in LegitimateDisturbance if d != LegitimateDisturbance.NONE]
    for j, disturbance in enumerate(disturbances):
        sim = SemanticSbaSimulator(seed=seed + 50_000 + 101 * j)
        windows = generate_windows(
            sim,
            n,
            50_000,
            lambda s, wid, _i, disturbance=disturbance: s.generate_window(
                wid,
                disturbance=disturbance,
                disturbance_events=disturbance_events,
            ),
            f"DISTURB {disturbance.value}",
        )
        # Scientific guard: these worlds must remain attack-label free.
        if any(w.label != 0 or w.attack_event_ids for w in windows):
            raise AssertionError("legitimate disturbance leaked into attack ground truth")

        scored = score_modes(windows, detectors)
        for mode in MODES:
            m = cell_metrics(windows, scored[mode])
            rows.append({
                "mode": mode.value,
                "disturbance": disturbance.value,
                "disturbance_events": disturbance_events,
                "operational_fpr": m["window_rate"],
                "semantic_operational_fpr": m["semantic_rate"],
                "transport_operational_fpr": m["transport_rate"],
                "fpr_ci_lo": m["window_rate_ci_lo"],
                "fpr_ci_hi": m["window_rate_ci_hi"],
                "false_attributed_events": m["false_attributed_events"],
            })
            LOG.info(
                "DISTURB %-13s %-26s FPR=%.4f semantic=%.4f transport=%.4f",
                mode.value,
                disturbance.value,
                m["window_rate"],
                m["semantic_rate"],
                m["transport_rate"],
            )
    return rows


def summarize(benign, attacks, injections, suppressions, disturbances):
    rows = []
    for mode in MODES:
        name = mode.value
        b = [r for r in benign if r["mode"] == name][0]
        a = [r for r in attacks if r["mode"] == name]
        inj = [r for r in injections if r["mode"] == name]
        sup = [r for r in suppressions if r["mode"] == name]
        dst = [r for r in disturbances if r["mode"] == name]
        rows.append({
            "mode": name,
            "benign_fpr": b["window_rate"],
            "benign_semantic_fpr": b["semantic_rate"],
            "benign_transport_fpr": b["transport_rate"],
            "null_auc": b["null_auc"],
            "mean_attack_window_recall": float(np.mean([r["window_rate"] for r in a])),
            "mean_attack_event_recall": float(np.mean([r["event_recall"] for r in a])),
            "mean_attack_transport_rate": float(np.mean([r["transport_rate"] for r in a])),
            "mean_suppress_window_recall": float(np.mean([r["mean_window_recall"] for r in sup])),
            "worst_suppress_window_recall": float(min(r["mean_window_recall"] for r in sup)),
            "mean_byzantine_fpr": float(np.mean([r["byzantine_fpr"] for r in inj])),
            "worst_byzantine_fpr": float(max(r["byzantine_fpr"] for r in inj)),
            "mean_transport_byzantine_fpr": float(np.mean([r["transport_byzantine_fpr"] for r in inj])),
            "worst_transport_byzantine_fpr": float(max(r["transport_byzantine_fpr"] for r in inj)),
            "mean_disturbance_fpr": float(np.mean([r["operational_fpr"] for r in dst])),
            "worst_disturbance_fpr": float(max(r["operational_fpr"] for r in dst)),
            "mean_semantic_disturbance_fpr": float(np.mean([r["semantic_operational_fpr"] for r in dst])),
            "worst_semantic_disturbance_fpr": float(max(r["semantic_operational_fpr"] for r in dst)),
        })
    return rows


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="SBA v6 Byzantine-resistant transport ablation")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--calib-windows", type=int, default=1200)
    p.add_argument("--eval-windows", type=int, default=200)
    p.add_argument("--trust-windows", type=int, default=200)
    p.add_argument("--disturbance-windows", type=int, default=200)
    p.add_argument("--disturbance-events", type=int, default=1)
    p.add_argument("--hidden", type=int, default=5)
    p.add_argument("--target-fpr", type=float, default=0.01)
    p.add_argument("--skip-suppress", action="store_true")
    p.add_argument("--out-dir", default="runs/sba_semantic_v6/transport_ablation")
    args = p.parse_args(argv)
    setup_logging()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    total_started = time.perf_counter()

    LOG.info("=" * 96)
    LOG.info(
        "V6 transport hardening | seed=%d | calib=%d eval=%d trust=%d disturbance=%d h=%d targetFPR=%.4f",
        args.seed,
        args.calib_windows,
        args.eval_windows,
        args.trust_windows,
        args.disturbance_windows,
        args.hidden,
        args.target_fpr,
    )
    LOG.info("Semantic candidate frozen: q3of5, persistence=off")

    calib_sim = SemanticSbaSimulator(seed=args.seed)
    contexts = list(calib_sim.CONTROL_CONTEXTS)
    calibration = generate_windows(
        calib_sim,
        args.calib_windows,
        0,
        lambda s, wid, i: s.generate_window(wid, context=contexts[i % len(contexts)]),
        "calibration paired world",
    )
    detectors = fit_detectors(calibration, args.target_fpr)

    benign = evaluate_benign(args.seed, args.eval_windows, detectors)
    attacks = evaluate_attacks(args.seed, args.eval_windows, args.hidden, detectors)
    injections = evaluate_transport_injection(args.seed, args.trust_windows, detectors)
    if args.skip_suppress:
        suppressions = [
            {
                "mode": mode.value,
                "origin": "skipped",
                "mean_window_recall": float("nan"),
                "mean_event_recall": float("nan"),
                "mean_transport_alert_rate": float("nan"),
                "mean_event_latency_s": float("nan"),
            }
            for mode in MODES
        ]
    else:
        suppressions = evaluate_transport_suppression(
            args.seed,
            args.trust_windows,
            args.hidden,
            detectors,
        )
    disturbances = evaluate_disturbances(
        args.seed,
        args.disturbance_windows,
        detectors,
        args.disturbance_events,
    )

    summary_rows = summarize(
        benign,
        attacks,
        injections,
        suppressions,
        disturbances,
    )

    pd.DataFrame(benign).to_csv(out / "benign_fpr.csv", index=False)
    pd.DataFrame(attacks).to_csv(out / "adaptive_attack_recall.csv", index=False)
    pd.DataFrame(injections).to_csv(out / "transport_byzantine_injection.csv", index=False)
    pd.DataFrame(suppressions).to_csv(out / "transport_suppression.csv", index=False)
    pd.DataFrame(disturbances).to_csv(out / "legitimate_disturbances.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(out / "transport_ablation_summary.csv", index=False)

    runtime = time.perf_counter() - total_started
    summary = {
        "seed": args.seed,
        "hidden": args.hidden,
        "target_fpr": args.target_fpr,
        "semantic_profile": "q3of5",
        "soft_persistence": False,
        "transport_modes": [m.value for m in MODES],
        "disturbance_events": args.disturbance_events,
        "runtime_seconds": runtime,
        "rows": summary_rows,
    }
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    LOG.info("=" * 96)
    LOG.info("FINAL TRANSPORT ABLATION")
    for row in summary_rows:
        LOG.info(
            "%-13s attackR=%.3f suppressWorst=%.3f ByzFPR mean/worst=%.3f/%.3f "
            "transportWorst=%.3f disturbance mean/worst=%.3f/%.3f semanticDistWorst=%.3f nullAUC=%.3f",
            row["mode"],
            row["mean_attack_window_recall"],
            row["worst_suppress_window_recall"],
            row["mean_byzantine_fpr"],
            row["worst_byzantine_fpr"],
            row["worst_transport_byzantine_fpr"],
            row["mean_disturbance_fpr"],
            row["worst_disturbance_fpr"],
            row["worst_semantic_disturbance_fpr"],
            row["null_auc"],
        )
    LOG.info("Runtime %.1fs | output=%s", runtime, out)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())