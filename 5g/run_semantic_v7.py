"""Paired multiseed experiment for SBA semantic consistency v7.

Compares on exactly the same generated worlds:

* v6 baseline: q3of5, persistence OFF, byz_3of4 transport;
* v7 candidate: the same frozen quorum/transport plus correlation/fact-scoped
  operational grace and state convergence.

The experiment measures ordinary benign FPR, adaptive attack Recall, legitimate
operational-disturbance FPR, and a grace-abuse stress case where an attack occurs
inside a legitimate-looking operational interval but does not recover.

Across seeds, paired deltas are summarized with bootstrap confidence intervals,
an exact/Monte-Carlo paired sign-flip test, and Holm correction.
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from sba_semantic_v6 import (
    SemanticQuorumDetector as V6SemanticQuorumDetector,
)
from sba_semantic_v7 import (
    ATTACK_OPERATIONAL_STATE,
    AttackFamily,
    LegitimateDisturbance,
    SemanticQuorumDetector,
    SemanticSbaSimulator,
    Sophistication,
    TransportMode,
    empirical_auc,
)

LOG = logging.getLogger("sba-semantic-v7")
ARM_V6 = "v6_static_q3of5"
ARM_V7 = "v7_operational_grace"
ARMS = (ARM_V6, ARM_V7)


def setup_logging(quiet: bool = False) -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure:
            reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(
        level=logging.WARNING if quiet else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )


def progress(label: str, i: int, n: int, started: float, log_every: int = 0) -> None:
    if n <= 0:
        return
    step = log_every if log_every > 0 else max(1, n // 5)
    if i not in (1, n) and i % step:
        return
    elapsed = max(time.perf_counter() - started, 1e-9)
    rate = i / elapsed
    eta = (n - i) / rate if rate > 0 else float("nan")
    LOG.info(
        "%-42s %4d/%-4d %5.1f%% %6.1f win/s ETA %6.1fs",
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


def cell_metrics(windows, results) -> Dict[str, float]:
    n = len(windows)
    alerts = int(sum(r.alert for r in results))
    semantic_alerts = int(sum(r.semantic_alert for r in results))
    transport_alerts = int(sum(r.transport_alert for r in results))
    hidden = 0
    detected = 0
    fired = 0
    false_ids = 0
    event_latencies: List[float] = []
    campaign_delays: List[float] = []
    deferred = 0
    resolved = 0
    expired = 0

    for window, result in zip(windows, results):
        aids = set(window.attack_event_ids)
        fids = set(result.fired_event_ids)
        hidden += len(aids)
        detected += len(aids & fids)
        fired += len(fids)
        false_ids += len(fids - aids)
        deferred += int(getattr(result, "grace_deferred_keys", 0))
        resolved += int(getattr(result, "grace_resolved_keys", 0))
        expired += int(getattr(result, "grace_expired_alerts", 0))

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
    return {
        "n_windows": n,
        "alerts": alerts,
        "window_rate": alerts / n if n else float("nan"),
        "window_rate_ci_lo": lo,
        "window_rate_ci_hi": hi,
        "semantic_rate": semantic_alerts / n if n else float("nan"),
        "transport_rate": transport_alerts / n if n else float("nan"),
        "event_recall": detected / hidden if hidden else float("nan"),
        "event_precision": detected / fired if fired else float("nan"),
        "false_attributed_events": false_ids,
        "event_latency_median_s": float(np.median(event_latencies)) if event_latencies else float("nan"),
        "event_latency_p95_s": float(np.percentile(event_latencies, 95)) if event_latencies else float("nan"),
        "campaign_ttd_median_s": float(np.median(campaign_delays)) if campaign_delays else float("nan"),
        "campaign_ttd_p95_s": float(np.percentile(campaign_delays, 95)) if campaign_delays else float("nan"),
        "grace_deferred_keys": deferred,
        "grace_resolved_keys": resolved,
        "grace_expired_alerts": expired,
    }


def generate_windows(sim, n: int, start_id: int, factory, label: str, log_every: int):
    out = []
    started = time.perf_counter()
    for i in range(n):
        out.append(factory(sim, start_id + i, i))
        progress(label, i + 1, n, started, log_every)
    return out


def fit_pair(calibration, target_fpr: float):
    baseline = V6SemanticQuorumDetector(
        target_fpr,
        transport_mode=TransportMode.BYZ_3OF4,
    ).fit(calibration)
    candidate = SemanticQuorumDetector(
        target_fpr,
        enable_operational_grace=True,
    ).fit(calibration)
    return {ARM_V6: baseline, ARM_V7: candidate}


def score_arms(windows, detectors):
    return {
        arm: [det.score(w) for w in windows]
        for arm, det in detectors.items()
    }


def evaluate_benign(seed: int, n: int, detectors, log_every: int):
    sim = SemanticSbaSimulator(seed=seed + 10_000)
    contexts = list(sim.CONTROL_CONTEXTS)
    windows = generate_windows(
        sim,
        n,
        10_000,
        lambda s, wid, i: s.generate_window(wid, context=contexts[i % len(contexts)]),
        f"seed {seed} benign",
        log_every,
    )
    scored = score_arms(windows, detectors)
    rng = np.random.default_rng(seed + 515_151)
    labels = rng.integers(0, 2, size=n)
    rows = []
    for arm in ARMS:
        m = cell_metrics(windows, scored[arm])
        auc = empirical_auc(labels, [r.score for r in scored[arm]])
        rows.append({
            "seed": seed,
            "arm": arm,
            "scenario": "benign",
            "null_auc": auc,
            **m,
        })
        LOG.info(
            "  BENIGN %-20s FPR=%.4f semantic=%.4f transport=%.4f nullAUC=%.4f",
            arm,
            m["window_rate"],
            m["semantic_rate"],
            m["transport_rate"],
            auc,
        )
    return rows


def evaluate_attacks(seed: int, n: int, hidden: int, detectors, log_every: int):
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
            f"seed {seed} attack {family.value}",
            log_every,
        )
        scored = score_arms(windows, detectors)
        for arm in ARMS:
            m = cell_metrics(windows, scored[arm])
            rows.append({
                "seed": seed,
                "arm": arm,
                "family": family.value,
                "hidden_calls": hidden,
                **m,
            })
            LOG.info(
                "  ATTACK %-20s %-20s winR=%.3f eventR=%.3f eventLat=%.3fs",
                arm,
                family.value,
                m["window_rate"],
                m["event_recall"],
                m["event_latency_median_s"],
            )
    return rows


def evaluate_disturbances(seed: int, n: int, detectors, disturbance_events: int, log_every: int):
    rows = []
    disturbances = [d for d in LegitimateDisturbance if d != LegitimateDisturbance.NONE]
    for j, disturbance in enumerate(disturbances):
        sim = SemanticSbaSimulator(seed=seed + 30_000 + 101 * j)
        windows = generate_windows(
            sim,
            n,
            30_000,
            lambda s, wid, _i, disturbance=disturbance: s.generate_window(
                wid,
                disturbance=disturbance,
                disturbance_events=disturbance_events,
            ),
            f"seed {seed} disturb {disturbance.value}",
            log_every,
        )
        if any(w.label or w.attack_event_ids for w in windows):
            raise AssertionError("operational disturbance leaked into attack ground truth")
        scored = score_arms(windows, detectors)
        for arm in ARMS:
            m = cell_metrics(windows, scored[arm])
            rows.append({
                "seed": seed,
                "arm": arm,
                "disturbance": disturbance.value,
                "disturbance_events": disturbance_events,
                "operational_fpr": m["window_rate"],
                "semantic_operational_fpr": m["semantic_rate"],
                "transport_operational_fpr": m["transport_rate"],
                "false_attributed_events": m["false_attributed_events"],
                "grace_deferred_keys": m["grace_deferred_keys"],
                "grace_resolved_keys": m["grace_resolved_keys"],
                "grace_expired_alerts": m["grace_expired_alerts"],
            })
            LOG.info(
                "  DISTURB %-20s %-26s FPR=%.3f semantic=%.3f deferred/resolved/expired=%d/%d/%d",
                arm,
                disturbance.value,
                m["window_rate"],
                m["semantic_rate"],
                m["grace_deferred_keys"],
                m["grace_resolved_keys"],
                m["grace_expired_alerts"],
            )
    return rows


def evaluate_grace_abuse(seed: int, n: int, detectors, log_every: int):
    rows = []
    families = list(ATTACK_OPERATIONAL_STATE)
    for j, family in enumerate(families):
        sim = SemanticSbaSimulator(seed=seed + 40_000 + 101 * j)
        windows = generate_windows(
            sim,
            n,
            40_000,
            lambda s, wid, _i, family=family: s.cover_attack_with_operational_marker(
                s.generate_window(
                    wid,
                    attack_family=family,
                    hidden_calls=1,
                    sophistication=Sophistication.ADAPTIVE,
                )
            ),
            f"seed {seed} grace-abuse {family.value}",
            log_every,
        )
        scored = score_arms(windows, detectors)
        for arm in ARMS:
            m = cell_metrics(windows, scored[arm])
            rows.append({
                "seed": seed,
                "arm": arm,
                "family": family.value,
                "scenario": "attack_during_operational_grace",
                **m,
            })
            LOG.info(
                "  GRACE-ABUSE %-16s %-20s winR=%.3f eventR=%.3f eventLat=%.3fs expired=%d",
                arm,
                family.value,
                m["window_rate"],
                m["event_recall"],
                m["event_latency_median_s"],
                m["grace_expired_alerts"],
            )
    return rows


def seed_summary(seed, benign, attacks, disturbances, grace_abuse):
    rows = []
    for arm in ARMS:
        b = next(r for r in benign if r["arm"] == arm)
        a = [r for r in attacks if r["arm"] == arm]
        d = [r for r in disturbances if r["arm"] == arm]
        g = [r for r in grace_abuse if r["arm"] == arm]
        rows.append({
            "seed": seed,
            "arm": arm,
            "benign_fpr": b["window_rate"],
            "null_auc": b["null_auc"],
            "mean_attack_window_recall": float(np.mean([r["window_rate"] for r in a])),
            "mean_attack_event_recall": float(np.mean([r["event_recall"] for r in a])),
            "mean_disturbance_fpr": float(np.mean([r["operational_fpr"] for r in d])),
            "worst_disturbance_fpr": float(max(r["operational_fpr"] for r in d)),
            "mean_semantic_disturbance_fpr": float(np.mean([r["semantic_operational_fpr"] for r in d])),
            "worst_semantic_disturbance_fpr": float(max(r["semantic_operational_fpr"] for r in d)),
            "grace_abuse_window_recall": float(np.mean([r["window_rate"] for r in g])),
            "grace_abuse_event_recall": float(np.mean([r["event_recall"] for r in g])),
            "grace_abuse_event_latency_s": float(np.nanmean([r["event_latency_median_s"] for r in g])),
            "grace_resolved_keys": int(sum(r["grace_resolved_keys"] for r in d)),
            "grace_expired_alerts": int(sum(r["grace_expired_alerts"] for r in d)),
        })
    return rows


def run_seed(
    *,
    seed: int,
    calib_windows: int,
    eval_windows: int,
    disturbance_windows: int,
    grace_abuse_windows: int,
    hidden: int,
    target_fpr: float,
    disturbance_events: int,
    out_dir: Path,
    log_every: int,
):
    started = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    LOG.info("=" * 96)
    LOG.info(
        "SEED %d | calib=%d eval=%d disturbance=%d grace-abuse=%d h=%d targetFPR=%.4f",
        seed,
        calib_windows,
        eval_windows,
        disturbance_windows,
        grace_abuse_windows,
        hidden,
        target_fpr,
    )

    sim = SemanticSbaSimulator(seed=seed)
    contexts = list(sim.CONTROL_CONTEXTS)
    calibration = generate_windows(
        sim,
        calib_windows,
        0,
        lambda s, wid, i: s.generate_window(wid, context=contexts[i % len(contexts)]),
        f"seed {seed} calibration",
        log_every,
    )
    detectors = fit_pair(calibration, target_fpr)

    benign = evaluate_benign(seed, eval_windows, detectors, log_every)
    attacks = evaluate_attacks(seed, eval_windows, hidden, detectors, log_every)
    disturbances = evaluate_disturbances(
        seed,
        disturbance_windows,
        detectors,
        disturbance_events,
        log_every,
    )
    grace_abuse = evaluate_grace_abuse(
        seed,
        grace_abuse_windows,
        detectors,
        log_every,
    )
    summary = seed_summary(seed, benign, attacks, disturbances, grace_abuse)

    pd.DataFrame(benign).to_csv(out_dir / "benign.csv", index=False)
    pd.DataFrame(attacks).to_csv(out_dir / "attacks.csv", index=False)
    pd.DataFrame(disturbances).to_csv(out_dir / "disturbances.csv", index=False)
    pd.DataFrame(grace_abuse).to_csv(out_dir / "grace_abuse.csv", index=False)
    pd.DataFrame(summary).to_csv(out_dir / "seed_summary.csv", index=False)

    runtime = time.perf_counter() - started
    for row in summary:
        LOG.info(
            "SEED %d %-20s attackR=%.3f distFPR mean/worst=%.3f/%.3f graceAbuseR=%.3f latency=%.3fs",
            seed,
            row["arm"],
            row["mean_attack_window_recall"],
            row["mean_disturbance_fpr"],
            row["worst_disturbance_fpr"],
            row["grace_abuse_window_recall"],
            row["grace_abuse_event_latency_s"],
        )
    LOG.info("SEED %d DONE %.1fs", seed, runtime)
    return {
        "benign": pd.DataFrame(benign),
        "attacks": pd.DataFrame(attacks),
        "disturbances": pd.DataFrame(disturbances),
        "grace_abuse": pd.DataFrame(grace_abuse),
        "summary": pd.DataFrame(summary),
        "runtime_seconds": runtime,
    }


def bootstrap_delta_ci(deltas: Sequence[float], seed: int, n_boot: int = 5000):
    x = np.asarray(deltas, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan"), float("nan")
    if len(x) == 1:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(int(n_boot), len(x)))
    means = x[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_signflip_p(deltas: Sequence[float], seed: int, n_perm: int = 10000) -> float:
    x = np.asarray(deltas, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return float("nan")
    if np.allclose(x, 0.0):
        return 1.0
    observed = abs(float(x.mean()))
    n = len(x)
    if n <= 16:
        total = 0
        extreme = 0
        for signs in itertools.product((-1.0, 1.0), repeat=n):
            total += 1
            value = abs(float(np.mean(x * np.asarray(signs))))
            extreme += int(value >= observed - 1e-15)
        return extreme / total

    rng = np.random.default_rng(seed)
    signs = rng.choice((-1.0, 1.0), size=(int(n_perm), n))
    null = np.abs((signs * x).mean(axis=1))
    return float((1 + np.sum(null >= observed)) / (len(null) + 1))


def holm_adjust(pvalues: Sequence[float]) -> List[float]:
    p = np.asarray(pvalues, dtype=float)
    out = np.full(len(p), np.nan)
    finite = np.flatnonzero(np.isfinite(p))
    if not len(finite):
        return out.tolist()
    order = finite[np.argsort(p[finite])]
    m = len(order)
    running = 0.0
    for rank, idx in enumerate(order):
        adjusted = (m - rank) * p[idx]
        running = max(running, adjusted)
        out[idx] = min(1.0, running)
    return out.tolist()


def _paired_row(
    df: pd.DataFrame,
    *,
    metric: str,
    scope: str,
    scenario: str,
    seed_offset: int,
    n_boot: int,
    n_perm: int,
):
    pivot = df.pivot_table(index="seed", columns="arm", values=metric, aggfunc="mean")
    pivot = pivot.dropna(subset=list(ARMS))
    if pivot.empty:
        return None
    baseline = pivot[ARM_V6].to_numpy(dtype=float)
    candidate = pivot[ARM_V7].to_numpy(dtype=float)
    deltas = candidate - baseline
    ci_lo, ci_hi = bootstrap_delta_ci(deltas, seed_offset, n_boot)
    p = paired_signflip_p(deltas, seed_offset + 1, n_perm)
    return {
        "scope": scope,
        "scenario": scenario,
        "metric": metric,
        "n_seeds": len(pivot),
        "v6_mean": float(np.mean(baseline)),
        "v7_mean": float(np.mean(candidate)),
        "delta_v7_minus_v6": float(np.mean(deltas)),
        "delta_ci_lo": ci_lo,
        "delta_ci_hi": ci_hi,
        "paired_signflip_p": p,
    }


def paired_statistics(
    summaries: pd.DataFrame,
    attacks: pd.DataFrame,
    disturbances: pd.DataFrame,
    grace_abuse: pd.DataFrame,
    *,
    seed: int,
    n_boot: int,
    n_perm: int,
) -> pd.DataFrame:
    rows = []
    overall_metrics = (
        "benign_fpr",
        "mean_attack_window_recall",
        "mean_attack_event_recall",
        "mean_disturbance_fpr",
        "worst_disturbance_fpr",
        "mean_semantic_disturbance_fpr",
        "grace_abuse_window_recall",
        "grace_abuse_event_recall",
        "grace_abuse_event_latency_s",
    )
    for i, metric in enumerate(overall_metrics):
        row = _paired_row(
            summaries,
            metric=metric,
            scope="overall",
            scenario="ALL",
            seed_offset=seed + 1000 * (i + 1),
            n_boot=n_boot,
            n_perm=n_perm,
        )
        if row:
            rows.append(row)

    for j, family in enumerate(sorted(attacks["family"].unique())):
        part = attacks[attacks["family"] == family]
        for k, metric in enumerate(("window_rate", "event_recall")):
            row = _paired_row(
                part,
                metric=metric,
                scope="attack",
                scenario=family,
                seed_offset=seed + 100_000 + 1000 * j + k,
                n_boot=n_boot,
                n_perm=n_perm,
            )
            if row:
                rows.append(row)

    for j, disturbance in enumerate(sorted(disturbances["disturbance"].unique())):
        part = disturbances[disturbances["disturbance"] == disturbance]
        for k, metric in enumerate(("operational_fpr", "semantic_operational_fpr")):
            row = _paired_row(
                part,
                metric=metric,
                scope="disturbance",
                scenario=disturbance,
                seed_offset=seed + 200_000 + 1000 * j + k,
                n_boot=n_boot,
                n_perm=n_perm,
            )
            if row:
                rows.append(row)

    for j, family in enumerate(sorted(grace_abuse["family"].unique())):
        part = grace_abuse[grace_abuse["family"] == family]
        for k, metric in enumerate(("window_rate", "event_recall", "event_latency_median_s")):
            row = _paired_row(
                part,
                metric=metric,
                scope="grace_abuse",
                scenario=family,
                seed_offset=seed + 300_000 + 1000 * j + k,
                n_boot=n_boot,
                n_perm=n_perm,
            )
            if row:
                rows.append(row)

    out = pd.DataFrame(rows)
    if not out.empty:
        out["holm_p"] = holm_adjust(out["paired_signflip_p"].to_numpy())
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="SBA v7 operational-grace paired multiseed experiment")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    p.add_argument("--calib-windows", type=int, default=1200)
    p.add_argument("--eval-windows", type=int, default=200)
    p.add_argument("--disturbance-windows", type=int, default=200)
    p.add_argument("--grace-abuse-windows", type=int, default=100)
    p.add_argument("--disturbance-events", type=int, default=1)
    p.add_argument("--hidden", type=int, default=5)
    p.add_argument("--target-fpr", type=float, default=0.01)
    p.add_argument("--bootstrap", type=int, default=5000)
    p.add_argument("--permutations", type=int, default=10000)
    p.add_argument("--log-every", type=int, default=0)
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--out-dir", default="runs/sba_semantic_v7/operational_grace")
    args = p.parse_args(argv)
    setup_logging(args.quiet)

    seeds = [int(x) for x in (args.seeds or [args.seed])]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    total_started = time.perf_counter()
    LOG.info(
        "V7 operational grace | seeds=%s | calib=%d eval=%d disturbance=%d grace-abuse=%d h=%d targetFPR=%.4f",
        seeds,
        args.calib_windows,
        args.eval_windows,
        args.disturbance_windows,
        args.grace_abuse_windows,
        args.hidden,
        args.target_fpr,
    )
    LOG.info("Frozen candidate: semantic=q3of5 persistence=off transport=byz_3of4")

    results = []
    for idx, seed in enumerate(seeds, 1):
        LOG.info("#" * 96)
        LOG.info("MULTISEED %d/%d -> seed=%d", idx, len(seeds), seed)
        seed_dir = out / f"seed_{seed}" if len(seeds) > 1 else out
        results.append(
            run_seed(
                seed=seed,
                calib_windows=args.calib_windows,
                eval_windows=args.eval_windows,
                disturbance_windows=args.disturbance_windows,
                grace_abuse_windows=args.grace_abuse_windows,
                hidden=args.hidden,
                target_fpr=args.target_fpr,
                disturbance_events=args.disturbance_events,
                out_dir=seed_dir,
                log_every=args.log_every,
            )
        )

    benign = pd.concat([r["benign"] for r in results], ignore_index=True)
    attacks = pd.concat([r["attacks"] for r in results], ignore_index=True)
    disturbances = pd.concat([r["disturbances"] for r in results], ignore_index=True)
    grace_abuse = pd.concat([r["grace_abuse"] for r in results], ignore_index=True)
    summaries = pd.concat([r["summary"] for r in results], ignore_index=True)

    benign.to_csv(out / "all_seed_benign.csv", index=False)
    attacks.to_csv(out / "all_seed_attacks.csv", index=False)
    disturbances.to_csv(out / "all_seed_disturbances.csv", index=False)
    grace_abuse.to_csv(out / "all_seed_grace_abuse.csv", index=False)
    summaries.to_csv(out / "all_seed_summary.csv", index=False)

    stats = paired_statistics(
        summaries,
        attacks,
        disturbances,
        grace_abuse,
        seed=min(seeds),
        n_boot=args.bootstrap,
        n_perm=args.permutations,
    )
    stats.to_csv(out / "paired_statistics.csv", index=False)

    runtime = time.perf_counter() - total_started
    primary = stats[
        (stats["scope"] == "overall")
        & stats["metric"].isin(
            [
                "mean_attack_window_recall",
                "mean_disturbance_fpr",
                "mean_semantic_disturbance_fpr",
                "grace_abuse_window_recall",
            ]
        )
    ] if not stats.empty else pd.DataFrame()

    LOG.info("=" * 96)
    LOG.info("FINAL PAIRED MULTISEED SUMMARY")
    if len(seeds) < 3:
        LOG.warning("Only %d seed(s): paired CI/p-values are diagnostic, not final evidence.", len(seeds))
    for _, row in primary.iterrows():
        LOG.info(
            "%-32s v6=%.4f v7=%.4f delta=%+.4f CI95=[%s,%s] p=%s Holm=%s",
            row["metric"],
            row["v6_mean"],
            row["v7_mean"],
            row["delta_v7_minus_v6"],
            f"{row['delta_ci_lo']:.4f}" if np.isfinite(row["delta_ci_lo"]) else "nan",
            f"{row['delta_ci_hi']:.4f}" if np.isfinite(row["delta_ci_hi"]) else "nan",
            f"{row['paired_signflip_p']:.4g}" if np.isfinite(row["paired_signflip_p"]) else "nan",
            f"{row['holm_p']:.4g}" if np.isfinite(row["holm_p"]) else "nan",
        )

    summary = {
        "seeds": seeds,
        "n_seeds": len(seeds),
        "target_fpr": args.target_fpr,
        "semantic_profile": "q3of5",
        "soft_persistence": False,
        "transport_mode": "byz_3of4",
        "operational_grace": True,
        "runtime_seconds": runtime,
        "primary_paired_statistics": primary.to_dict(orient="records") if not primary.empty else [],
    }
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    LOG.info("Runtime %.1fs | output=%s", runtime, out)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
