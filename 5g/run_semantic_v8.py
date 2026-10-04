"""Paired multiseed experiment for SBA semantic consistency v8.

V8 evaluates the same frozen q3of5 + byz_3of4 detector with and without noisy
operational grace on exactly the same generated worlds. Unlike v7, the marker
and recovery channels are imperfect and are generated independently from the
latent operational state.

Primary questions:
* does noisy operational context still reduce transient semantic FPR?
* does ordinary adaptive Recall remain stable in the presence of false markers?
* when an adaptive attack occurs during a genuine operational state, is Recall
  preserved and what latency cost is introduced?
* how do marker miss, wrong-correlation, late-marker and missing/late-recovery
  rates explain residual errors?
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from run_semantic_v7 import bootstrap_delta_ci, holm_adjust, paired_signflip_p
from sba_semantic_v6 import SemanticQuorumDetector as V6SemanticQuorumDetector
from sba_semantic_v8 import (
    ATTACK_STATE,
    AttackFamily,
    OperationalContextConfig,
    OperationalState,
    STATE_SPECS,
    SemanticFact,
    SemanticQuorumDetector,
    SemanticSbaSimulator,
    Sophistication,
    TransportMode,
    empirical_auc,
)

LOG = logging.getLogger("sba-semantic-v8")
ARM_STATIC = "static_q3of5"
ARM_NOISY = "noisy_operational_grace"
ARMS = (ARM_STATIC, ARM_NOISY)


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
        "%-44s %4d/%-4d %5.1f%% %6.1f win/s ETA %6.1fs",
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
    semantic = int(sum(r.semantic_alert for r in results))
    transport = int(sum(r.transport_alert for r in results))
    hidden = detected = fired = false_ids = 0
    latencies: List[float] = []
    campaign: List[float] = []
    deferred = resolved = expired = markers_seen = 0

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
        markers_seen += int(getattr(result, "markers_seen", 0))
        by_id = {e.event_id: e for e in window.events}
        for eid in aids & fids:
            if eid in result.event_alert_times and eid in by_id:
                latencies.append(max(0.0, result.event_alert_times[eid] - by_id[eid].timestamp))
        if aids and result.alert and np.isfinite(result.first_alert_time):
            t0 = min(by_id[eid].timestamp for eid in aids if eid in by_id)
            campaign.append(max(0.0, result.first_alert_time - t0))

    lo, hi = wilson_interval(alerts, n)
    return {
        "n_windows": n,
        "alerts": alerts,
        "window_rate": alerts / n if n else float("nan"),
        "window_rate_ci_lo": lo,
        "window_rate_ci_hi": hi,
        "semantic_rate": semantic / n if n else float("nan"),
        "transport_rate": transport / n if n else float("nan"),
        "event_recall": detected / hidden if hidden else float("nan"),
        "event_precision": detected / fired if fired else float("nan"),
        "false_attributed_events": false_ids,
        "event_latency_median_s": float(np.median(latencies)) if latencies else float("nan"),
        "event_latency_p95_s": float(np.percentile(latencies, 95)) if latencies else float("nan"),
        "campaign_ttd_median_s": float(np.median(campaign)) if campaign else float("nan"),
        "grace_deferred_keys": deferred,
        "grace_resolved_keys": resolved,
        "grace_expired_alerts": expired,
        "markers_seen": markers_seen,
    }


def generate_windows(sim, n, start_id, factory, label, log_every):
    out = []
    started = time.perf_counter()
    for i in range(n):
        out.append(factory(sim, start_id + i, i))
        progress(label, i + 1, n, started, log_every)
    return out


def fit_pair(calibration, target_fpr: float):
    static = V6SemanticQuorumDetector(
        target_fpr,
        transport_mode=TransportMode.BYZ_3OF4,
    ).fit(calibration)
    noisy = SemanticQuorumDetector(
        target_fpr,
        enable_operational_grace=True,
    ).fit(calibration)
    return {ARM_STATIC: static, ARM_NOISY: noisy}


def score_arms(windows, detectors):
    return {arm: [det.score(w) for w in windows] for arm, det in detectors.items()}


def marker_diagnostics(windows) -> Dict[str, float]:
    n = len(windows)
    expected = sum(int(w.marker_expected) for w in windows)
    emitted = sum(int(w.marker_emitted) for w in windows)
    correct = sum(int(w.marker_correct) for w in windows)
    manifested = sum(int(w.transient_manifested) for w in windows)
    recovered = sum(int(w.recovery_emitted) for w in windows)
    false_markers = sum(
        int(any(m.is_false_marker for m in w.operational_markers)) for w in windows
    )

    late_markers = 0
    recovery_before_grace = 0
    recovery_with_timing = 0
    for w in windows:
        if w.latent_operational_state == OperationalState.NORMAL:
            continue
        spec = STATE_SPECS[w.latent_operational_state]
        by_id = {e.event_id: e for e in w.events}
        for event_id in w.operational_event_ids:
            event = by_id.get(event_id)
            if event is None:
                continue
            expiry = event.timestamp + spec.grace_s
            for marker in w.operational_markers:
                if marker.correlation_id == event.correlation_id and marker.fact == spec.fact:
                    late_markers += int(marker.observed_at >= expiry)
            if w.recovery_emitted:
                inconsistent = [
                    o for o in w.observations
                    if o.event_id == event_id
                    and o.fact == spec.fact
                    and o.state.value == "inconsistent"
                ]
                last_bad = max((o.observed_at for o in inconsistent), default=-math.inf)
                recovered_obs = [
                    o for o in w.observations
                    if o.event_id == event_id
                    and o.fact == spec.fact
                    and o.state.value == "consistent"
                    and o.observed_at > last_bad
                ]
                if recovered_obs:
                    recovery_with_timing += 1
                    recovery_before_grace += int(max(o.observed_at for o in recovered_obs) < expiry)

    return {
        "n_windows": n,
        "state_expected_rate": expected / n if n else float("nan"),
        "transient_manifest_rate": manifested / n if n else float("nan"),
        "marker_emission_rate": emitted / expected if expected else emitted / n if n else float("nan"),
        "marker_correct_rate": correct / expected if expected else float("nan"),
        "wrong_or_missing_marker_rate": (expected - correct) / expected if expected else float("nan"),
        "false_marker_window_rate": false_markers / n if n else float("nan"),
        "recovery_emission_rate": recovered / manifested if manifested else float("nan"),
        "recovery_before_grace_rate": recovery_before_grace / recovery_with_timing if recovery_with_timing else float("nan"),
        "late_marker_count": late_markers,
    }


def evaluate_benign(seed, n, detectors, cfg, log_every):
    sim = SemanticSbaSimulator(seed=seed + 10_000, operational_cfg=cfg)
    contexts = list(sim.CONTROL_CONTEXTS)
    windows = generate_windows(
        sim,
        n,
        10_000,
        lambda s, wid, i: s.generate_window(
            wid,
            context=contexts[i % len(contexts)],
            operational_state=OperationalState.NORMAL,
            false_marker=True,
        ),
        f"seed {seed} benign+false-marker",
        log_every,
    )
    scored = score_arms(windows, detectors)
    rng = np.random.default_rng(seed + 515_151)
    labels = rng.integers(0, 2, size=n)
    rows = []
    diag = marker_diagnostics(windows)
    for arm in ARMS:
        m = cell_metrics(windows, scored[arm])
        rows.append({
            "seed": seed,
            "arm": arm,
            "scenario": "benign_false_marker_channel",
            "null_auc": empirical_auc(labels, [r.score for r in scored[arm]]),
            **diag,
            **m,
        })
        LOG.info(
            "  BENIGN %-24s FPR=%.4f semantic=%.4f falseMarker=%.3f nullAUC=%.4f",
            arm,
            m["window_rate"],
            m["semantic_rate"],
            diag["false_marker_window_rate"],
            rows[-1]["null_auc"],
        )
    return rows


def evaluate_attacks(seed, n, hidden, detectors, cfg, log_every):
    rows = []
    families = [f for f in AttackFamily if f != AttackFamily.NONE]
    for j, family in enumerate(families):
        sim = SemanticSbaSimulator(seed=seed + 20_000 + 101 * j, operational_cfg=cfg)
        windows = generate_windows(
            sim,
            n,
            20_000,
            lambda s, wid, _i, family=family: s.generate_window(
                wid,
                attack_family=family,
                hidden_calls=hidden,
                sophistication=Sophistication.ADAPTIVE,
                operational_state=OperationalState.NORMAL,
                false_marker=True,
            ),
            f"seed {seed} attack {family.value}",
            log_every,
        )
        scored = score_arms(windows, detectors)
        diag = marker_diagnostics(windows)
        for arm in ARMS:
            m = cell_metrics(windows, scored[arm])
            rows.append({
                "seed": seed,
                "arm": arm,
                "family": family.value,
                "hidden_calls": hidden,
                **diag,
                **m,
            })
            LOG.info(
                "  ATTACK %-24s %-20s winR=%.3f eventR=%.3f falseMarker=%.3f",
                arm,
                family.value,
                m["window_rate"],
                m["event_recall"],
                diag["false_marker_window_rate"],
            )
    return rows


def evaluate_operational_states(seed, n, detectors, cfg, log_every):
    rows = []
    states = [s for s in OperationalState if s != OperationalState.NORMAL]
    for j, state in enumerate(states):
        sim = SemanticSbaSimulator(seed=seed + 30_000 + 101 * j, operational_cfg=cfg)
        windows = generate_windows(
            sim,
            n,
            30_000,
            lambda s, wid, _i, state=state: s.generate_window(
                wid,
                operational_state=state,
                false_marker=False,
            ),
            f"seed {seed} op-state {state.value}",
            log_every,
        )
        if any(w.label or w.attack_event_ids for w in windows):
            raise AssertionError("operational state leaked into attack ground truth")
        scored = score_arms(windows, detectors)
        diag = marker_diagnostics(windows)
        for arm in ARMS:
            m = cell_metrics(windows, scored[arm])
            rows.append({
                "seed": seed,
                "arm": arm,
                "state": state.value,
                "fact": STATE_SPECS[state].fact.value,
                "operational_fpr": m["window_rate"],
                "semantic_operational_fpr": m["semantic_rate"],
                "transport_operational_fpr": m["transport_rate"],
                **diag,
                **m,
            })
            LOG.info(
                "  OPSTATE %-23s %-18s FPR=%.3f sem=%.3f marker=%.3f correct=%.3f recovery=%.3f beforeGrace=%.3f",
                arm,
                state.value,
                m["window_rate"],
                m["semantic_rate"],
                diag["marker_emission_rate"],
                diag["marker_correct_rate"],
                diag["recovery_emission_rate"],
                diag["recovery_before_grace_rate"],
            )
    return rows


def evaluate_attack_during_state(seed, n, detectors, cfg, log_every):
    rows = []
    families = list(ATTACK_STATE)
    for j, family in enumerate(families):
        sim = SemanticSbaSimulator(seed=seed + 40_000 + 101 * j, operational_cfg=cfg)
        windows = generate_windows(
            sim,
            n,
            40_000,
            lambda s, wid, _i, family=family: s.generate_attack_during_state(
                wid,
                attack_family=family,
                hidden_calls=1,
                sophistication=Sophistication.ADAPTIVE,
            ),
            f"seed {seed} attack-in-state {family.value}",
            log_every,
        )
        scored = score_arms(windows, detectors)
        diag = marker_diagnostics(windows)
        for arm in ARMS:
            m = cell_metrics(windows, scored[arm])
            # Conditional recall among windows with a correct observed marker.
            usable_idx = [i for i, w in enumerate(windows) if w.marker_correct]
            unusable_idx = [i for i, w in enumerate(windows) if not w.marker_correct]
            usable_recall = float(np.mean([scored[arm][i].alert for i in usable_idx])) if usable_idx else float("nan")
            unusable_recall = float(np.mean([scored[arm][i].alert for i in unusable_idx])) if unusable_idx else float("nan")
            rows.append({
                "seed": seed,
                "arm": arm,
                "family": family.value,
                "state": ATTACK_STATE[family].value,
                "scenario": "adaptive_attack_during_genuine_state",
                "correct_marker_window_recall": usable_recall,
                "missing_or_wrong_marker_window_recall": unusable_recall,
                **diag,
                **m,
            })
            LOG.info(
                "  STATE-ATTACK %-19s %-20s winR=%.3f eventR=%.3f lat=%.3fs marker=%.3f correct=%.3f",
                arm,
                family.value,
                m["window_rate"],
                m["event_recall"],
                m["event_latency_median_s"],
                diag["marker_emission_rate"],
                diag["marker_correct_rate"],
            )
    return rows


def seed_summary(seed, benign, attacks, operational, state_attacks):
    rows = []
    for arm in ARMS:
        b = next(r for r in benign if r["arm"] == arm)
        a = [r for r in attacks if r["arm"] == arm]
        o = [r for r in operational if r["arm"] == arm]
        s = [r for r in state_attacks if r["arm"] == arm]
        rows.append({
            "seed": seed,
            "arm": arm,
            "benign_fpr": b["window_rate"],
            "null_auc": b["null_auc"],
            "false_marker_window_rate": b["false_marker_window_rate"],
            "mean_attack_window_recall": float(np.mean([r["window_rate"] for r in a])),
            "mean_attack_event_recall": float(np.mean([r["event_recall"] for r in a])),
            "mean_operational_fpr": float(np.mean([r["operational_fpr"] for r in o])),
            "worst_operational_fpr": float(max(r["operational_fpr"] for r in o)),
            "mean_semantic_operational_fpr": float(np.mean([r["semantic_operational_fpr"] for r in o])),
            "mean_marker_emission_rate": float(np.mean([r["marker_emission_rate"] for r in o])),
            "mean_marker_correct_rate": float(np.mean([r["marker_correct_rate"] for r in o])),
            "mean_recovery_before_grace_rate": float(np.nanmean([r["recovery_before_grace_rate"] for r in o])),
            "state_attack_window_recall": float(np.mean([r["window_rate"] for r in s])),
            "state_attack_event_recall": float(np.mean([r["event_recall"] for r in s])),
            "state_attack_event_latency_s": float(np.nanmean([r["event_latency_median_s"] for r in s])),
            "state_attack_correct_marker_recall": float(np.nanmean([r["correct_marker_window_recall"] for r in s])),
        })
    return rows


def run_seed(
    *, seed, calib_windows, eval_windows, operational_windows,
    state_attack_windows, hidden, target_fpr, cfg, out_dir, log_every,
):
    started = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    LOG.info("=" * 100)
    LOG.info(
        "SEED %d | calib=%d eval=%d operational=%d state-attack=%d h=%d | markerRecall=%.2f falseMarker=%.3f corrErr=%.3f recovery=%.2f",
        seed, calib_windows, eval_windows, operational_windows,
        state_attack_windows, hidden, cfg.marker_recall,
        cfg.false_marker_probability, cfg.correlation_error_probability,
        cfg.recovery_observation_probability,
    )

    sim = SemanticSbaSimulator(seed=seed, operational_cfg=cfg)
    contexts = list(sim.CONTROL_CONTEXTS)
    calibration = generate_windows(
        sim,
        calib_windows,
        0,
        lambda s, wid, i: s.generate_window(
            wid,
            context=contexts[i % len(contexts)],
            operational_state=OperationalState.NORMAL,
            false_marker=False,
        ),
        f"seed {seed} calibration",
        log_every,
    )
    detectors = fit_pair(calibration, target_fpr)

    benign = evaluate_benign(seed, eval_windows, detectors, cfg, log_every)
    attacks = evaluate_attacks(seed, eval_windows, hidden, detectors, cfg, log_every)
    operational = evaluate_operational_states(
        seed, operational_windows, detectors, cfg, log_every
    )
    state_attacks = evaluate_attack_during_state(
        seed, state_attack_windows, detectors, cfg, log_every
    )
    summary = seed_summary(seed, benign, attacks, operational, state_attacks)

    pd.DataFrame(benign).to_csv(out_dir / "benign.csv", index=False)
    pd.DataFrame(attacks).to_csv(out_dir / "attacks.csv", index=False)
    pd.DataFrame(operational).to_csv(out_dir / "operational_states.csv", index=False)
    pd.DataFrame(state_attacks).to_csv(out_dir / "attack_during_state.csv", index=False)
    pd.DataFrame(summary).to_csv(out_dir / "seed_summary.csv", index=False)

    runtime = time.perf_counter() - started
    for row in summary:
        LOG.info(
            "SEED %d %-24s attackR=%.3f opFPR=%.3f stateAttackR=%.3f stateLat=%.3fs markerCorrect=%.3f",
            seed,
            row["arm"],
            row["mean_attack_window_recall"],
            row["mean_operational_fpr"],
            row["state_attack_window_recall"],
            row["state_attack_event_latency_s"],
            row["mean_marker_correct_rate"],
        )
    LOG.info("SEED %d DONE %.1fs", seed, runtime)
    return {
        "benign": pd.DataFrame(benign),
        "attacks": pd.DataFrame(attacks),
        "operational": pd.DataFrame(operational),
        "state_attacks": pd.DataFrame(state_attacks),
        "summary": pd.DataFrame(summary),
    }


def paired_row(df, metric, scope, scenario, seed_offset, n_boot, n_perm):
    pivot = df.pivot_table(index="seed", columns="arm", values=metric, aggfunc="mean")
    pivot = pivot.dropna(subset=list(ARMS))
    if pivot.empty:
        return None
    baseline = pivot[ARM_STATIC].to_numpy(dtype=float)
    candidate = pivot[ARM_NOISY].to_numpy(dtype=float)
    delta = candidate - baseline
    lo, hi = bootstrap_delta_ci(delta, seed_offset, n_boot)
    p = paired_signflip_p(delta, seed_offset + 1, n_perm)
    return {
        "scope": scope,
        "scenario": scenario,
        "metric": metric,
        "n_seeds": len(pivot),
        "static_mean": float(np.mean(baseline)),
        "v8_mean": float(np.mean(candidate)),
        "delta_v8_minus_static": float(np.mean(delta)),
        "delta_ci_lo": lo,
        "delta_ci_hi": hi,
        "paired_signflip_p": p,
    }


def paired_statistics(summaries, attacks, operational, state_attacks, seed, n_boot, n_perm):
    rows = []
    metrics = (
        "benign_fpr",
        "mean_attack_window_recall",
        "mean_attack_event_recall",
        "mean_operational_fpr",
        "worst_operational_fpr",
        "mean_semantic_operational_fpr",
        "state_attack_window_recall",
        "state_attack_event_recall",
        "state_attack_event_latency_s",
    )
    for i, metric in enumerate(metrics):
        row = paired_row(
            summaries, metric, "overall", "ALL",
            seed + 1000 * (i + 1), n_boot, n_perm,
        )
        if row:
            rows.append(row)

    for j, state in enumerate(sorted(operational["state"].unique())):
        part = operational[operational["state"] == state]
        for k, metric in enumerate(("operational_fpr", "semantic_operational_fpr")):
            row = paired_row(
                part, metric, "operational_state", state,
                seed + 100_000 + 1000 * j + k, n_boot, n_perm,
            )
            if row:
                rows.append(row)

    for j, family in enumerate(sorted(state_attacks["family"].unique())):
        part = state_attacks[state_attacks["family"] == family]
        for k, metric in enumerate(("window_rate", "event_recall", "event_latency_median_s")):
            row = paired_row(
                part, metric, "attack_during_state", family,
                seed + 200_000 + 1000 * j + k, n_boot, n_perm,
            )
            if row:
                rows.append(row)

    out = pd.DataFrame(rows)
    if not out.empty:
        out["holm_p"] = holm_adjust(out["paired_signflip_p"].to_numpy())
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="SBA v8 noisy operational-context paired multiseed experiment")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    p.add_argument("--calib-windows", type=int, default=1200)
    p.add_argument("--eval-windows", type=int, default=200)
    p.add_argument("--operational-windows", type=int, default=200)
    p.add_argument("--state-attack-windows", type=int, default=100)
    p.add_argument("--hidden", type=int, default=5)
    p.add_argument("--target-fpr", type=float, default=0.01)
    p.add_argument("--marker-recall", type=float, default=0.85)
    p.add_argument("--false-marker-prob", type=float, default=0.02)
    p.add_argument("--corr-error-prob", type=float, default=0.03)
    p.add_argument("--marker-latency", type=float, default=0.12)
    p.add_argument("--recovery-prob", type=float, default=0.90)
    p.add_argument("--recovery-latency-multiplier", type=float, default=1.0)
    p.add_argument("--manifest-prob", type=float, default=0.95)
    p.add_argument("--bootstrap", type=int, default=5000)
    p.add_argument("--permutations", type=int, default=10000)
    p.add_argument("--log-every", type=int, default=0)
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--out-dir", default="runs/sba_semantic_v8/noisy_context")
    args = p.parse_args(argv)
    setup_logging(args.quiet)

    cfg = OperationalContextConfig(
        marker_recall=args.marker_recall,
        false_marker_probability=args.false_marker_prob,
        correlation_error_probability=args.corr_error_prob,
        marker_latency_median_s=args.marker_latency,
        recovery_observation_probability=args.recovery_prob,
        recovery_latency_multiplier=args.recovery_latency_multiplier,
        disturbance_manifest_probability=args.manifest_prob,
    )
    cfg.validate()
    seeds = [int(x) for x in (args.seeds or [args.seed])]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    LOG.info(
        "V8 noisy context | seeds=%s | calib=%d eval=%d operational=%d stateAttack=%d h=%d targetFPR=%.4f",
        seeds, args.calib_windows, args.eval_windows, args.operational_windows,
        args.state_attack_windows, args.hidden, args.target_fpr,
    )
    LOG.info(
        "Context channel: markerRecall=%.2f falseMarker=%.3f corrErr=%.3f markerLatency=%.3fs recovery=%.2f recoveryMult=%.2f manifest=%.2f",
        cfg.marker_recall, cfg.false_marker_probability,
        cfg.correlation_error_probability, cfg.marker_latency_median_s,
        cfg.recovery_observation_probability, cfg.recovery_latency_multiplier,
        cfg.disturbance_manifest_probability,
    )
    LOG.info("Frozen detector core: semantic=q3of5 persistence=off transport=byz_3of4")

    results = []
    for idx, seed in enumerate(seeds, 1):
        LOG.info("#" * 100)
        LOG.info("MULTISEED %d/%d -> seed=%d", idx, len(seeds), seed)
        seed_dir = out / f"seed_{seed}" if len(seeds) > 1 else out
        results.append(run_seed(
            seed=seed,
            calib_windows=args.calib_windows,
            eval_windows=args.eval_windows,
            operational_windows=args.operational_windows,
            state_attack_windows=args.state_attack_windows,
            hidden=args.hidden,
            target_fpr=args.target_fpr,
            cfg=cfg,
            out_dir=seed_dir,
            log_every=args.log_every,
        ))

    benign = pd.concat([r["benign"] for r in results], ignore_index=True)
    attacks = pd.concat([r["attacks"] for r in results], ignore_index=True)
    operational = pd.concat([r["operational"] for r in results], ignore_index=True)
    state_attacks = pd.concat([r["state_attacks"] for r in results], ignore_index=True)
    summaries = pd.concat([r["summary"] for r in results], ignore_index=True)

    benign.to_csv(out / "all_seed_benign.csv", index=False)
    attacks.to_csv(out / "all_seed_attacks.csv", index=False)
    operational.to_csv(out / "all_seed_operational_states.csv", index=False)
    state_attacks.to_csv(out / "all_seed_attack_during_state.csv", index=False)
    summaries.to_csv(out / "all_seed_summary.csv", index=False)

    stats = paired_statistics(
        summaries, attacks, operational, state_attacks,
        min(seeds), args.bootstrap, args.permutations,
    )
    stats.to_csv(out / "paired_statistics.csv", index=False)

    runtime = time.perf_counter() - started
    primary_metrics = [
        "mean_attack_window_recall",
        "mean_operational_fpr",
        "mean_semantic_operational_fpr",
        "state_attack_window_recall",
        "state_attack_event_latency_s",
    ]
    primary = stats[
        (stats["scope"] == "overall") & stats["metric"].isin(primary_metrics)
    ] if not stats.empty else pd.DataFrame()

    LOG.info("=" * 100)
    LOG.info("FINAL V8 PAIRED SUMMARY")
    if len(seeds) < 3:
        LOG.warning("Only %d seed(s): CI/p-values are diagnostic, not final evidence.", len(seeds))
    for _, row in primary.iterrows():
        LOG.info(
            "%-34s static=%.4f v8=%.4f delta=%+.4f CI95=[%s,%s] p=%s Holm=%s",
            row["metric"], row["static_mean"], row["v8_mean"],
            row["delta_v8_minus_static"],
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
        "operational_context": cfg.__dict__,
        "runtime_seconds": runtime,
        "primary_paired_statistics": primary.to_dict(orient="records") if not primary.empty else [],
    }
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    LOG.info("Runtime %.1fs | output=%s", runtime, out)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
