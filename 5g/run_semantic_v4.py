"""Research runner for 5G SBA semantic consistency v4.

V4 evaluates partial/correlated evidence and observation-time quorum decisions.
It reports window/event metrics, online TTD, UNKNOWN-evidence rate, separate
semantic/transport false positives, trust-domain suppression/injection, and
NAIVE/STATISTICAL/ADAPTIVE detectability curves.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

from sba_semantic_v4 import (
    AttackFamily,
    CompromiseMode,
    DetectionResult,
    EvidenceState,
    SemanticQuorumDetector,
    SemanticSbaSimulator,
    SemanticWindow,
    Sophistication,
    TrustDomain,
    empirical_auc,
)

LOG = logging.getLogger("sba-semantic-v4")
PROFILE_DEFAULTS = {
    "quick": (60, 30),
    "dev": (1200, 300),
    "research": (6000, 1200),
}


def _configure_utf8_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure:
            reconfigure(encoding="utf-8", errors="replace")


def _setup_logging(quiet: bool = False) -> None:
    _configure_utf8_stdio()
    logging.basicConfig(
        level=logging.WARNING if quiet else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )


def wilson_interval(
    successes: int,
    n: int,
    z: float = 1.959963984540054,
) -> Tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")
    p = successes / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = (
        z
        * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n)
        / denom
    )
    return max(0.0, center - half), min(1.0, center + half)


def bootstrap_auc_ci(
    labels: Sequence[int],
    scores: Sequence[float],
    *,
    seed: int,
    n_boot: int,
) -> Tuple[float, float]:
    y = np.asarray(labels, dtype=int)
    s = np.asarray(scores, dtype=float)
    if len(y) < 4 or len(np.unique(y)) < 2 or n_boot <= 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    aucs: List[float] = []
    for _ in range(int(n_boot)):
        idx = rng.integers(0, len(y), size=len(y))
        yb = y[idx]
        if len(np.unique(yb)) < 2:
            continue
        aucs.append(empirical_auc(yb, s[idx]))
    if len(aucs) < 20:
        return float("nan"), float("nan")
    return (
        float(np.percentile(aucs, 2.5)),
        float(np.percentile(aucs, 97.5)),
    )


def _progress(
    label: str,
    current: int,
    total: int,
    started: float,
    log_every: int = 0,
) -> None:
    if total <= 0:
        return
    step = log_every if log_every > 0 else max(1, total // 10)
    if current not in (1, total) and current % step:
        return
    elapsed = max(time.perf_counter() - started, 1e-9)
    rate = current / elapsed
    eta = (total - current) / rate if rate > 0 else float("nan")
    LOG.info(
        "%s: %d/%d (%.1f%%), %.2f win/s, ETA %.1fs",
        label,
        current,
        total,
        100.0 * current / total,
        rate,
        eta,
    )


def _attack_events(window: SemanticWindow):
    ids = window.attack_event_ids
    return [e for e in window.events if e.event_id in ids]


def _event_metrics(
    window: SemanticWindow,
    result: DetectionResult,
) -> Dict[str, float]:
    attack_events = _attack_events(window)
    attack_ids = {e.event_id for e in attack_events}
    fired_ids = set(result.fired_event_ids)
    detected_ids = attack_ids & fired_ids
    false_ids = fired_ids - attack_ids

    hidden = len(attack_ids)
    detected = len(detected_ids)
    fired = len(fired_ids)

    semantic_ttd = float("nan")
    window_ttd = float("nan")
    if attack_events:
        t0 = min(e.timestamp for e in attack_events)
        detected_times = [
            result.event_alert_times[eid]
            for eid in detected_ids
            if eid in result.event_alert_times
        ]
        if detected_times:
            semantic_ttd = max(min(detected_times) - t0, 0.0)
        if result.alert and np.isfinite(result.first_alert_time):
            window_ttd = max(result.first_alert_time - t0, 0.0)

    return {
        "window_alert": float(result.alert),
        "semantic_window_alert": float(result.semantic_alert),
        "transport_window_alert": float(result.transport_alert),
        "hidden_events": float(hidden),
        "detected_attack_events": float(detected),
        "fired_event_ids": float(fired),
        "false_attributed_events": float(len(false_ids)),
        "event_recall": detected / hidden if hidden else float("nan"),
        "event_precision": detected / fired if fired else float("nan"),
        "semantic_ttd_s": float(semantic_ttd),
        "window_ttd_s": float(window_ttd),
        "detected_hidden_calls": float(detected),
        "hidden_abs_error": float(abs(detected - hidden)),
        "unknown_fraction": float(result.unknown_fraction),
        "transport_stat": float(result.transport_stat),
        "transport_threshold": float(result.transport_threshold),
    }


def _aggregate_attack_records(
    records: List[Dict[str, float]],
) -> Dict[str, float]:
    if not records:
        return {}

    n = len(records)
    alerts = int(sum(r["window_alert"] for r in records))
    semantic_alerts = int(sum(r["semantic_window_alert"] for r in records))
    transport_alerts = int(sum(r["transport_window_alert"] for r in records))
    w_lo, w_hi = wilson_interval(alerts, n)

    hidden_total = int(sum(r["hidden_events"] for r in records))
    detected_total = int(sum(r["detected_attack_events"] for r in records))
    e_lo, e_hi = wilson_interval(detected_total, hidden_total)

    fired_total = int(sum(r["fired_event_ids"] for r in records))
    false_total = int(sum(r["false_attributed_events"] for r in records))
    if fired_total:
        p_lo, p_hi = wilson_interval(detected_total, fired_total)
    else:
        p_lo, p_hi = float("nan"), float("nan")

    semantic_ttd = np.asarray(
        [r["semantic_ttd_s"] for r in records if np.isfinite(r["semantic_ttd_s"])],
        dtype=float,
    )
    window_ttd = np.asarray(
        [r["window_ttd_s"] for r in records if np.isfinite(r["window_ttd_s"])],
        dtype=float,
    )

    return {
        "n_windows": n,
        "window_alerts": alerts,
        "window_recall": alerts / n,
        "window_recall_ci_lo": w_lo,
        "window_recall_ci_hi": w_hi,
        "semantic_window_alert_rate": semantic_alerts / n,
        "transport_window_alert_rate": transport_alerts / n,
        "attack_events": hidden_total,
        "detected_attack_events": detected_total,
        "event_recall": detected_total / hidden_total if hidden_total else float("nan"),
        "event_recall_ci_lo": e_lo,
        "event_recall_ci_hi": e_hi,
        "fired_event_ids": fired_total,
        "false_attributed_events": false_total,
        "event_precision": detected_total / fired_total if fired_total else float("nan"),
        "event_precision_ci_lo": p_lo,
        "event_precision_ci_hi": p_hi,
        "semantic_ttd_median_s": float(np.median(semantic_ttd)) if len(semantic_ttd) else float("nan"),
        "semantic_ttd_p95_s": float(np.percentile(semantic_ttd, 95)) if len(semantic_ttd) else float("nan"),
        "window_ttd_median_s": float(np.median(window_ttd)) if len(window_ttd) else float("nan"),
        "window_ttd_p95_s": float(np.percentile(window_ttd, 95)) if len(window_ttd) else float("nan"),
        "mean_detected_hidden_calls": float(np.mean([r["detected_hidden_calls"] for r in records])),
        "hidden_calls_mae": float(np.mean([r["hidden_abs_error"] for r in records])),
        "mean_unknown_fraction": float(np.nanmean([r["unknown_fraction"] for r in records])),
    }


def _evidence_diagnostics(
    windows: Sequence[SemanticWindow],
    seed: int,
) -> pd.DataFrame:
    counts: Dict[Tuple[str, str, str], Dict[str, int]] = {}
    for window in windows:
        for obs in window.observations:
            key = (window.context, obs.fact.value, obs.domain.value)
            row = counts.setdefault(
                key,
                {"n": 0, "unknown": 0, "inconsistent": 0, "consistent": 0},
            )
            row["n"] += 1
            row[obs.state.value] += 1

    rows = []
    for (context, fact, domain), row in counts.items():
        n = row["n"]
        rows.append(
            {
                "seed": seed,
                "context": context,
                "fact": fact,
                "domain": domain,
                "n": n,
                "unknown_rate": row["unknown"] / n,
                "inconsistent_rate": row["inconsistent"] / n,
                "consistent_rate": row["consistent"] / n,
            }
        )
    return pd.DataFrame(rows)


def _evaluate_attack_cell(
    sim: SemanticSbaSimulator,
    det: SemanticQuorumDetector,
    *,
    family: AttackFamily,
    sophistication: Sophistication,
    hidden: int,
    n_windows: int,
    start_window_id: int,
    context: str,
    compromised_domains: Iterable[TrustDomain] = (),
    compromise_mode: CompromiseMode = CompromiseMode.PASSIVE,
    log_every: int = 0,
) -> Tuple[Dict[str, float], int]:
    records: List[Dict[str, float]] = []
    facts: Dict[str, int] = {}
    started = time.perf_counter()
    wid = start_window_id

    label = f"{sophistication.value} h={hidden} {family.value}"
    for i in range(n_windows):
        window = sim.generate_window(
            wid,
            context=context,
            attack_family=family,
            hidden_calls=hidden,
            sophistication=sophistication,
            compromised_domains=compromised_domains,
            compromise_mode=compromise_mode,
        )
        wid += 1
        result = det.score(window)
        records.append(_event_metrics(window, result))
        for fact in result.fired_facts:
            facts[fact.value] = facts.get(fact.value, 0) + 1
        _progress(label, i + 1, n_windows, started, log_every)

    out = _aggregate_attack_records(records)
    out["dominant_facts"] = json.dumps(facts, ensure_ascii=False, sort_keys=True)
    return out, wid


def _calibration_guard(
    calibration: Sequence[SemanticWindow],
    contexts: Sequence[str],
    target_fpr: float,
) -> Dict[str, int]:
    counts = {c: sum(w.context == c for w in calibration) for c in contexts}
    minimum = int(math.ceil(1.0 / max(target_fpr, 1e-9)))
    recommended = max(500, 10 * minimum)
    for context, n in counts.items():
        if n < minimum:
            LOG.warning(
                "Калибровка %s: %d окон < теоретического минимума %d для FPR=%.3f",
                context,
                n,
                minimum,
                target_fpr,
            )
        elif n < recommended:
            LOG.warning(
                "Калибровка %s: %d окон; для устойчивой оценки %.1f%% хвоста желательно ~%d",
                context,
                n,
                100 * target_fpr,
                recommended,
            )
    return counts


def run_single_seed(
    *,
    seed: int,
    calib_windows: int,
    eval_windows: int,
    hidden_grid: Sequence[int],
    sophistication_grid: Sequence[Sophistication],
    trust_hidden: int,
    trust_eval_windows: int,
    out_dir: Path,
    target_fpr: float,
    auc_bootstrap: int,
    log_every: int,
    attack_context: str,
    skip_trust: bool,
) -> Dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    started_total = time.perf_counter()
    sim = SemanticSbaSimulator(seed=seed)
    contexts = list(sim.CONTROL_CONTEXTS)

    LOG.info("=" * 88)
    LOG.info(
        "SEED %d | calib=%d | benign_eval=%d | hidden=%s | sophistication=%s | target FPR=%.3f",
        seed,
        calib_windows,
        eval_windows,
        list(hidden_grid),
        [s.value for s in sophistication_grid],
        target_fpr,
    )
    LOG.info("Output: %s", out_dir)

    LOG.info("[1/6] Генерация calibration world...")
    t0 = time.perf_counter()
    calibration: List[SemanticWindow] = []
    for i in range(calib_windows):
        calibration.append(
            sim.generate_window(i, context=contexts[i % len(contexts)])
        )
        _progress("calibration", i + 1, calib_windows, t0, log_every)

    counts = _calibration_guard(calibration, contexts, target_fpr)
    detector = SemanticQuorumDetector(target_fpr).fit(calibration)
    threshold_rows = []
    LOG.info("[1/6] Context-dependent transport thresholds:")
    for context in contexts:
        threshold = detector.transport_thresholds.get(context, np.nan)
        threshold_rows.append(
            {
                "seed": seed,
                "context": context,
                "n_calibration": counts[context],
                "transport_threshold": threshold,
            }
        )
        LOG.info(
            "  %-16s n=%4d threshold=%.6f",
            context,
            counts[context],
            threshold,
        )
    pd.DataFrame(threshold_rows).to_csv(
        out_dir / "calibration_thresholds.csv", index=False
    )

    LOG.info("[2/6] Independent benign evaluation...")
    benign: List[SemanticWindow] = []
    benign_results: List[DetectionResult] = []
    t0 = time.perf_counter()
    for i in range(eval_windows):
        window = sim.generate_window(
            10_000 + i,
            context=contexts[i % len(contexts)],
        )
        benign.append(window)
        benign_results.append(detector.score(window))
        _progress("benign-eval", i + 1, eval_windows, t0, log_every)

    fpr_rows = []
    for context in contexts + ["ALL"]:
        idx = (
            list(range(len(benign)))
            if context == "ALL"
            else [i for i, w in enumerate(benign) if w.context == context]
        )
        n = len(idx)
        alerts = int(sum(benign_results[i].alert for i in idx))
        semantic_alerts = int(sum(benign_results[i].semantic_alert for i in idx))
        transport_alerts = int(sum(benign_results[i].transport_alert for i in idx))
        both_alerts = int(
            sum(
                benign_results[i].semantic_alert
                and benign_results[i].transport_alert
                for i in idx
            )
        )
        lo, hi = wilson_interval(alerts, n)
        total_events = int(sum(len(benign[i].events) for i in idx))
        false_event_ids = int(sum(len(benign_results[i].fired_event_ids) for i in idx))
        unknown = float(
            np.nanmean([benign_results[i].unknown_fraction for i in idx])
        ) if idx else float("nan")
        row = {
            "seed": seed,
            "context": context,
            "n": n,
            "alerts": alerts,
            "fpr": alerts / n if n else float("nan"),
            "fpr_ci_lo": lo,
            "fpr_ci_hi": hi,
            "semantic_alerts": semantic_alerts,
            "semantic_fpr": semantic_alerts / n if n else float("nan"),
            "transport_alerts": transport_alerts,
            "transport_fpr": transport_alerts / n if n else float("nan"),
            "both_alerts": both_alerts,
            "events": total_events,
            "false_attributed_events": false_event_ids,
            "false_attributed_events_per_1000": (
                1000 * false_event_ids / total_events if total_events else float("nan")
            ),
            "mean_unknown_fraction": unknown,
        }
        fpr_rows.append(row)
        LOG.info(
            "  FPR %-16s %3d/%-3d=%.4f CI95=[%.4f,%.4f] | semantic=%.4f transport=%.4f | UNKNOWN=%.3f",
            context,
            alerts,
            n,
            row["fpr"],
            lo,
            hi,
            row["semantic_fpr"],
            row["transport_fpr"],
            unknown,
        )
    fpr_df = pd.DataFrame(fpr_rows)
    fpr_df.to_csv(out_dir / "fpr_by_context.csv", index=False)

    evidence_df = _evidence_diagnostics(benign, seed)
    evidence_df.to_csv(out_dir / "evidence_diagnostics.csv", index=False)
    LOG.info(
        "  Benign evidence UNKNOWN mean=%.3f, max cell=%.3f",
        evidence_df["unknown_rate"].mean(),
        evidence_df["unknown_rate"].max(),
    )

    LOG.info("[3/6] Null-world gate...")
    rng = np.random.default_rng(seed + 424242)
    labels = rng.integers(0, 2, size=len(benign))
    scores = np.asarray([r.score for r in benign_results], dtype=float)
    auc = empirical_auc(labels, scores)
    auc_lo, auc_hi = bootstrap_auc_ci(
        labels,
        scores,
        seed=seed + 989898,
        n_boot=auc_bootstrap,
    )
    null_test = {
        "seed": seed,
        "n": len(benign),
        "auc": auc,
        "auc_ci_lo": auc_lo,
        "auc_ci_hi": auc_hi,
        "pass_practical": bool(np.isfinite(auc) and abs(auc - 0.5) <= 0.10),
    }
    (out_dir / "null_test.json").write_text(
        json.dumps(null_test, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    LOG.info(
        "  Null AUC=%.4f CI95=[%.4f,%.4f] -> %s",
        auc,
        auc_lo,
        auc_hi,
        "PASS" if null_test["pass_practical"] else "FAIL",
    )

    LOG.info("[4/6] Detectability surface sophistication × hidden × attack...")
    families = [f for f in AttackFamily if f != AttackFamily.NONE]
    attack_rows: List[Dict[str, object]] = []
    wid = 20_000
    for sophistication in sophistication_grid:
        LOG.info("  sophistication=%s", sophistication.value.upper())
        for hidden in hidden_grid:
            LOG.info("    hidden_calls=%d", hidden)
            for family in families:
                cell_start = time.perf_counter()
                metrics, wid = _evaluate_attack_cell(
                    sim,
                    detector,
                    family=family,
                    sophistication=sophistication,
                    hidden=int(hidden),
                    n_windows=eval_windows,
                    start_window_id=wid,
                    context=attack_context,
                    log_every=log_every,
                )
                row = {
                    "seed": seed,
                    "sophistication": sophistication.value,
                    "family": family.value,
                    "hidden_calls": int(hidden),
                    "context": attack_context,
                    **metrics,
                }
                attack_rows.append(row)
                LOG.info(
                    "      %-20s winR=%.3f eventR=%.3f eventP=%.3f | semTTD med/p95=%.3f/%.3fs | winTTD=%.3fs | UNKNOWN=%.3f | sem/transport=%.3f/%.3f | %.1fs",
                    family.value,
                    row["window_recall"],
                    row["event_recall"],
                    row["event_precision"],
                    row["semantic_ttd_median_s"],
                    row["semantic_ttd_p95_s"],
                    row["window_ttd_median_s"],
                    row["mean_unknown_fraction"],
                    row["semantic_window_alert_rate"],
                    row["transport_window_alert_rate"],
                    time.perf_counter() - cell_start,
                )
    attack_df = pd.DataFrame(attack_rows)
    attack_df.to_csv(out_dir / "detectability_surface.csv", index=False)
    attack_df.to_csv(out_dir / "per_attack.csv", index=False)

    trust_rows: List[Dict[str, object]] = []
    inject_rows: List[Dict[str, object]] = []
    if skip_trust:
        LOG.info("[5/6] Trust-domain SUPPRESS skipped.")
        LOG.info("[6/6] Trust-domain INJECT skipped.")
    else:
        LOG.info(
            "[5/6] Trust-domain SUPPRESS | adaptive | h=%d | n/cell=%d",
            trust_hidden,
            trust_eval_windows,
        )
        for domain in TrustDomain:
            for family in families:
                metrics, wid = _evaluate_attack_cell(
                    sim,
                    detector,
                    family=family,
                    sophistication=Sophistication.ADAPTIVE,
                    hidden=trust_hidden,
                    n_windows=trust_eval_windows,
                    start_window_id=wid,
                    context=attack_context,
                    compromised_domains=[domain],
                    compromise_mode=CompromiseMode.SUPPRESS,
                    log_every=log_every,
                )
                trust_rows.append(
                    {
                        "seed": seed,
                        "domain": domain.value,
                        "family": family.value,
                        "sophistication": Sophistication.ADAPTIVE.value,
                        "hidden_calls": trust_hidden,
                        **metrics,
                    }
                )
            domain_df = pd.DataFrame(trust_rows)
            part = domain_df[domain_df["domain"] == domain.value]
            LOG.info(
                "  SUPPRESS %-8s mean windowR=%.3f eventR=%.3f UNKNOWN=%.3f",
                domain.value,
                part["window_recall"].mean(),
                part["event_recall"].mean(),
                part["mean_unknown_fraction"].mean(),
            )
        pd.DataFrame(trust_rows).to_csv(
            out_dir / "trust_domain_suppression.csv", index=False
        )

        LOG.info(
            "[6/6] Trust-domain INJECT on benign world | n/domain=%d",
            trust_eval_windows,
        )
        for domain in TrustDomain:
            t0 = time.perf_counter()
            results: List[DetectionResult] = []
            for i in range(trust_eval_windows):
                window = sim.generate_window(
                    wid,
                    context=contexts[i % len(contexts)],
                    compromised_domains=[domain],
                    compromise_mode=CompromiseMode.INJECT,
                )
                wid += 1
                results.append(detector.score(window))
                _progress(
                    f"INJECT {domain.value}",
                    i + 1,
                    trust_eval_windows,
                    t0,
                    log_every,
                )
            n = len(results)
            alerts = sum(r.alert for r in results)
            semantic_alerts = sum(r.semantic_alert for r in results)
            transport_alerts = sum(r.transport_alert for r in results)
            row = {
                "seed": seed,
                "domain": domain.value,
                "n": n,
                "false_alarm_rate": alerts / n,
                "semantic_false_alarm_rate": semantic_alerts / n,
                "transport_false_alarm_rate": transport_alerts / n,
                "mean_unknown_fraction": float(np.nanmean([r.unknown_fraction for r in results])),
            }
            inject_rows.append(row)
            LOG.info(
                "  INJECT %-8s FPR=%.4f semantic=%.4f transport=%.4f",
                domain.value,
                row["false_alarm_rate"],
                row["semantic_false_alarm_rate"],
                row["transport_false_alarm_rate"],
            )
        pd.DataFrame(inject_rows).to_csv(
            out_dir / "trust_domain_injection.csv", index=False
        )

    overall = fpr_df[fpr_df["context"] == "ALL"].iloc[0]
    primary = attack_df[
        (attack_df["sophistication"] == Sophistication.ADAPTIVE.value)
        & (attack_df["hidden_calls"] == trust_hidden)
    ]
    if primary.empty:
        primary = attack_df[
            attack_df["sophistication"] == sophistication_grid[-1].value
        ]
        if not primary.empty:
            primary = primary[
                primary["hidden_calls"] == primary["hidden_calls"].max()
            ]

    runtime = time.perf_counter() - started_total
    summary = {
        "seed": seed,
        "calib_windows": calib_windows,
        "eval_windows": eval_windows,
        "hidden_grid": [int(x) for x in hidden_grid],
        "sophistication_grid": [s.value for s in sophistication_grid],
        "target_fpr": target_fpr,
        "overall_fpr": float(overall["fpr"]),
        "semantic_fpr": float(overall["semantic_fpr"]),
        "transport_fpr": float(overall["transport_fpr"]),
        "overall_fpr_ci95": [float(overall["fpr_ci_lo"]), float(overall["fpr_ci_hi"])],
        "null_auc": float(auc),
        "null_auc_ci95": [float(auc_lo), float(auc_hi)],
        "null_gate_pass": bool(null_test["pass_practical"]),
        "mean_benign_unknown_fraction": float(overall["mean_unknown_fraction"]),
        "primary_mean_window_recall": float(primary["window_recall"].mean()) if not primary.empty else float("nan"),
        "primary_mean_event_recall": float(primary["event_recall"].mean()) if not primary.empty else float("nan"),
        "primary_semantic_ttd_median_s": float(primary["semantic_ttd_median_s"].median()) if not primary.empty else float("nan"),
        "primary_hidden_calls_mae": float(primary["hidden_calls_mae"].mean()) if not primary.empty else float("nan"),
        "runtime_seconds": runtime,
        "transport_thresholds": detector.transport_thresholds,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    LOG.info(
        "SEED %d DONE | FPR=%.4f (semantic=%.4f transport=%.4f) | null AUC=%.4f | primary winR=%.3f eventR=%.3f semTTD=%.3fs | runtime=%.1fs",
        seed,
        summary["overall_fpr"],
        summary["semantic_fpr"],
        summary["transport_fpr"],
        summary["null_auc"],
        summary["primary_mean_window_recall"],
        summary["primary_mean_event_recall"],
        summary["primary_semantic_ttd_median_s"],
        runtime,
    )

    return {
        "summary": summary,
        "fpr": fpr_df,
        "attacks": attack_df,
        "evidence": evidence_df,
        "trust_suppress": pd.DataFrame(trust_rows),
        "trust_inject": pd.DataFrame(inject_rows),
    }


def _bootstrap_seed_mean_ci(
    values: Sequence[float],
    *,
    seed: int,
    n_boot: int = 5000,
) -> Tuple[float, float, float]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    if len(x) == 1:
        return float(x[0]), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    bs = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    return (
        float(x.mean()),
        float(np.percentile(bs, 2.5)),
        float(np.percentile(bs, 97.5)),
    )


def run_many(
    *,
    seeds: Sequence[int],
    out_dir: Path,
    **kwargs,
) -> Dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for i, seed in enumerate(seeds, 1):
        LOG.info("#" * 88)
        LOG.info("MULTISEED %d/%d -> seed=%d", i, len(seeds), seed)
        seed_dir = out_dir if len(seeds) == 1 else out_dir / f"seed_{seed}"
        results.append(
            run_single_seed(seed=int(seed), out_dir=seed_dir, **kwargs)
        )

    if len(seeds) == 1:
        return results[0]["summary"]

    attacks = pd.concat([r["attacks"] for r in results], ignore_index=True)
    fpr = pd.concat([r["fpr"] for r in results], ignore_index=True)
    evidence = pd.concat([r["evidence"] for r in results], ignore_index=True)
    attacks.to_csv(out_dir / "all_seed_detectability.csv", index=False)
    fpr.to_csv(out_dir / "all_seed_fpr.csv", index=False)
    evidence.to_csv(out_dir / "all_seed_evidence_diagnostics.csv", index=False)

    trust_s = [r["trust_suppress"] for r in results if not r["trust_suppress"].empty]
    trust_i = [r["trust_inject"] for r in results if not r["trust_inject"].empty]
    if trust_s:
        pd.concat(trust_s, ignore_index=True).to_csv(
            out_dir / "all_seed_trust_domain_suppression.csv", index=False
        )
    if trust_i:
        pd.concat(trust_i, ignore_index=True).to_csv(
            out_dir / "all_seed_trust_domain_injection.csv", index=False
        )

    metrics = [
        "window_recall",
        "event_recall",
        "event_precision",
        "semantic_ttd_median_s",
        "window_ttd_median_s",
        "hidden_calls_mae",
        "mean_unknown_fraction",
    ]
    agg_rows = []
    for (soph, family, hidden), part in attacks.groupby(
        ["sophistication", "family", "hidden_calls"]
    ):
        row: Dict[str, object] = {
            "sophistication": soph,
            "family": family,
            "hidden_calls": int(hidden),
            "n_seeds": int(part["seed"].nunique()),
        }
        for j, metric in enumerate(metrics):
            mean, lo, hi = _bootstrap_seed_mean_ci(
                part[metric].to_numpy(),
                seed=min(seeds) + 900_000 + 97 * j + int(hidden),
            )
            row[f"{metric}_mean"] = mean
            row[f"{metric}_ci_lo"] = lo
            row[f"{metric}_ci_hi"] = hi
        agg_rows.append(row)
    aggregate = pd.DataFrame(agg_rows)
    aggregate.to_csv(out_dir / "multiseed_aggregate.csv", index=False)

    overall_fpr = fpr[fpr["context"] == "ALL"]
    summary = {
        "seeds": [int(s) for s in seeds],
        "n_seeds": len(seeds),
        "mean_overall_fpr": float(overall_fpr["fpr"].mean()),
        "mean_semantic_fpr": float(overall_fpr["semantic_fpr"].mean()),
        "mean_transport_fpr": float(overall_fpr["transport_fpr"].mean()),
        "null_gate_pass_all": all(r["summary"]["null_gate_pass"] for r in results),
        "aggregate_rows": len(aggregate),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    LOG.info(
        "MULTISEED DONE | seeds=%s | FPR=%.4f semantic=%.4f transport=%.4f | null all=%s",
        list(seeds),
        summary["mean_overall_fpr"],
        summary["mean_semantic_fpr"],
        summary["mean_transport_fpr"],
        summary["null_gate_pass_all"],
    )
    return summary


def _resolve(args):
    profile = "quick" if args.quick else args.profile
    default_calib, default_eval = PROFILE_DEFAULTS[profile]
    calib = args.calib_windows or default_calib
    eval_n = args.eval_windows or default_eval

    if args.hidden_grid:
        hidden = sorted(set(int(x) for x in args.hidden_grid if int(x) > 0))
    elif profile == "research":
        hidden = [1, 2, 5, 10, 20, 40, 80]
    else:
        hidden = [int(args.hidden)]

    if args.sophistication_grid:
        sophistication = [Sophistication(x) for x in args.sophistication_grid]
    elif profile == "research":
        sophistication = list(Sophistication)
    else:
        sophistication = [Sophistication(args.sophistication)]

    trust_hidden = args.trust_hidden or (
        args.hidden if args.hidden in hidden else hidden[-1]
    )
    trust_eval = args.trust_eval_windows or min(
        eval_n, 200 if profile != "research" else 600
    )
    seeds = args.seeds or [args.seed]
    auc_boot = args.auc_bootstrap
    if profile == "quick" and auc_boot == 1000:
        auc_boot = 200

    return (
        profile,
        int(calib),
        int(eval_n),
        hidden,
        sophistication,
        int(trust_hidden),
        int(trust_eval),
        [int(s) for s in seeds],
        int(auc_boot),
    )


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="5G SBA semantic consistency v4: online partial evidence"
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    p.add_argument("--profile", choices=sorted(PROFILE_DEFAULTS), default="dev")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--calib-windows", type=int, default=None)
    p.add_argument("--eval-windows", type=int, default=None)
    p.add_argument("--hidden", type=int, default=40)
    p.add_argument("--hidden-grid", type=int, nargs="+", default=None)
    p.add_argument(
        "--sophistication",
        choices=[s.value for s in Sophistication],
        default=Sophistication.ADAPTIVE.value,
    )
    p.add_argument(
        "--sophistication-grid",
        choices=[s.value for s in Sophistication],
        nargs="+",
        default=None,
    )
    p.add_argument("--trust-hidden", type=int, default=None)
    p.add_argument("--trust-eval-windows", type=int, default=None)
    p.add_argument("--target-fpr", type=float, default=0.01)
    p.add_argument("--attack-context", default="normal_indirect")
    p.add_argument("--auc-bootstrap", type=int, default=1000)
    p.add_argument("--log-every", type=int, default=0)
    p.add_argument("--skip-trust", action="store_true")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--out-dir", default="runs/sba_semantic_v4")
    args = p.parse_args(argv)

    _setup_logging(args.quiet)
    (
        profile,
        calib,
        eval_n,
        hidden,
        sophistication,
        trust_hidden,
        trust_eval,
        seeds,
        auc_boot,
    ) = _resolve(args)

    LOG.info(
        "Profile=%s | seeds=%s | calib=%d | eval=%d | hidden=%s | sophistication=%s | trust h=%d n=%d",
        profile,
        seeds,
        calib,
        eval_n,
        hidden,
        [s.value for s in sophistication],
        trust_hidden,
        trust_eval,
    )
    if profile == "quick":
        LOG.warning("QUICK is smoke-test only; do not use its FPR/Recall for dissertation claims.")

    summary = run_many(
        seeds=seeds,
        out_dir=Path(args.out_dir),
        calib_windows=calib,
        eval_windows=eval_n,
        hidden_grid=hidden,
        sophistication_grid=sophistication,
        trust_hidden=trust_hidden,
        trust_eval_windows=trust_eval,
        target_fpr=float(args.target_fpr),
        auc_bootstrap=auc_boot,
        log_every=int(args.log_every),
        attack_context=str(args.attack_context),
        skip_trust=bool(args.skip_trust),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
