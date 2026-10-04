"""Research runner for event-level 5G SBA semantic consistency v3.

Adds:
- context-aware FPR + Wilson CI;
- null-world AUC + bootstrap CI;
- detectability curves over hidden-call grids;
- window-level and event-level Recall/Precision;
- TTD (time to first attributed malicious event);
- hidden-volume estimation error;
- trust-domain suppression analysis;
- optional multi-seed aggregation;
- informative terminal progress logs.

The detector never consumes ground truth. Ground truth is used only here, in the
evaluation harness. Current simulator construction appends hidden attack events
after benign events; ``_attack_events`` uses that evaluator-only convention.
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

from sba_semantic_v3 import (
    AttackFamily,
    CompromiseMode,
    DetectionResult,
    SemanticQuorumDetector,
    SemanticSbaSimulator,
    SemanticWindow,
    TrustDomain,
    empirical_auc,
)


LOG = logging.getLogger("sba-semantic-v3")
PROFILE_DEFAULTS = {
    "quick": (36, 20),
    "dev": (1200, 1200),
    "research": (6000, 6000),
}


def _configure_utf8_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure:
            reconfigure(encoding="utf-8", errors="replace")


def _setup_logging(quiet: bool = False) -> None:
    _configure_utf8_stdio()
    level = logging.WARNING if quiet else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )


def wilson_interval(successes: int, n: int, z: float = 1.959963984540054) -> Tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")
    p = float(successes) / float(n)
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
    n_boot: int = 1000,
) -> Tuple[float, float]:
    y = np.asarray(labels, dtype=int)
    s = np.asarray(scores, dtype=float)
    if len(y) < 4 or len(np.unique(y)) < 2 or n_boot <= 0:
        return float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    aucs: List[float] = []
    n = len(y)
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        yb = y[idx]
        if len(np.unique(yb)) < 2:
            continue
        aucs.append(empirical_auc(yb, s[idx]))

    if len(aucs) < 20:
        return float("nan"), float("nan")
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def _progress(
    label: str,
    current: int,
    total: int,
    started: float,
    log_every: int = 0,
) -> None:
    if total <= 0:
        return
    step = int(log_every) if log_every and log_every > 0 else max(1, total // 10)
    if current not in (1, total) and current % step:
        return
    elapsed = max(time.perf_counter() - started, 1e-9)
    rate = current / elapsed
    remaining = max(total - current, 0)
    eta = remaining / rate if rate > 0 else float("nan")
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
    """Evaluator-only ground truth.

    SemanticSbaSimulator appends exactly ``hidden_calls`` malicious semantic
    events after benign events. The detector itself does not access this helper.
    """
    h = int(max(window.hidden_calls, 0))
    if h == 0:
        return []
    if h > len(window.events):
        raise ValueError(
            f"hidden_calls={h} exceeds events={len(window.events)} "
            f"for window {window.window_id}"
        )
    return list(window.events[-h:])


def _event_metrics(window: SemanticWindow, result: DetectionResult) -> Dict[str, float]:
    attacks = _attack_events(window)
    attack_ids = {e.event_id for e in attacks}
    fired_ids = set(result.fired_event_ids)
    detected_ids = attack_ids & fired_ids

    hidden = len(attacks)
    detected = len(detected_ids)
    fired = len(fired_ids)

    event_recall = detected / hidden if hidden else float("nan")
    event_precision = detected / fired if fired else float("nan")

    ttd = float("nan")
    if attacks and detected_ids:
        t0 = min(e.timestamp for e in attacks)
        td = min(e.timestamp for e in attacks if e.event_id in detected_ids)
        ttd = max(float(td - t0), 0.0)

    return {
        "window_alert": float(result.alert),
        "hidden_events": float(hidden),
        "detected_attack_events": float(detected),
        "fired_event_ids": float(fired),
        "event_recall": float(event_recall),
        "event_precision": float(event_precision),
        "ttd_s": float(ttd),
        "detected_hidden_calls": float(detected),
        "hidden_abs_error": float(abs(detected - hidden)),
        "transport_only_alert": float(result.alert and not fired_ids),
    }


def _aggregate_attack_records(
    records: List[Dict[str, float]],
) -> Dict[str, float]:
    if not records:
        return {}

    n = len(records)
    alerts = int(sum(int(r["window_alert"]) for r in records))
    w_lo, w_hi = wilson_interval(alerts, n)

    hidden_total = int(sum(r["hidden_events"] for r in records))
    detected_total = int(sum(r["detected_attack_events"] for r in records))
    e_lo, e_hi = wilson_interval(detected_total, hidden_total)

    fired_total = int(sum(r["fired_event_ids"] for r in records))
    p_lo, p_hi = wilson_interval(detected_total, fired_total) if fired_total else (float("nan"), float("nan"))

    ttd = np.asarray([r["ttd_s"] for r in records if np.isfinite(r["ttd_s"])], dtype=float)

    return {
        "n_windows": n,
        "window_alerts": alerts,
        "window_recall": alerts / n,
        "window_recall_ci_lo": w_lo,
        "window_recall_ci_hi": w_hi,
        "attack_events": hidden_total,
        "detected_attack_events": detected_total,
        "event_recall": detected_total / hidden_total if hidden_total else float("nan"),
        "event_recall_ci_lo": e_lo,
        "event_recall_ci_hi": e_hi,
        "fired_event_ids": fired_total,
        "event_precision": detected_total / fired_total if fired_total else float("nan"),
        "event_precision_ci_lo": p_lo,
        "event_precision_ci_hi": p_hi,
        "ttd_median_s": float(np.median(ttd)) if len(ttd) else float("nan"),
        "ttd_p95_s": float(np.percentile(ttd, 95)) if len(ttd) else float("nan"),
        "ttd_observed_windows": int(len(ttd)),
        "mean_detected_hidden_calls": float(np.mean([r["detected_hidden_calls"] for r in records])),
        "hidden_calls_mae": float(np.mean([r["hidden_abs_error"] for r in records])),
        "transport_only_alert_rate": float(np.mean([r["transport_only_alert"] for r in records])),
    }


def _evaluate_attack_cell(
    sim: SemanticSbaSimulator,
    det: SemanticQuorumDetector,
    *,
    family: AttackFamily,
    hidden: int,
    n_windows: int,
    start_window_id: int,
    context: str,
    compromised_domains: Iterable[TrustDomain] = (),
    compromise_mode: CompromiseMode = CompromiseMode.PASSIVE,
    log_every: int = 0,
    label: str = "attack",
) -> Tuple[Dict[str, float], int]:
    records: List[Dict[str, float]] = []
    facts: Dict[str, int] = {}
    started = time.perf_counter()
    wid = int(start_window_id)

    for i in range(n_windows):
        w = sim.generate_window(
            wid,
            context=context,
            attack_family=family,
            hidden_calls=int(hidden),
            compromised_domains=compromised_domains,
            compromise_mode=compromise_mode,
        )
        wid += 1
        r = det.score(w)
        records.append(_event_metrics(w, r))
        for fact in r.fired_facts:
            facts[fact.value] = facts.get(fact.value, 0) + 1
        _progress(label, i + 1, n_windows, started, log_every)

    out = _aggregate_attack_records(records)
    out["dominant_facts"] = json.dumps(facts, ensure_ascii=False, sort_keys=True)
    return out, wid


def _calibration_guard(
    calib: Sequence[SemanticWindow],
    contexts: Sequence[str],
    target_fpr: float,
) -> Dict[str, int]:
    counts = {c: sum(w.context == c for w in calib) for c in contexts}
    tail_min = int(math.ceil(1.0 / max(target_fpr, 1e-9)))
    recommended = max(500, 10 * tail_min)
    for c in contexts:
        n = counts[c]
        if n < tail_min:
            LOG.warning(
                "Калибровка %s: только %d окон; это меньше теоретического "
                "минимума %d для FPR=%.3f.",
                c, n, tail_min, target_fpr,
            )
        elif n < recommended:
            LOG.warning(
                "Калибровка %s: %d окон. Для устойчивой оценки %.1f%% хвоста "
                "желательно около %d окон.",
                c, n, 100 * target_fpr, recommended,
            )
    return counts


def run_single_seed(
    *,
    seed: int,
    calib_windows: int,
    eval_windows: int,
    hidden_grid: Sequence[int],
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
    total_started = time.perf_counter()

    LOG.info("=" * 78)
    LOG.info(
        "SEED %d | calibration=%d | benign_eval=%d | hidden_grid=%s | target FPR=%.3f",
        seed, calib_windows, eval_windows, list(hidden_grid), target_fpr,
    )
    LOG.info("Output: %s", out_dir)

    sim = SemanticSbaSimulator(seed=seed)
    contexts = list(sim.CONTROL_CONTEXTS)

    LOG.info("[1/5] Генерация benign calibration windows...")
    started = time.perf_counter()
    calib: List[SemanticWindow] = []
    for i in range(calib_windows):
        calib.append(sim.generate_window(i, context=contexts[i % len(contexts)]))
        _progress("calibration", i + 1, calib_windows, started, log_every)

    calib_counts = _calibration_guard(calib, contexts, target_fpr)
    LOG.info("[1/5] Подгонка context-dependent transport thresholds...")
    det = SemanticQuorumDetector(target_fpr=target_fpr).fit(calib)

    threshold_rows = []
    for c in contexts:
        threshold_rows.append({
            "seed": seed,
            "context": c,
            "n_calibration": calib_counts[c],
            "transport_threshold": det.transport_thresholds.get(c, np.nan),
        })
        LOG.info(
            "  context=%-16s n=%4d threshold=%.6f",
            c,
            calib_counts[c],
            det.transport_thresholds.get(c, np.nan),
        )
    pd.DataFrame(threshold_rows).to_csv(
        out_dir / "calibration_thresholds.csv", index=False
    )

    LOG.info("[2/5] Независимая benign evaluation...")
    started = time.perf_counter()
    benign: List[SemanticWindow] = []
    benign_results: List[DetectionResult] = []
    for i in range(eval_windows):
        w = sim.generate_window(10_000 + i, context=contexts[i % len(contexts)])
        benign.append(w)
        benign_results.append(det.score(w))
        _progress("benign-eval", i + 1, eval_windows, started, log_every)

    fpr_rows = []
    for context in contexts + ["ALL"]:
        idx = (
            list(range(len(benign)))
            if context == "ALL"
            else [i for i, w in enumerate(benign) if w.context == context]
        )
        n = len(idx)
        alerts = int(sum(benign_results[i].alert for i in idx))
        lo, hi = wilson_interval(alerts, n)
        total_events = int(sum(len(benign[i].events) for i in idx))
        fired_events = int(sum(len(benign_results[i].fired_event_ids) for i in idx))
        event_fp_per_1000 = (
            1000.0 * fired_events / total_events if total_events else float("nan")
        )
        row = {
            "seed": seed,
            "context": context,
            "n": n,
            "alerts": alerts,
            "fpr": alerts / n if n else float("nan"),
            "fpr_ci_lo": lo,
            "fpr_ci_hi": hi,
            "events": total_events,
            "false_attributed_events": fired_events,
            "false_attributed_events_per_1000": event_fp_per_1000,
        }
        fpr_rows.append(row)
        LOG.info(
            "  FPR %-16s %4d/%-4d = %.4f  CI95=[%.4f, %.4f]  "
            "false-event/1000=%.3f",
            context, alerts, n, row["fpr"], lo, hi, event_fp_per_1000,
        )
    fpr_df = pd.DataFrame(fpr_rows)
    fpr_df.to_csv(out_dir / "fpr_by_context.csv", index=False)

    LOG.info("[3/5] Null-world test...")
    rng = np.random.default_rng(seed + 424242)
    labels = rng.integers(0, 2, size=len(benign))
    scores = np.asarray([r.score for r in benign_results], dtype=float)
    auc = empirical_auc(labels, scores)
    auc_lo, auc_hi = bootstrap_auc_ci(
        labels, scores, seed=seed + 989898, n_boot=auc_bootstrap
    )
    null_test = {
        "seed": seed,
        "n": len(benign),
        "auc": auc,
        "auc_ci_lo": auc_lo,
        "auc_ci_hi": auc_hi,
        "bootstrap_samples": int(auc_bootstrap),
        "pass_practical": bool(np.isfinite(auc) and abs(auc - 0.5) <= 0.10),
    }
    (out_dir / "null_test.json").write_text(
        json.dumps(null_test, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    LOG.info(
        "  Null AUC=%.4f  CI95=[%.4f, %.4f]  practical gate |AUC-0.5|<=0.10: %s",
        auc, auc_lo, auc_hi, "PASS" if null_test["pass_practical"] else "FAIL",
    )

    LOG.info("[4/5] Detectability grid...")
    attack_rows: List[Dict[str, object]] = []
    wid = 20_000
    families = [f for f in AttackFamily if f != AttackFamily.NONE]
    for hidden in hidden_grid:
        LOG.info("  hidden_calls=%d", hidden)
        for family in families:
            cell_started = time.perf_counter()
            metrics, wid = _evaluate_attack_cell(
                sim,
                det,
                family=family,
                hidden=int(hidden),
                n_windows=eval_windows,
                start_window_id=wid,
                context=attack_context,
                log_every=0,
                label=f"h={hidden} {family.value}",
            )
            row: Dict[str, object] = {
                "seed": seed,
                "family": family.value,
                "hidden_calls": int(hidden),
                "context": attack_context,
                **metrics,
            }
            attack_rows.append(row)
            LOG.info(
                "    %-20s windowR=%.4f [%.4f,%.4f] | eventR=%.4f "
                "[%.4f,%.4f] | eventP=%.4f | TTD med/p95=%.2f/%.2fs | "
                "detected h=%.2f | MAE=%.2f | %.1fs",
                family.value,
                row["window_recall"],
                row["window_recall_ci_lo"],
                row["window_recall_ci_hi"],
                row["event_recall"],
                row["event_recall_ci_lo"],
                row["event_recall_ci_hi"],
                row["event_precision"],
                row["ttd_median_s"],
                row["ttd_p95_s"],
                row["mean_detected_hidden_calls"],
                row["hidden_calls_mae"],
                time.perf_counter() - cell_started,
            )

    attack_df = pd.DataFrame(attack_rows)
    attack_df.to_csv(out_dir / "detectability_curve.csv", index=False)
    attack_df.to_csv(out_dir / "per_attack.csv", index=False)

    trust_rows: List[Dict[str, object]] = []
    if skip_trust:
        LOG.info("[5/5] Trust-domain analysis skipped (--skip-trust).")
    else:
        LOG.info(
            "[5/5] Trust-domain SUPPRESS analysis: hidden=%d, n/cell=%d...",
            trust_hidden, trust_eval_windows,
        )
        for domain in TrustDomain:
            LOG.info("  compromised domain=%s", domain.value)
            for family in families:
                metrics, wid = _evaluate_attack_cell(
                    sim,
                    det,
                    family=family,
                    hidden=int(trust_hidden),
                    n_windows=trust_eval_windows,
                    start_window_id=wid,
                    context=attack_context,
                    compromised_domains=[domain],
                    compromise_mode=CompromiseMode.SUPPRESS,
                    log_every=0,
                    label=f"{domain.value} {family.value}",
                )
                row = {
                    "seed": seed,
                    "domain": domain.value,
                    "family": family.value,
                    "hidden_calls": int(trust_hidden),
                    "context": attack_context,
                    **metrics,
                }
                trust_rows.append(row)
                LOG.info(
                    "    %-20s windowR=%.4f eventR=%.4f eventP=%.4f "
                    "TTDmed=%.2fs",
                    family.value,
                    row["window_recall"],
                    row["event_recall"],
                    row["event_precision"],
                    row["ttd_median_s"],
                )
        pd.DataFrame(trust_rows).to_csv(
            out_dir / "trust_domain_suppression.csv", index=False
        )

    primary_hidden = int(trust_hidden)
    primary = attack_df[attack_df["hidden_calls"] == primary_hidden]
    if primary.empty:
        primary = attack_df[attack_df["hidden_calls"] == int(hidden_grid[-1])]

    overall = fpr_df[fpr_df["context"] == "ALL"].iloc[0]
    runtime = time.perf_counter() - total_started
    summary: Dict[str, object] = {
        "seed": seed,
        "calib_windows": int(calib_windows),
        "eval_windows": int(eval_windows),
        "hidden_grid": [int(x) for x in hidden_grid],
        "trust_hidden_calls": int(trust_hidden),
        "trust_eval_windows": int(trust_eval_windows),
        "target_fpr": float(target_fpr),
        "null_auc": float(auc),
        "null_auc_ci95": [float(auc_lo), float(auc_hi)],
        "null_gate_pass": bool(null_test["pass_practical"]),
        "overall_fpr": float(overall["fpr"]),
        "overall_fpr_ci95": [
            float(overall["fpr_ci_lo"]),
            float(overall["fpr_ci_hi"]),
        ],
        "mean_attack_recall": float(primary["window_recall"].mean()),
        "mean_attack_event_recall": float(primary["event_recall"].mean()),
        "mean_attack_event_precision": float(primary["event_precision"].mean()),
        "mean_hidden_calls_mae": float(primary["hidden_calls_mae"].mean()),
        "transport_thresholds": det.transport_thresholds,
        "runtime_seconds": float(runtime),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    LOG.info(
        "SEED %d DONE | FPR=%.4f CI95=[%.4f,%.4f] | null AUC=%.4f | "
        "primary windowR=%.4f | eventR=%.4f | runtime=%.1fs",
        seed,
        summary["overall_fpr"],
        summary["overall_fpr_ci95"][0],
        summary["overall_fpr_ci95"][1],
        summary["null_auc"],
        summary["mean_attack_recall"],
        summary["mean_attack_event_recall"],
        runtime,
    )

    return {
        "summary": summary,
        "fpr": fpr_df,
        "attacks": attack_df,
        "trust": pd.DataFrame(trust_rows),
        "thresholds": pd.DataFrame(threshold_rows),
    }


def _bootstrap_seed_mean_ci(values: Sequence[float], seed: int, n_boot: int = 5000):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    if len(x) == 1:
        v = float(x[0])
        return v, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    bs = rng.choice(x, size=(int(n_boot), len(x)), replace=True).mean(axis=1)
    return (
        float(x.mean()),
        float(np.percentile(bs, 2.5)),
        float(np.percentile(bs, 97.5)),
    )


def _aggregate_multiseed(
    attacks: pd.DataFrame,
    out_dir: Path,
    seed: int,
) -> pd.DataFrame:
    metrics = [
        "window_recall",
        "event_recall",
        "event_precision",
        "ttd_median_s",
        "hidden_calls_mae",
    ]
    rows = []
    for (family, hidden), part in attacks.groupby(["family", "hidden_calls"]):
        row: Dict[str, object] = {
            "family": family,
            "hidden_calls": int(hidden),
            "n_seeds": int(part["seed"].nunique()),
        }
        for j, metric in enumerate(metrics):
            mean, lo, hi = _bootstrap_seed_mean_ci(
                part[metric].to_numpy(),
                seed=seed + 1000 + 101 * j + int(hidden),
            )
            row[f"{metric}_mean"] = mean
            row[f"{metric}_ci_lo"] = lo
            row[f"{metric}_ci_hi"] = hi
            row[f"{metric}_std"] = (
                float(part[metric].std(ddof=1))
                if len(part[metric].dropna()) > 1
                else float("nan")
            )
        rows.append(row)
    agg = pd.DataFrame(rows)
    agg.to_csv(out_dir / "multiseed_aggregate.csv", index=False)
    return agg


def run_many(
    *,
    seeds: Sequence[int],
    calib_windows: int,
    eval_windows: int,
    hidden_grid: Sequence[int],
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
    results = []

    for i, seed in enumerate(seeds, 1):
        LOG.info("#" * 78)
        LOG.info("MULTISEED %d/%d -> seed=%d", i, len(seeds), seed)
        seed_dir = out_dir if len(seeds) == 1 else out_dir / f"seed_{seed}"
        results.append(
            run_single_seed(
                seed=int(seed),
                calib_windows=calib_windows,
                eval_windows=eval_windows,
                hidden_grid=hidden_grid,
                trust_hidden=trust_hidden,
                trust_eval_windows=trust_eval_windows,
                out_dir=seed_dir,
                target_fpr=target_fpr,
                auc_bootstrap=auc_bootstrap,
                log_every=log_every,
                attack_context=attack_context,
                skip_trust=skip_trust,
            )
        )

    if len(seeds) == 1:
        return results[0]["summary"]

    all_fpr = pd.concat([r["fpr"] for r in results], ignore_index=True)
    all_attacks = pd.concat([r["attacks"] for r in results], ignore_index=True)
    all_thresholds = pd.concat([r["thresholds"] for r in results], ignore_index=True)
    all_fpr.to_csv(out_dir / "all_seed_fpr.csv", index=False)
    all_attacks.to_csv(out_dir / "all_seed_detectability.csv", index=False)
    all_thresholds.to_csv(out_dir / "all_seed_thresholds.csv", index=False)

    trust_frames = [r["trust"] for r in results if not r["trust"].empty]
    if trust_frames:
        pd.concat(trust_frames, ignore_index=True).to_csv(
            out_dir / "all_seed_trust_domain_suppression.csv", index=False
        )

    seed_summary = pd.DataFrame([r["summary"] for r in results])
    if "transport_thresholds" in seed_summary:
        seed_summary["transport_thresholds"] = seed_summary[
            "transport_thresholds"
        ].map(lambda x: json.dumps(x, ensure_ascii=False, sort_keys=True))
    seed_summary.to_csv(out_dir / "seed_summary.csv", index=False)

    agg = _aggregate_multiseed(all_attacks, out_dir, seed=min(seeds) + 700000)

    overall_fpr = all_fpr[all_fpr["context"] == "ALL"]
    multiseed_summary = {
        "seeds": [int(s) for s in seeds],
        "n_seeds": len(seeds),
        "calib_windows_per_seed": int(calib_windows),
        "eval_windows_per_seed": int(eval_windows),
        "hidden_grid": [int(x) for x in hidden_grid],
        "target_fpr": float(target_fpr),
        "mean_overall_fpr": float(overall_fpr["fpr"].mean()),
        "std_overall_fpr": float(overall_fpr["fpr"].std(ddof=1)),
        "null_gate_pass_all": bool(
            all(bool(r["summary"]["null_gate_pass"]) for r in results)
        ),
        "aggregate_rows": int(len(agg)),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(multiseed_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    LOG.info("=" * 78)
    LOG.info(
        "MULTISEED DONE | seeds=%s | mean FPR=%.4f | null gates all=%s",
        list(seeds),
        multiseed_summary["mean_overall_fpr"],
        multiseed_summary["null_gate_pass_all"],
    )
    return multiseed_summary


def _resolve_profile(args):
    profile = "quick" if args.quick else args.profile
    default_calib, default_eval = PROFILE_DEFAULTS[profile]
    calib = args.calib_windows if args.calib_windows is not None else default_calib
    eval_n = args.eval_windows if args.eval_windows is not None else default_eval

    if args.hidden_grid:
        hidden_grid = sorted(set(int(x) for x in args.hidden_grid if int(x) > 0))
    elif profile == "research":
        hidden_grid = [1, 2, 5, 10, 20, 40, 80]
    else:
        hidden_grid = [int(args.hidden)]

    if not hidden_grid:
        raise ValueError("hidden grid must contain positive integers")

    trust_hidden = (
        int(args.trust_hidden)
        if args.trust_hidden is not None
        else (int(args.hidden) if int(args.hidden) in hidden_grid else int(hidden_grid[-1]))
    )
    trust_eval = (
        int(args.trust_eval_windows)
        if args.trust_eval_windows is not None
        else min(int(eval_n), 400 if profile != "research" else 1000)
    )
    seeds = [int(s) for s in args.seeds] if args.seeds else [int(args.seed)]

    auc_boot = int(args.auc_bootstrap)
    if profile == "quick" and args.auc_bootstrap == 1000:
        auc_boot = 200

    return profile, int(calib), int(eval_n), hidden_grid, trust_hidden, trust_eval, seeds, auc_boot


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="5G SBA semantic-consistency research runner"
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    p.add_argument("--profile", choices=sorted(PROFILE_DEFAULTS), default="dev")
    p.add_argument("--quick", action="store_true", help="alias for --profile quick")
    p.add_argument("--calib-windows", type=int, default=None)
    p.add_argument("--eval-windows", type=int, default=None)
    p.add_argument("--hidden", type=int, default=40, help="single/default hidden volume")
    p.add_argument(
        "--hidden-grid",
        type=int,
        nargs="+",
        default=None,
        help="e.g. --hidden-grid 1 2 5 10 20 40 80",
    )
    p.add_argument("--trust-hidden", type=int, default=None)
    p.add_argument("--trust-eval-windows", type=int, default=None)
    p.add_argument("--target-fpr", type=float, default=0.01)
    p.add_argument("--attack-context", default="normal_indirect")
    p.add_argument("--auc-bootstrap", type=int, default=1000)
    p.add_argument("--log-every", type=int, default=0)
    p.add_argument("--skip-trust", action="store_true")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--out-dir", default="runs/sba_semantic_v3")
    a = p.parse_args(argv)

    _setup_logging(a.quiet)
    (
        profile,
        calib_windows,
        eval_windows,
        hidden_grid,
        trust_hidden,
        trust_eval,
        seeds,
        auc_boot,
    ) = _resolve_profile(a)

    LOG.info(
        "Profile=%s | seeds=%s | calib=%d | eval=%d | grid=%s | "
        "trust hidden=%d n=%d | target FPR=%.3f",
        profile,
        seeds,
        calib_windows,
        eval_windows,
        hidden_grid,
        trust_hidden,
        trust_eval,
        a.target_fpr,
    )
    if profile == "quick":
        LOG.warning(
            "QUICK profile is a smoke test only; FPR/Recall estimates are not "
            "suitable for dissertation claims."
        )

    summary = run_many(
        seeds=seeds,
        calib_windows=calib_windows,
        eval_windows=eval_windows,
        hidden_grid=hidden_grid,
        trust_hidden=trust_hidden,
        trust_eval_windows=trust_eval,
        out_dir=Path(a.out_dir),
        target_fpr=float(a.target_fpr),
        auc_bootstrap=auc_boot,
        log_every=int(a.log_every),
        attack_context=str(a.attack_context),
        skip_trust=bool(a.skip_trust),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
