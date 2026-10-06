from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import rankdata, t as student_t, wilcoxon

from cascade_detector import fit_guarded_cascade_family
from config import ExperimentConfig
from decision_fusion import fit_decision_fusion_family
from features import TemporalState, add_features
from models import FEATURE_SETS, binary_metrics, detection_latency_windows, fit_iforest
from simulator import AttackSpec, FiveGCoreSimulator


ATTACKS = [
    "compromised_lie",
    "adaptive_lie",
    "compromised_truth",
    "slow_drift_lie",
    "fake_nf",
    "rapid_reset",
    "oauth_abuse",
    "nrf_poisoning",
    "cross_slice",
    "coordinated_majority",
]

STANDARD_SBI_ATTACKS = [
    "rapid_reset",
    "oauth_abuse",
    "nrf_poisoning",
    "cross_slice",
]


def _featurize(seed: int, cfg: ExperimentConfig, attack: AttackSpec | None, n_windows: int) -> pd.DataFrame:
    sim = FiveGCoreSimulator(cfg.fleet, seed=seed)
    warm = sim.generate(cfg.warmup_windows)
    body = sim.generate(n_windows, attack=attack)
    state = TemporalState()
    _ = add_features(warm, temporal=state)
    return add_features(body, temporal=state)


def _seed_sequence(n: int) -> tuple[int, ...]:
    return tuple(101 + 101 * i for i in range(n))


def _aggregate(metrics: pd.DataFrame) -> pd.DataFrame:
    cols = ["precision", "recall", "f1", "fpr", "pr_auc", "latency_windows"]
    g = (
        metrics.groupby(["method", "scenario", "severity"], dropna=False)[cols]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    g.columns = [
        "_".join([c for c in tup if c]) if isinstance(tup, tuple) else tup
        for tup in g.columns
    ]
    return g


def _rank_biserial(diff: np.ndarray) -> float:
    d = np.asarray(diff, dtype=float)
    d = d[~np.isclose(d, 0.0)]
    if d.size == 0:
        return 0.0
    ranks = rankdata(np.abs(d), method="average")
    wp = float(ranks[d > 0].sum())
    wm = float(ranks[d < 0].sum())
    return 0.0 if wp + wm == 0 else (wp - wm) / (wp + wm)


def _paired_stats(metrics: pd.DataFrame, baseline: str = "iforest_sbi") -> pd.DataFrame:
    rows: List[dict] = []
    attacks = metrics[metrics["scenario"] != "normal"]
    for method in sorted(set(attacks["method"]) - {baseline}):
        for scenario in sorted(attacks["scenario"].unique()):
            for severity in sorted(attacks["severity"].dropna().unique()):
                sub = attacks[(attacks["scenario"] == scenario) & (attacks["severity"] == severity)]
                a = sub[sub["method"] == method].set_index("seed")["recall"]
                b = sub[sub["method"] == baseline].set_index("seed")["recall"]
                common = a.index.intersection(b.index)
                if len(common) < 2:
                    continue
                diff = (a.loc[common] - b.loc[common]).to_numpy(float)
                if np.allclose(diff, 0):
                    stat, p = 0.0, 1.0
                else:
                    stat, p = wilcoxon(diff, zero_method="wilcox", alternative="two-sided")
                n = len(diff)
                mean = float(diff.mean())
                sd = float(diff.std(ddof=1)) if n > 1 else 0.0
                se = sd / np.sqrt(n) if n > 1 else 0.0
                crit = float(student_t.ppf(0.975, n - 1)) if n > 1 else 0.0
                rows.append({
                    "method": method,
                    "baseline": baseline,
                    "scenario": scenario,
                    "severity": float(severity),
                    "n_pairs": n,
                    "mean_recall_gain": mean,
                    "median_recall_gain": float(np.median(diff)),
                    "gain_ci95_low": mean - crit * se,
                    "gain_ci95_high": mean + crit * se,
                    "rank_biserial": float(_rank_biserial(diff)),
                    "wilcoxon_stat": float(stat),
                    "p_value": float(p),
                })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    order = np.argsort(out["p_value"].to_numpy(float))
    p = out["p_value"].to_numpy(float)
    adj = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[idx]))
        adj[idx] = running
    out["p_holm"] = adj
    return out


def _noninferiority(metrics: pd.DataFrame, baseline: str = "iforest_sbi", margin: float = -0.03) -> pd.DataFrame:
    rows = []
    for method in ["cascade_preserve_sbi", "cascade_equal_fpr"]:
        for scenario in STANDARD_SBI_ATTACKS:
            for severity in sorted(metrics["severity"].dropna().unique()):
                sub = metrics[(metrics["scenario"] == scenario) & (metrics["severity"] == severity)]
                a = sub[sub["method"] == method].set_index("seed")["recall"]
                b = sub[sub["method"] == baseline].set_index("seed")["recall"]
                common = a.index.intersection(b.index)
                if len(common) < 2:
                    continue
                d = (a.loc[common] - b.loc[common]).to_numpy(float)
                n = len(d)
                mean = float(d.mean())
                se = float(d.std(ddof=1) / np.sqrt(n))
                crit = float(student_t.ppf(0.975, n - 1))
                low = mean - crit * se
                high = mean + crit * se
                rows.append({
                    "method": method,
                    "scenario": scenario,
                    "severity": float(severity),
                    "n_pairs": n,
                    "margin": margin,
                    "mean_recall_diff": mean,
                    "ci95_low": low,
                    "ci95_high": high,
                    "noninferior_by_ci": bool(low > margin),
                })
    return pd.DataFrame(rows)


def run(cfg: ExperimentConfig, output_dir: Path) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_rows: List[dict] = []
    attr_rows: List[dict] = []

    for seed in cfg.seeds:
        train = _featurize(seed + 1, cfg, None, cfg.train_windows)
        cal = _featurize(seed + 2, cfg, None, cfg.calibration_windows)

        sbi = fit_iforest(
            train, cal, FEATURE_SETS["SBI"],
            target_fpr=cfg.contamination_target_fpr,
            seed=seed,
        )
        legacy_full = fit_iforest(
            train, cal, FEATURE_SETS["FULL"],
            target_fpr=cfg.contamination_target_fpr,
            seed=seed + 19,
        )
        fusion_maxq = fit_decision_fusion_family(
            train, cal,
            target_fpr=cfg.contamination_target_fpr,
            seed=seed + 31,
        )["fusion_maxq"]
        cascades = fit_guarded_cascade_family(
            train, cal,
            seed=seed,
            target_fpr=cfg.contamination_target_fpr,
        )

        methods = {
            "iforest_sbi": sbi,
            "iforest_full_legacy": legacy_full,
            "fusion_maxq": fusion_maxq,
            **cascades,
        }

        frames = [("normal", np.nan, _featurize(seed + 4, cfg, None, cfg.eval_windows))]
        for severity in cfg.severities:
            for i, scenario in enumerate(ATTACKS):
                frames.append((
                    scenario,
                    severity,
                    _featurize(
                        seed + 70_000 + i * 1_000 + int(severity * 100),
                        cfg,
                        AttackSpec(scenario, severity),
                        cfg.eval_windows,
                    ),
                ))

        for scenario, severity, frame in frames:
            for method_name, model in methods.items():
                score = model.score(frame)
                pred = model.predict(frame)
                m = binary_metrics(frame, score, pred)
                m["latency_windows"] = detection_latency_windows(frame, pred)
                m.update(
                    seed=int(seed),
                    method=method_name,
                    scenario=scenario,
                    severity=float(severity) if not np.isnan(severity) else np.nan,
                )
                metric_rows.append(m)

                if method_name.startswith("cascade_") and scenario != "normal":
                    y = frame["label"].to_numpy(int).astype(bool)
                    stages = model.stage_predictions(frame)
                    denom = max(int(y.sum()), 1)
                    sbi_hit = y & stages["sbi"]
                    active_added = y & (~stages["sbi"]) & stages["active"]
                    context_added = y & (~stages["sbi"]) & (~stages["active"]) & stages["context"]
                    attr_rows.append({
                        "seed": int(seed),
                        "method": method_name,
                        "scenario": scenario,
                        "severity": float(severity),
                        "positive_count": int(y.sum()),
                        "sbi_detected_fraction": float(sbi_hit.sum() / denom),
                        "active_added_fraction": float(active_added.sum() / denom),
                        "context_added_fraction": float(context_added.sum() / denom),
                        "total_recall_from_stages": float((sbi_hit | active_added | context_added).sum() / denom),
                    })

    metrics = pd.DataFrame(metric_rows)
    summary = _aggregate(metrics)
    paired = _paired_stats(metrics)
    noninf = _noninferiority(metrics)
    attribution = pd.DataFrame(attr_rows)
    attr_summary = (
        attribution.groupby(["method", "scenario", "severity"])[
            ["sbi_detected_fraction", "active_added_fraction", "context_added_fraction", "total_recall_from_stages"]
        ]
        .agg(["mean", "std"])
        .reset_index()
    )
    attr_summary.columns = [
        "_".join([c for c in tup if c]) if isinstance(tup, tuple) else tup
        for tup in attr_summary.columns
    ]

    paths = {
        "metrics": output_dir / "metrics_per_seed.csv",
        "summary": output_dir / "summary.csv",
        "paired": output_dir / "paired_stats_vs_sbi.csv",
        "noninferiority": output_dir / "noninferiority_standard_attacks.csv",
        "attribution": output_dir / "stage_attribution_per_seed.csv",
        "attribution_summary": output_dir / "stage_attribution_summary.csv",
    }
    metrics.to_csv(paths["metrics"], index=False)
    summary.to_csv(paths["summary"], index=False)
    paired.to_csv(paths["paired"], index=False)
    noninf.to_csv(paths["noninferiority"], index=False)
    attribution.to_csv(paths["attribution"], index=False)
    attr_summary.to_csv(paths["attribution_summary"], index=False)

    medium = summary[np.isclose(summary["severity"].fillna(-1), 1.0)].copy()
    focus = medium[medium["scenario"].isin(ATTACKS)][
        ["method", "scenario", "recall_mean", "recall_std", "f1_mean", "fpr_mean", "pr_auc_mean", "latency_windows_mean"]
    ]
    focus_path = output_dir / "focus_medium.json"
    focus_path.write_text(
        json.dumps(focus.to_dict("records"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    paths["focus"] = focus_path
    return paths


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="5G KIE guarded-cascade experiment")
    p.add_argument("--output", type=Path, default=Path(__file__).parent / "results_cascade")
    p.add_argument("--seeds", type=int, default=30)
    p.add_argument("--quick", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    n = 3 if args.quick else max(2, args.seeds)
    cfg = ExperimentConfig(seeds=_seed_sequence(n))
    if args.quick:
        cfg = ExperimentConfig(
            train_windows=120,
            calibration_windows=70,
            warmup_windows=40,
            eval_windows=40,
            supervised_attack_windows=20,
            seeds=_seed_sequence(n),
            severities=(0.55, 1.0),
            random_forest_trees=50,
        )
    paths = run(cfg, args.output)
    print("Guarded-cascade experiment complete")
    print(f"seeds: {len(cfg.seeds)}")
    for k, p in paths.items():
        print(f"{k}: {p}")


if __name__ == "__main__":
    main()
