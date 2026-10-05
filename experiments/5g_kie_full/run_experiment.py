from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from config import ExperimentConfig
from features import TemporalState, add_features
from models import (
    FEATURE_SETS,
    binary_metrics,
    detection_latency_windows,
    fit_iforest,
    fit_random_forest,
)
from simulator import AttackSpec, FiveGCoreSimulator


ATTACKS = [
    "compromised_lie",
    "compromised_truth",
    "slow_drift_lie",
    "fake_nf",
    "rapid_reset",
    "oauth_abuse",
    "nrf_poisoning",
    "cross_slice",
    "coordinated_majority",
]


def _featurize_independent(seed: int, cfg: ExperimentConfig, attack: AttackSpec | None, n_windows: int) -> pd.DataFrame:
    sim = FiveGCoreSimulator(cfg.fleet, seed=seed)
    warm = sim.generate(cfg.warmup_windows)
    body = sim.generate(n_windows, attack=attack)
    state = TemporalState()
    _ = add_features(warm, temporal=state)
    return add_features(body, temporal=state)


def _build_supervised_train(seed: int, cfg: ExperimentConfig) -> pd.DataFrame:
    parts: List[pd.DataFrame] = []
    benign = _featurize_independent(seed + 10_000, cfg, None, cfg.supervised_attack_windows)
    parts.append(benign)
    for idx, name in enumerate(ATTACKS):
        for severity in (0.65, 1.15):
            part = _featurize_independent(
                seed + 20_000 + idx * 100 + int(severity * 10),
                cfg,
                AttackSpec(name, severity),
                cfg.supervised_attack_windows,
            )
            parts.append(part)
    return pd.concat(parts, ignore_index=True)


def _aggregate(metrics: pd.DataFrame) -> pd.DataFrame:
    cols = ["precision", "recall", "f1", "fpr", "pr_auc", "latency_windows"]
    grouped = (
        metrics.groupby(["model", "feature_set", "scenario", "severity"], dropna=False)[cols]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    grouped.columns = [
        "_".join([c for c in tup if c]) if isinstance(tup, tuple) else tup
        for tup in grouped.columns
    ]
    return grouped


def _wilcoxon_table(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    attack_rows = metrics[metrics["scenario"] != "normal"]
    for model in sorted(attack_rows["model"].unique()):
        for scenario in sorted(attack_rows["scenario"].unique()):
            for severity in sorted(attack_rows["severity"].dropna().unique()):
                sub = attack_rows[
                    (attack_rows["model"] == model)
                    & (attack_rows["scenario"] == scenario)
                    & (attack_rows["severity"] == severity)
                ]
                a = sub[sub["feature_set"] == "FULL"].sort_values("seed")["recall"].to_numpy(float)
                b = sub[sub["feature_set"] == "SBI"].sort_values("seed")["recall"].to_numpy(float)
                if len(a) != len(b) or len(a) < 2:
                    continue
                diff = a - b
                if np.allclose(diff, 0):
                    stat, p = 0.0, 1.0
                else:
                    stat, p = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
                rows.append(
                    dict(
                        model=model,
                        scenario=scenario,
                        severity=float(severity),
                        mean_recall_gain=float(np.mean(diff)),
                        median_recall_gain=float(np.median(diff)),
                        wilcoxon_stat=float(stat),
                        p_value=float(p),
                    )
                )
    return pd.DataFrame(rows)


def run(cfg: ExperimentConfig, output_dir: Path) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: List[dict] = []

    for seed in cfg.seeds:
        train_benign = _featurize_independent(seed + 1, cfg, None, cfg.train_windows)
        cal_benign = _featurize_independent(seed + 2, cfg, None, cfg.calibration_windows)
        sup_train = _build_supervised_train(seed + 3, cfg)

        if_models = {}
        rf_models = {}
        for feature_name, features in FEATURE_SETS.items():
            if_models[feature_name] = fit_iforest(
                train_benign,
                cal_benign,
                features=features,
                target_fpr=cfg.contamination_target_fpr,
                seed=seed,
            )
            rf_models[feature_name] = fit_random_forest(
                sup_train,
                features=features,
                seed=seed,
                n_estimators=cfg.random_forest_trees,
                n_jobs=cfg.n_jobs,
            )

        normal = _featurize_independent(seed + 4, cfg, None, cfg.eval_windows)
        scenario_frames = [("normal", np.nan, normal)]
        for severity in cfg.severities:
            for i, scenario in enumerate(ATTACKS):
                frame = _featurize_independent(
                    seed + 50_000 + i * 1_000 + int(severity * 100),
                    cfg,
                    AttackSpec(scenario, severity),
                    cfg.eval_windows,
                )
                scenario_frames.append((scenario, severity, frame))

        for scenario, severity, frame in scenario_frames:
            for feature_name in FEATURE_SETS:
                for model_name, model in (
                    ("iforest", if_models[feature_name]),
                    ("random_forest", rf_models[feature_name]),
                ):
                    score = model.score(frame)
                    pred = model.predict(frame)
                    m = binary_metrics(frame, score, pred)
                    m["latency_windows"] = detection_latency_windows(frame, pred)
                    m.update(
                        seed=int(seed),
                        model=model_name,
                        feature_set=feature_name,
                        scenario=scenario,
                        severity=float(severity) if not np.isnan(severity) else np.nan,
                    )
                    all_rows.append(m)

    metrics = pd.DataFrame(all_rows)
    summary = _aggregate(metrics)
    stats = _wilcoxon_table(metrics)

    metrics_path = output_dir / "metrics_per_seed.csv"
    summary_path = output_dir / "summary.csv"
    stats_path = output_dir / "wilcoxon_full_vs_sbi.csv"
    metrics.to_csv(metrics_path, index=False)
    summary.to_csv(summary_path, index=False)
    stats.to_csv(stats_path, index=False)

    focus = metrics[
        (metrics["feature_set"].isin(["SBI", "FULL"]))
        & (metrics["model"] == "iforest")
    ].copy()
    compact = (
        focus.groupby(["feature_set", "scenario", "severity"], dropna=False)[
            ["recall", "fpr", "pr_auc", "latency_windows"]
        ]
        .mean()
        .reset_index()
        .to_dict("records")
    )
    json_path = output_dir / "focus_iforest.json"
    json_path.write_text(json.dumps(compact, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "metrics": metrics_path,
        "summary": summary_path,
        "statistics": stats_path,
        "focus": json_path,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="5G Core active-control / SBI anomaly experiment")
    p.add_argument("--output", type=Path, default=Path(__file__).parent / "results")
    p.add_argument("--quick", action="store_true", help="faster smoke/CI run")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ExperimentConfig()
    if args.quick:
        cfg = ExperimentConfig(
            train_windows=120,
            calibration_windows=60,
            warmup_windows=40,
            eval_windows=35,
            supervised_attack_windows=22,
            seeds=(101, 202),
            severities=(0.55, 1.0),
            random_forest_trees=100,
        )
    paths = run(cfg, args.output)
    print("Experiment complete")
    for k, v in paths.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
