from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import rankdata, t as student_t, wilcoxon

from config import ExperimentConfig
from decision_fusion import fit_decision_fusion_family
from features import TemporalState, add_features
from models import FEATURE_SETS, binary_metrics, detection_latency_windows, fit_iforest
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

FOCUS_ATTACKS = [
    "compromised_lie",
    "slow_drift_lie",
    "compromised_truth",
    "rapid_reset",
    "oauth_abuse",
    "nrf_poisoning",
    "coordinated_majority",
]


def _featurize_independent(
    seed: int,
    cfg: ExperimentConfig,
    attack: AttackSpec | None,
    n_windows: int,
) -> pd.DataFrame:
    sim = FiveGCoreSimulator(cfg.fleet, seed=seed)
    warm = sim.generate(cfg.warmup_windows)
    body = sim.generate(n_windows, attack=attack)
    state = TemporalState()
    _ = add_features(warm, temporal=state)
    return add_features(body, temporal=state)


def _aggregate(metrics: pd.DataFrame) -> pd.DataFrame:
    cols = ["precision", "recall", "f1", "fpr", "pr_auc", "latency_windows"]
    grouped = (
        metrics.groupby(["method", "scenario", "severity"], dropna=False)[cols]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    grouped.columns = [
        "_".join([c for c in tup if c]) if isinstance(tup, tuple) else tup
        for tup in grouped.columns
    ]
    return grouped


def _rank_biserial(diff: np.ndarray) -> float:
    d = np.asarray(diff, dtype=float)
    d = d[~np.isclose(d, 0.0)]
    if d.size == 0:
        return 0.0
    ranks = rankdata(np.abs(d), method="average")
    w_plus = float(ranks[d > 0].sum())
    w_minus = float(ranks[d < 0].sum())
    denom = w_plus + w_minus
    return 0.0 if denom == 0 else (w_plus - w_minus) / denom


def _paired_statistics(metrics: pd.DataFrame, baseline: str = "iforest_sbi") -> pd.DataFrame:
    rows: List[dict] = []
    attacks = metrics[metrics["scenario"] != "normal"].copy()
    methods = [m for m in sorted(attacks["method"].unique()) if m != baseline]

    for method in methods:
        for scenario in sorted(attacks["scenario"].unique()):
            for severity in sorted(attacks["severity"].dropna().unique()):
                sub = attacks[
                    (attacks["scenario"] == scenario)
                    & (attacks["severity"] == severity)
                ]
                a = (
                    sub[sub["method"] == method]
                    .sort_values("seed")
                    .set_index("seed")["recall"]
                )
                b = (
                    sub[sub["method"] == baseline]
                    .sort_values("seed")
                    .set_index("seed")["recall"]
                )
                common = a.index.intersection(b.index)
                if len(common) < 2:
                    continue
                diff = (a.loc[common] - b.loc[common]).to_numpy(float)
                if np.allclose(diff, 0):
                    stat, p = 0.0, 1.0
                else:
                    stat, p = wilcoxon(diff, zero_method="wilcox", alternative="two-sided")

                n = len(diff)
                mean_gain = float(np.mean(diff))
                if n > 1:
                    se = float(np.std(diff, ddof=1) / np.sqrt(n))
                    crit = float(student_t.ppf(0.975, df=n - 1))
                    ci_lo = mean_gain - crit * se
                    ci_hi = mean_gain + crit * se
                else:
                    ci_lo = ci_hi = mean_gain

                rows.append(
                    dict(
                        method=method,
                        baseline=baseline,
                        scenario=scenario,
                        severity=float(severity),
                        n_pairs=int(n),
                        mean_recall_gain=mean_gain,
                        median_recall_gain=float(np.median(diff)),
                        gain_ci95_low=float(ci_lo),
                        gain_ci95_high=float(ci_hi),
                        rank_biserial=float(_rank_biserial(diff)),
                        wilcoxon_stat=float(stat),
                        p_value=float(p),
                    )
                )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    # Holm correction over the complete planned family of paired tests.
    order = np.argsort(out["p_value"].to_numpy(float))
    p = out["p_value"].to_numpy(float)
    adjusted = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, idx in enumerate(order):
        val = min(1.0, (m - rank) * p[idx])
        running = max(running, val)
        adjusted[idx] = running
    out["p_holm"] = adjusted
    return out


def _seed_sequence(n: int) -> tuple[int, ...]:
    # Deterministic, spaced seeds; first five include the original experiment's
    # rough scale but are intentionally not limited to those five.
    return tuple(101 + 101 * i for i in range(n))


def run(cfg: ExperimentConfig, output_dir: Path) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: List[dict] = []

    for seed in cfg.seeds:
        train_benign = _featurize_independent(seed + 1, cfg, None, cfg.train_windows)
        cal_benign = _featurize_independent(seed + 2, cfg, None, cfg.calibration_windows)

        legacy_sbi = fit_iforest(
            train_benign,
            cal_benign,
            features=FEATURE_SETS["SBI"],
            target_fpr=cfg.contamination_target_fpr,
            seed=seed,
        )
        legacy_full = fit_iforest(
            train_benign,
            cal_benign,
            features=FEATURE_SETS["FULL"],
            target_fpr=cfg.contamination_target_fpr,
            seed=seed + 17,
        )
        fusion_models = fit_decision_fusion_family(
            train_benign,
            cal_benign,
            target_fpr=cfg.contamination_target_fpr,
            seed=seed + 31,
        )

        methods = {
            "iforest_sbi": legacy_sbi,
            "iforest_full_legacy": legacy_full,
            **fusion_models,
        }

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
                all_rows.append(m)

    metrics = pd.DataFrame(all_rows)
    summary = _aggregate(metrics)
    stats = _paired_statistics(metrics)

    metrics_path = output_dir / "metrics_per_seed.csv"
    summary_path = output_dir / "summary.csv"
    stats_path = output_dir / "paired_stats_vs_sbi.csv"
    metrics.to_csv(metrics_path, index=False)
    summary.to_csv(summary_path, index=False)
    stats.to_csv(stats_path, index=False)

    focus = metrics[
        metrics["scenario"].isin(FOCUS_ATTACKS)
        & np.isclose(metrics["severity"].fillna(-1.0), 1.0)
    ].copy()
    compact = (
        focus.groupby(["method", "scenario"])[
            ["recall", "precision", "f1", "fpr", "pr_auc", "latency_windows"]
        ]
        .agg(["mean", "std"])
        .reset_index()
    )
    compact.columns = [
        "_".join([c for c in tup if c]) if isinstance(tup, tuple) else tup
        for tup in compact.columns
    ]
    json_path = output_dir / "focus_medium.json"
    json_path.write_text(
        json.dumps(compact.to_dict("records"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return {
        "metrics": metrics_path,
        "summary": summary_path,
        "statistics": stats_path,
        "focus": json_path,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="5G Core decision-level fusion experiment")
    p.add_argument("--output", type=Path, default=Path(__file__).parent / "results_fusion")
    p.add_argument("--seeds", type=int, default=20, help="number of independent seeds")
    p.add_argument("--quick", action="store_true", help="3-seed smoke run")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    n_seeds = 3 if args.quick else max(2, int(args.seeds))
    cfg = ExperimentConfig(seeds=_seed_sequence(n_seeds))

    if args.quick:
        cfg = ExperimentConfig(
            train_windows=120,
            calibration_windows=70,
            warmup_windows=40,
            eval_windows=40,
            supervised_attack_windows=20,
            seeds=_seed_sequence(n_seeds),
            severities=(0.55, 1.0),
            random_forest_trees=50,
        )

    paths = run(cfg, args.output)
    print("Decision-fusion experiment complete")
    print(f"seeds: {len(cfg.seeds)}")
    for k, v in paths.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
