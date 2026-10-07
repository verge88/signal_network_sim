from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import rankdata, t as student_t, wilcoxon

from cascade_detector import fit_preserve_sbi_cascade_sweep
from config import ExperimentConfig
from decision_fusion import fit_decision_fusion_family
from features import TemporalState, add_features
from models import FEATURE_SETS, binary_metrics, detection_latency_windows, fit_iforest
from simulator import AttackSpec, FiveGCoreSimulator


REPORT_ALPHAS = (0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)
ACTIVE_FPR_BUDGETS = (0.0001, 0.00025, 0.0005, 0.001, 0.002, 0.005)
CONTEXT_FPR_BUDGET = 0.0005
PRACTICAL_GAIN = 0.05


def _seed_sequence(n: int) -> tuple[int, ...]:
    return tuple(101 + 101 * i for i in range(n))


def _featurize(
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


def _rank_biserial(diff: np.ndarray) -> float:
    d = np.asarray(diff, dtype=float)
    d = d[~np.isclose(d, 0.0)]
    if d.size == 0:
        return 0.0
    ranks = rankdata(np.abs(d), method="average")
    wp = float(ranks[d > 0].sum())
    wm = float(ranks[d < 0].sum())
    return 0.0 if wp + wm == 0 else (wp - wm) / (wp + wm)


def _aggregate(metrics: pd.DataFrame) -> pd.DataFrame:
    cols = ["precision", "recall", "f1", "fpr", "pr_auc", "latency_windows"]
    g = (
        metrics.groupby(
            ["method", "report_alpha", "active_fpr_budget", "scenario", "severity"],
            dropna=False,
        )[cols]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    g.columns = [
        "_".join([c for c in tup if c]) if isinstance(tup, tuple) else tup
        for tup in g.columns
    ]
    return g


def _paired_stats(metrics: pd.DataFrame) -> pd.DataFrame:
    attack = metrics[metrics["scenario"] == "adaptive_lie"].copy()
    baseline = attack[attack["method"] == "iforest_sbi"].copy()
    rows: List[dict] = []

    methods = sorted(
        m for m in attack["method"].unique()
        if m != "iforest_sbi"
    )
    for method in methods:
        mframe = attack[attack["method"] == method]
        for severity in sorted(attack["severity"].dropna().unique()):
            for alpha in REPORT_ALPHAS:
                a = mframe[
                    np.isclose(mframe["severity"], severity)
                    & np.isclose(mframe["report_alpha"], alpha)
                ].set_index("seed")["recall"]
                b = baseline[
                    np.isclose(baseline["severity"], severity)
                    & np.isclose(baseline["report_alpha"], alpha)
                ].set_index("seed")["recall"]
                common = a.index.intersection(b.index)
                if len(common) < 2:
                    continue
                diff = (a.loc[common] - b.loc[common]).to_numpy(float)
                if np.allclose(diff, 0.0):
                    stat, p = 0.0, 1.0
                else:
                    stat, p = wilcoxon(
                        diff,
                        zero_method="wilcox",
                        alternative="two-sided",
                    )
                n = len(diff)
                mean = float(np.mean(diff))
                sd = float(np.std(diff, ddof=1)) if n > 1 else 0.0
                se = sd / np.sqrt(n) if n > 1 else 0.0
                crit = float(student_t.ppf(0.975, n - 1)) if n > 1 else 0.0
                budget = (
                    float(mframe["active_fpr_budget"].dropna().iloc[0])
                    if mframe["active_fpr_budget"].notna().any()
                    else np.nan
                )
                rows.append({
                    "method": method,
                    "severity": float(severity),
                    "report_alpha": float(alpha),
                    "active_fpr_budget": budget,
                    "n_pairs": int(n),
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
    out["statistically_positive"] = (
        (out["p_holm"] < 0.05)
        & (out["gain_ci95_low"] > 0.0)
        & (out["mean_recall_gain"] > 0.0)
    )
    out["practically_useful"] = (
        out["statistically_positive"]
        & (out["mean_recall_gain"] >= PRACTICAL_GAIN)
    )
    return out


def _critical_alpha(stats: pd.DataFrame) -> pd.DataFrame:
    rows = []
    cascade = stats[stats["method"].str.startswith("cascade_active_")].copy()
    for (method, severity), sub in cascade.groupby(["method", "severity"]):
        useful = sub[sub["practically_useful"]].sort_values("report_alpha")
        if useful.empty:
            critical = np.nan
            gain = np.nan
            p_holm = np.nan
        else:
            last = useful.iloc[-1]
            critical = float(last["report_alpha"])
            gain = float(last["mean_recall_gain"])
            p_holm = float(last["p_holm"])
        budget = float(sub["active_fpr_budget"].dropna().iloc[0])
        rows.append({
            "method": method,
            "severity": float(severity),
            "active_fpr_budget": budget,
            "practical_gain_threshold": PRACTICAL_GAIN,
            "critical_report_alpha": critical,
            "gain_at_critical_alpha": gain,
            "p_holm_at_critical_alpha": p_holm,
        })
    return pd.DataFrame(rows)


def _pareto_table(metrics: pd.DataFrame) -> pd.DataFrame:
    casc = metrics[metrics["method"].str.startswith("cascade_active_")].copy()
    normal = casc[casc["scenario"] == "normal"].groupby(
        ["method", "active_fpr_budget"]
    )["fpr"].mean().rename("normal_fpr_mean").reset_index()

    attack = casc[casc["scenario"] == "adaptive_lie"].groupby(
        ["method", "active_fpr_budget", "severity", "report_alpha"]
    )[["recall", "f1", "pr_auc", "latency_windows"]].mean().reset_index()

    out = attack.merge(normal, on=["method", "active_fpr_budget"], how="left")
    out["dominated"] = False
    for (severity, alpha), idx in out.groupby(["severity", "report_alpha"]).groups.items():
        locs = list(idx)
        for i in locs:
            fi = float(out.loc[i, "normal_fpr_mean"])
            ri = float(out.loc[i, "recall"])
            for j in locs:
                if i == j:
                    continue
                fj = float(out.loc[j, "normal_fpr_mean"])
                rj = float(out.loc[j, "recall"])
                if (fj <= fi and rj >= ri) and (fj < fi or rj > ri):
                    out.loc[i, "dominated"] = True
                    break
    return out


def run(cfg: ExperimentConfig, output_dir: Path) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: List[dict] = []
    attr_rows: List[dict] = []

    for seed in cfg.seeds:
        train = _featurize(seed + 1, cfg, None, cfg.train_windows)
        cal = _featurize(seed + 2, cfg, None, cfg.calibration_windows)

        sbi = fit_iforest(
            train,
            cal,
            FEATURE_SETS["SBI"],
            target_fpr=cfg.contamination_target_fpr,
            seed=seed,
        )
        legacy_full = fit_iforest(
            train,
            cal,
            FEATURE_SETS["FULL"],
            target_fpr=cfg.contamination_target_fpr,
            seed=seed + 19,
        )
        fusion_maxq = fit_decision_fusion_family(
            train,
            cal,
            target_fpr=cfg.contamination_target_fpr,
            seed=seed + 31,
        )["fusion_maxq"]
        cascades = fit_preserve_sbi_cascade_sweep(
            train,
            cal,
            seed=seed,
            active_fpr_budgets=ACTIVE_FPR_BUDGETS,
            context_fpr_budget=CONTEXT_FPR_BUDGET,
            sbi_target_fpr=cfg.contamination_target_fpr,
        )

        methods: Dict[str, object] = {
            "iforest_sbi": sbi,
            "iforest_full_legacy": legacy_full,
            "fusion_maxq": fusion_maxq,
        }
        for budget, model in cascades.items():
            methods[model.name] = model

        normal = _featurize(seed + 4, cfg, None, cfg.eval_windows)
        for method_name, model in methods.items():
            score = model.score(normal)
            pred = model.predict(normal)
            m = binary_metrics(normal, score, pred)
            m["latency_windows"] = detection_latency_windows(normal, pred)
            m.update({
                "seed": int(seed),
                "method": method_name,
                "scenario": "normal",
                "severity": np.nan,
                "report_alpha": np.nan,
                "active_fpr_budget": (
                    next(
                        (float(b) for b, cm in cascades.items() if cm.name == method_name),
                        np.nan,
                    )
                ),
            })
            rows.append(m)

        for severity in cfg.severities:
            for a_idx, alpha in enumerate(REPORT_ALPHAS):
                frame = _featurize(
                    seed + 80_000 + int(severity * 1000) + a_idx * 131,
                    cfg,
                    AttackSpec(
                        "adaptive_lie",
                        severity=float(severity),
                        report_alpha=float(alpha),
                    ),
                    cfg.eval_windows,
                )

                for method_name, model in methods.items():
                    score = model.score(frame)
                    pred = model.predict(frame)
                    m = binary_metrics(frame, score, pred)
                    m["latency_windows"] = detection_latency_windows(frame, pred)
                    budget = next(
                        (float(b) for b, cm in cascades.items() if cm.name == method_name),
                        np.nan,
                    )
                    m.update({
                        "seed": int(seed),
                        "method": method_name,
                        "scenario": "adaptive_lie",
                        "severity": float(severity),
                        "report_alpha": float(alpha),
                        "active_fpr_budget": budget,
                    })
                    rows.append(m)

                    if method_name.startswith("cascade_active_"):
                        y = frame["label"].to_numpy(int).astype(bool)
                        stages = model.stage_predictions(frame)
                        denom = max(int(y.sum()), 1)
                        sbi_hit = y & stages["sbi"]
                        active_added = y & (~stages["sbi"]) & stages["active"]
                        context_added = (
                            y
                            & (~stages["sbi"])
                            & (~stages["active"])
                            & stages["context"]
                        )
                        attr_rows.append({
                            "seed": int(seed),
                            "method": method_name,
                            "severity": float(severity),
                            "report_alpha": float(alpha),
                            "active_fpr_budget": float(budget),
                            "sbi_detected_fraction": float(sbi_hit.sum() / denom),
                            "active_added_fraction": float(active_added.sum() / denom),
                            "context_added_fraction": float(context_added.sum() / denom),
                            "total_recall": float(
                                (sbi_hit | active_added | context_added).sum() / denom
                            ),
                        })

    metrics = pd.DataFrame(rows)
    summary = _aggregate(metrics)
    paired = _paired_stats(metrics)
    critical = _critical_alpha(paired)
    pareto = _pareto_table(metrics)
    attribution = pd.DataFrame(attr_rows)
    attr_summary = (
        attribution.groupby(
            ["method", "severity", "report_alpha", "active_fpr_budget"]
        )[
            [
                "sbi_detected_fraction",
                "active_added_fraction",
                "context_added_fraction",
                "total_recall",
            ]
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
        "critical": output_dir / "critical_alpha.csv",
        "pareto": output_dir / "pareto.csv",
        "attribution": output_dir / "stage_attribution_summary.csv",
    }
    metrics.to_csv(paths["metrics"], index=False)
    summary.to_csv(paths["summary"], index=False)
    paired.to_csv(paths["paired"], index=False)
    critical.to_csv(paths["critical"], index=False)
    pareto.to_csv(paths["pareto"], index=False)
    attr_summary.to_csv(paths["attribution"], index=False)

    medium = summary[
        (summary["scenario"] == "adaptive_lie")
        & np.isclose(summary["severity"].fillna(-1.0), 1.0)
    ].copy()
    focus = output_dir / "focus_medium.json"
    focus.write_text(
        json.dumps(medium.to_dict("records"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    paths["focus"] = focus
    return paths


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Adaptive attacker / FPR Pareto sweep")
    p.add_argument("--output", type=Path, default=Path(__file__).parent / "results_adaptive_sweep")
    p.add_argument("--seeds", type=int, default=30)
    p.add_argument("--quick", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    n = 3 if args.quick else max(2, int(args.seeds))
    cfg = ExperimentConfig(seeds=_seed_sequence(n))
    if args.quick:
        cfg = ExperimentConfig(
            train_windows=110,
            calibration_windows=70,
            warmup_windows=35,
            eval_windows=35,
            supervised_attack_windows=20,
            seeds=_seed_sequence(n),
            severities=(0.55, 1.0),
            random_forest_trees=50,
        )

    paths = run(cfg, args.output)
    print("Adaptive sweep complete")
    print(f"seeds: {len(cfg.seeds)}")
    print(f"report alphas: {REPORT_ALPHAS}")
    print(f"active FPR budgets: {ACTIVE_FPR_BUDGETS}")
    for k, p in paths.items():
        print(f"{k}: {p}")


if __name__ == "__main__":
    main()
