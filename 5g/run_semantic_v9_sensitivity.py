"""Exploratory sensitivity grid for SBA semantic consistency v9.

This runner does not change the v8 detector and does not make confirmatory
claims. It probes whether the operational-grace trade-off survives degraded
marker/recovery telemetry. Results are exploratory by design and are kept out
of the v9 primary Holm family.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from run_semantic_v7 import bootstrap_delta_ci
from run_semantic_v8 import ARM_NOISY, ARM_STATIC, run_seed, setup_logging
from sba_semantic_v8 import OperationalContextConfig

LOG = logging.getLogger("sba-semantic-v9-sensitivity")
DEFAULT_SENSITIVITY_SEEDS = (42, 47, 51)


@dataclass(frozen=True)
class SensitivityPoint:
    point_id: str
    marker_recall: float = 0.85
    recovery_probability: float = 0.90
    recovery_latency_multiplier: float = 1.0
    correlation_error_probability: float = 0.03
    false_marker_probability: float = 0.02
    marker_latency_median_s: float = 0.12
    disturbance_manifest_probability: float = 0.95

    def config(self) -> OperationalContextConfig:
        cfg = OperationalContextConfig(
            marker_recall=self.marker_recall,
            false_marker_probability=self.false_marker_probability,
            correlation_error_probability=self.correlation_error_probability,
            marker_latency_median_s=self.marker_latency_median_s,
            recovery_observation_probability=self.recovery_probability,
            recovery_latency_multiplier=self.recovery_latency_multiplier,
            disturbance_manifest_probability=self.disturbance_manifest_probability,
        )
        cfg.validate()
        return cfg


SENSITIVITY_GRID = (
    SensitivityPoint("baseline"),
    SensitivityPoint("marker_060", marker_recall=0.60),
    SensitivityPoint("marker_075", marker_recall=0.75),
    SensitivityPoint("marker_095", marker_recall=0.95),
    SensitivityPoint("recovery_060", recovery_probability=0.60),
    SensitivityPoint("recovery_075", recovery_probability=0.75),
    SensitivityPoint("recovery_slow_2x", recovery_latency_multiplier=2.0),
    SensitivityPoint("recovery_slow_4x", recovery_latency_multiplier=4.0),
    SensitivityPoint("corr_error_010", correlation_error_probability=0.10),
    SensitivityPoint(
        "degraded_combo",
        marker_recall=0.60,
        recovery_probability=0.60,
        recovery_latency_multiplier=2.0,
        correlation_error_probability=0.10,
    ),
)

KEY_METRICS = (
    "benign_fpr",
    "mean_attack_window_recall",
    "mean_operational_fpr",
    "mean_semantic_operational_fpr",
    "state_attack_window_recall",
    "state_attack_event_latency_s",
)


def stable_offset(*parts: str) -> int:
    raw = "|".join(parts).encode("utf-8")
    return int(hashlib.sha256(raw).hexdigest()[:8], 16) % 1_000_000


def paired_seed_rows(point: SensitivityPoint, summaries: pd.DataFrame) -> pd.DataFrame:
    static = summaries[summaries["arm"] == ARM_STATIC].set_index("seed")
    noisy = summaries[summaries["arm"] == ARM_NOISY].set_index("seed")
    seeds = sorted(set(static.index) & set(noisy.index))
    rows: List[Dict] = []
    for seed in seeds:
        row: Dict = {"point_id": point.point_id, "seed": int(seed), **asdict(point)}
        for metric in KEY_METRICS:
            if metric not in summaries.columns:
                continue
            s = float(static.loc[seed, metric])
            n = float(noisy.loc[seed, metric])
            row[f"static_{metric}"] = s
            row[f"v9_{metric}"] = n
            row[f"delta_{metric}"] = n - s
        base_fpr = row.get("static_mean_operational_fpr", float("nan"))
        cand_fpr = row.get("v9_mean_operational_fpr", float("nan"))
        row["operational_fpr_relative_reduction"] = (
            (base_fpr - cand_fpr) / base_fpr
            if np.isfinite(base_fpr) and base_fpr > 0 and np.isfinite(cand_fpr)
            else float("nan")
        )
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_point(seed_rows: pd.DataFrame, bootstrap: int, seed: int) -> Dict:
    point_id = str(seed_rows["point_id"].iloc[0])
    first = seed_rows.iloc[0]
    out: Dict = {
        "point_id": point_id,
        "n_seeds": int(seed_rows["seed"].nunique()),
        "marker_recall": float(first["marker_recall"]),
        "recovery_probability": float(first["recovery_probability"]),
        "recovery_latency_multiplier": float(first["recovery_latency_multiplier"]),
        "correlation_error_probability": float(first["correlation_error_probability"]),
        "false_marker_probability": float(first["false_marker_probability"]),
    }
    for metric in KEY_METRICS:
        delta_col = f"delta_{metric}"
        if delta_col not in seed_rows:
            continue
        x = seed_rows[delta_col].to_numpy(dtype=float)
        boot_seed = int(seed) + stable_offset(point_id, metric)
        lo, hi = bootstrap_delta_ci(x, boot_seed, bootstrap)
        out[f"delta_{metric}_mean"] = float(np.nanmean(x))
        out[f"delta_{metric}_ci_lo"] = float(lo)
        out[f"delta_{metric}_ci_hi"] = float(hi)
        out[f"static_{metric}_mean"] = float(np.nanmean(seed_rows[f"static_{metric}"].to_numpy(dtype=float)))
        out[f"v9_{metric}_mean"] = float(np.nanmean(seed_rows[f"v9_{metric}"].to_numpy(dtype=float)))
    out["operational_fpr_relative_reduction_mean"] = float(
        np.nanmean(seed_rows["operational_fpr_relative_reduction"].to_numpy(dtype=float))
    )
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="SBA v9 exploratory operational-context sensitivity grid")
    p.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SENSITIVITY_SEEDS))
    p.add_argument("--points", nargs="+", default=None, help="Optional point IDs; default runs the full fixed grid")
    p.add_argument("--calib-windows", type=int, default=600)
    p.add_argument("--eval-windows", type=int, default=80)
    p.add_argument("--operational-windows", type=int, default=100)
    p.add_argument("--state-attack-windows", type=int, default=60)
    p.add_argument("--hidden", type=int, default=5)
    p.add_argument("--target-fpr", type=float, default=0.01)
    p.add_argument("--bootstrap", type=int, default=3000)
    p.add_argument("--log-every", type=int, default=0)
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--out-dir", default="runs/sba_semantic_v9/sensitivity")
    args = p.parse_args(argv)
    setup_logging(args.quiet)

    by_id = {p.point_id: p for p in SENSITIVITY_GRID}
    requested = args.points or list(by_id)
    unknown = [x for x in requested if x not in by_id]
    if unknown:
        raise ValueError(f"unknown sensitivity point(s): {unknown}")
    points = [by_id[x] for x in requested]
    seeds = [int(x) for x in args.seeds]
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be unique")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    plan = {
        "mode": "exploratory_sensitivity",
        "confirmatory": False,
        "detector_core": {
            "semantic_profile": "q3of5",
            "soft_persistence": False,
            "transport_mode": "byz_3of4",
            "implementation": "sba_semantic_v8",
        },
        "seeds": seeds,
        "grid": [asdict(p) for p in points],
        "note": "Sensitivity results are exploratory and are not included in the confirmatory Holm family.",
    }
    (out / "sensitivity_plan.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    all_seed_rows = []
    point_summaries = []
    started = time.perf_counter()
    for pi, point in enumerate(points, 1):
        cfg = point.config()
        LOG.info("=" * 100)
        LOG.info(
            "SENSITIVITY %d/%d %s | marker=%.2f recovery=%.2f mult=%.2f corrErr=%.2f",
            pi, len(points), point.point_id, point.marker_recall,
            point.recovery_probability, point.recovery_latency_multiplier,
            point.correlation_error_probability,
        )
        results = []
        for si, seed in enumerate(seeds, 1):
            LOG.info("  point=%s seed %d/%d -> %d", point.point_id, si, len(seeds), seed)
            results.append(run_seed(
                seed=seed,
                calib_windows=args.calib_windows,
                eval_windows=args.eval_windows,
                operational_windows=args.operational_windows,
                state_attack_windows=args.state_attack_windows,
                hidden=args.hidden,
                target_fpr=args.target_fpr,
                cfg=cfg,
                out_dir=out / point.point_id / f"seed_{seed}",
                log_every=args.log_every,
            ))
        summaries = pd.concat([r["summary"] for r in results], ignore_index=True)
        summaries.to_csv(out / point.point_id / "all_seed_summary.csv", index=False)
        paired = paired_seed_rows(point, summaries)
        paired.to_csv(out / point.point_id / "paired_seed_metrics.csv", index=False)
        all_seed_rows.append(paired)
        point_summaries.append(aggregate_point(paired, args.bootstrap, min(seeds)))

    seed_df = pd.concat(all_seed_rows, ignore_index=True) if all_seed_rows else pd.DataFrame()
    summary_df = pd.DataFrame(point_summaries)
    seed_df.to_csv(out / "sensitivity_seed_metrics.csv", index=False)
    summary_df.to_csv(out / "sensitivity_summary.csv", index=False)

    runtime = time.perf_counter() - started
    summary = {
        "mode": "exploratory_sensitivity",
        "points": requested,
        "seeds": seeds,
        "runtime_seconds": runtime,
        "summary_rows": point_summaries,
    }
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    LOG.info("=" * 100)
    LOG.info("FINAL V9 SENSITIVITY SUMMARY")
    for row in point_summaries:
        LOG.info(
            "%-20s opFPR static/v9=%.4f/%.4f delta=%+.4f relRed=%.1f%% stateR delta=%+.4f latency delta=%+.3fs",
            row["point_id"],
            row.get("static_mean_operational_fpr_mean", float("nan")),
            row.get("v9_mean_operational_fpr_mean", float("nan")),
            row.get("delta_mean_operational_fpr_mean", float("nan")),
            100.0 * row.get("operational_fpr_relative_reduction_mean", float("nan")),
            row.get("delta_state_attack_window_recall_mean", float("nan")),
            row.get("delta_state_attack_event_latency_s_mean", float("nan")),
        )
    LOG.info("Runtime %.1fs | output=%s", runtime, out)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
