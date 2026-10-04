"""Confirmatory paired experiment for SBA semantic consistency v9.

V9 does not change the detector.  It freezes the v8 candidate and fixes the
primary inferential family *before* a fresh holdout run.  The previous 42..51
multiseed experiment remains exploratory/pilot evidence; the default holdout
seeds here are 52..61 so the multiplicity family is not chosen after seeing the
same outcomes.

Primary family (Holm corrected only within these four endpoints)
----------------------------------------------------------------
H1  mean_operational_fpr: v8 < static (superiority)
H2  mean_attack_window_recall: v8 non-inferior within 1 percentage point
H3  state_attack_window_recall: v8 non-inferior within 1 percentage point
H4  state_attack_event_latency_s: quantify the expected latency cost

H2/H3 are judged by the paired bootstrap CI lower bound relative to the fixed
non-inferiority margin.  A two-sided paired sign-flip p-value is still reported
for transparency, but zero difference is not evidence against non-inferiority.
H1/H4 use the paired sign-flip test with Holm correction across the full four-
endpoint primary family plus bootstrap CIs.

All per-state, per-family and diagnostic endpoints are secondary/exploratory
and never enter the primary Holm family.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd

from run_semantic_v7 import bootstrap_delta_ci, holm_adjust, paired_signflip_p
from run_semantic_v8 import ARM_NOISY, ARM_STATIC, run_seed, setup_logging
from sba_semantic_v8 import OperationalContextConfig

LOG = logging.getLogger("sba-semantic-v9-confirmatory")
PLAN_VERSION = "v9-confirmatory-1"
DEFAULT_HOLDOUT_SEEDS = tuple(range(52, 62))
DEFAULT_NONINFERIORITY_MARGIN = 0.01


@dataclass(frozen=True)
class PrimaryEndpoint:
    hypothesis: str
    metric: str
    claim: str
    direction: str
    margin: float = 0.0
    rationale: str = ""


PRIMARY_ENDPOINTS = (
    PrimaryEndpoint(
        "H1",
        "mean_operational_fpr",
        "superiority",
        "lower",
        0.0,
        "Noisy operational context should reduce transient operational false alarms.",
    ),
    PrimaryEndpoint(
        "H2",
        "mean_attack_window_recall",
        "noninferiority",
        "not_lower_by_more_than_margin",
        DEFAULT_NONINFERIORITY_MARGIN,
        "Ordinary adaptive-attack Recall must not materially decrease.",
    ),
    PrimaryEndpoint(
        "H3",
        "state_attack_window_recall",
        "noninferiority",
        "not_lower_by_more_than_margin",
        DEFAULT_NONINFERIORITY_MARGIN,
        "Attack Recall during genuine operational states must not materially decrease.",
    ),
    PrimaryEndpoint(
        "H4",
        "state_attack_event_latency_s",
        "cost_characterization",
        "higher",
        0.0,
        "Operational grace is expected to trade false alarms for added detection latency.",
    ),
)

SECONDARY_METRICS = (
    "benign_fpr",
    "mean_attack_event_recall",
    "worst_operational_fpr",
    "mean_semantic_operational_fpr",
    "state_attack_event_recall",
    "state_attack_correct_marker_recall",
    "mean_marker_emission_rate",
    "mean_marker_correct_rate",
    "mean_recovery_before_grace_rate",
)


def canonical_sha256(payload: Dict) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def build_analysis_plan(
    *,
    seeds: Sequence[int],
    cfg: OperationalContextConfig,
    target_fpr: float,
    calib_windows: int,
    eval_windows: int,
    operational_windows: int,
    state_attack_windows: int,
    hidden: int,
    bootstrap: int,
    permutations: int,
    alpha: float,
    recall_margin: float,
) -> Dict:
    endpoints = []
    for endpoint in PRIMARY_ENDPOINTS:
        row = asdict(endpoint)
        if endpoint.claim == "noninferiority":
            row["margin"] = float(recall_margin)
        endpoints.append(row)
    return {
        "plan_version": PLAN_VERSION,
        "status": "fixed_before_simulation",
        "fresh_holdout_expected": True,
        "holdout_seeds": [int(x) for x in seeds],
        "detector_core": {
            "semantic_profile": "q3of5",
            "soft_persistence": False,
            "transport_mode": "byz_3of4",
            "implementation": "sba_semantic_v8",
        },
        "operational_context": dict(cfg.__dict__),
        "experiment": {
            "target_fpr": float(target_fpr),
            "calib_windows": int(calib_windows),
            "eval_windows": int(eval_windows),
            "operational_windows": int(operational_windows),
            "state_attack_windows": int(state_attack_windows),
            "hidden": int(hidden),
            "bootstrap": int(bootstrap),
            "permutations": int(permutations),
        },
        "alpha": float(alpha),
        "multiplicity": {
            "method": "Holm",
            "family": "PRIMARY_ENDPOINTS_ONLY",
            "family_size": len(PRIMARY_ENDPOINTS),
        },
        "primary_endpoints": endpoints,
        "secondary_metrics": list(SECONDARY_METRICS),
        "decision_notes": {
            "H1": "PASS if upper bootstrap CI for delta(v8-static) < 0 and Holm p < alpha.",
            "H2": "PASS non-inferiority if lower bootstrap CI > -margin.",
            "H3": "PASS non-inferiority if lower bootstrap CI > -margin.",
            "H4": "COST CONFIRMED if lower bootstrap CI > 0 and Holm p < alpha; this is not a benefit criterion.",
        },
    }


def write_locked_plan(out: Path, plan: Dict) -> str:
    out.mkdir(parents=True, exist_ok=True)
    plan_path = out / "analysis_plan.json"
    digest = canonical_sha256(plan)
    wrapped = {"analysis_plan_sha256": digest, **plan}
    text = json.dumps(wrapped, ensure_ascii=False, indent=2)
    if plan_path.exists():
        existing = json.loads(plan_path.read_text(encoding="utf-8"))
        existing_digest = existing.get("analysis_plan_sha256")
        if existing_digest != digest:
            raise RuntimeError(
                "analysis_plan.json already exists with a different SHA256; "
                "use a new output directory for a new confirmatory plan"
            )
    else:
        plan_path.write_text(text, encoding="utf-8")
    return digest


def _paired_metric_row(
    summaries: pd.DataFrame,
    metric: str,
    *,
    seed_offset: int,
    n_boot: int,
    n_perm: int,
) -> Dict:
    pivot = summaries.pivot_table(index="seed", columns="arm", values=metric, aggfunc="mean")
    pivot = pivot.dropna(subset=[ARM_STATIC, ARM_NOISY])
    if pivot.empty:
        raise ValueError(f"no paired observations for {metric}")
    static = pivot[ARM_STATIC].to_numpy(dtype=float)
    noisy = pivot[ARM_NOISY].to_numpy(dtype=float)
    delta = noisy - static
    lo, hi = bootstrap_delta_ci(delta, seed_offset, n_boot)
    p = paired_signflip_p(delta, seed_offset + 1, n_perm)
    return {
        "metric": metric,
        "n_seeds": int(len(pivot)),
        "static_mean": float(np.mean(static)),
        "v9_mean": float(np.mean(noisy)),
        "delta_v9_minus_static": float(np.mean(delta)),
        "delta_ci_lo": float(lo),
        "delta_ci_hi": float(hi),
        "paired_signflip_p": float(p),
    }


def classify_primary(row: Dict, endpoint: PrimaryEndpoint, alpha: float, margin: float) -> Dict:
    lo = float(row["delta_ci_lo"])
    hi = float(row["delta_ci_hi"])
    holm = float(row["holm_p"])
    out = dict(row)
    out.update({
        "hypothesis": endpoint.hypothesis,
        "claim": endpoint.claim,
        "direction": endpoint.direction,
        "noninferiority_margin": float(margin if endpoint.claim == "noninferiority" else 0.0),
    })
    if endpoint.hypothesis == "H1":
        passed = bool(np.isfinite(hi) and hi < 0.0 and np.isfinite(holm) and holm < alpha)
        out["decision"] = "PASS_SUPERIORITY" if passed else "NOT_CONFIRMED"
    elif endpoint.claim == "noninferiority":
        passed = bool(np.isfinite(lo) and lo > -float(margin))
        out["decision"] = "PASS_NONINFERIORITY" if passed else "NONINFERIORITY_NOT_CONFIRMED"
    else:
        passed = bool(np.isfinite(lo) and lo > 0.0 and np.isfinite(holm) and holm < alpha)
        out["decision"] = "LATENCY_COST_CONFIRMED" if passed else "LATENCY_COST_NOT_CONFIRMED"
    out["decision_pass"] = bool(passed)
    return out


def primary_statistics(
    summaries: pd.DataFrame,
    *,
    seed: int,
    n_boot: int,
    n_perm: int,
    alpha: float,
    recall_margin: float,
) -> pd.DataFrame:
    rows: List[Dict] = []
    for i, endpoint in enumerate(PRIMARY_ENDPOINTS):
        row = _paired_metric_row(
            summaries,
            endpoint.metric,
            seed_offset=seed + 1000 * (i + 1),
            n_boot=n_boot,
            n_perm=n_perm,
        )
        row["hypothesis"] = endpoint.hypothesis
        rows.append(row)
    adjusted = holm_adjust([r["paired_signflip_p"] for r in rows])
    classified: List[Dict] = []
    for endpoint, row, p_adj in zip(PRIMARY_ENDPOINTS, rows, adjusted):
        row["holm_p"] = float(p_adj)
        classified.append(classify_primary(row, endpoint, alpha, recall_margin))
    return pd.DataFrame(classified)


def secondary_statistics(
    summaries: pd.DataFrame,
    *,
    seed: int,
    n_boot: int,
    n_perm: int,
) -> pd.DataFrame:
    rows = []
    for i, metric in enumerate(SECONDARY_METRICS):
        if metric not in summaries.columns:
            continue
        row = _paired_metric_row(
            summaries,
            metric,
            seed_offset=seed + 100_000 + 1000 * i,
            n_boot=n_boot,
            n_perm=n_perm,
        )
        row["scope"] = "secondary_exploratory"
        row["holm_p"] = float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="SBA v9 preregistered confirmatory holdout experiment")
    p.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_HOLDOUT_SEEDS))
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
    p.add_argument("--recall-noninferiority-margin", type=float, default=DEFAULT_NONINFERIORITY_MARGIN)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--bootstrap", type=int, default=5000)
    p.add_argument("--permutations", type=int, default=10000)
    p.add_argument("--log-every", type=int, default=0)
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--out-dir", default="runs/sba_semantic_v9/confirmatory_52_61")
    args = p.parse_args(argv)
    setup_logging(args.quiet)

    if not 0.0 < args.alpha < 1.0:
        raise ValueError("alpha must be in (0,1)")
    if args.recall_noninferiority_margin <= 0.0:
        raise ValueError("recall non-inferiority margin must be positive")

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
    seeds = [int(x) for x in args.seeds]
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be unique")

    out = Path(args.out_dir)
    plan = build_analysis_plan(
        seeds=seeds,
        cfg=cfg,
        target_fpr=args.target_fpr,
        calib_windows=args.calib_windows,
        eval_windows=args.eval_windows,
        operational_windows=args.operational_windows,
        state_attack_windows=args.state_attack_windows,
        hidden=args.hidden,
        bootstrap=args.bootstrap,
        permutations=args.permutations,
        alpha=args.alpha,
        recall_margin=args.recall_noninferiority_margin,
    )
    plan_sha = write_locked_plan(out, plan)
    LOG.info("V9 analysis plan locked before simulation | sha256=%s", plan_sha)
    LOG.info("Primary Holm family size=%d | holdout seeds=%s", len(PRIMARY_ENDPOINTS), seeds)

    started = time.perf_counter()
    results = []
    for idx, seed in enumerate(seeds, 1):
        LOG.info("#" * 100)
        LOG.info("CONFIRMATORY %d/%d -> seed=%d", idx, len(seeds), seed)
        results.append(run_seed(
            seed=seed,
            calib_windows=args.calib_windows,
            eval_windows=args.eval_windows,
            operational_windows=args.operational_windows,
            state_attack_windows=args.state_attack_windows,
            hidden=args.hidden,
            target_fpr=args.target_fpr,
            cfg=cfg,
            out_dir=out / f"seed_{seed}",
            log_every=args.log_every,
        ))

    summaries = pd.concat([r["summary"] for r in results], ignore_index=True)
    summaries.to_csv(out / "all_seed_summary.csv", index=False)
    primary = primary_statistics(
        summaries,
        seed=min(seeds),
        n_boot=args.bootstrap,
        n_perm=args.permutations,
        alpha=args.alpha,
        recall_margin=args.recall_noninferiority_margin,
    )
    secondary = secondary_statistics(
        summaries,
        seed=min(seeds),
        n_boot=args.bootstrap,
        n_perm=args.permutations,
    )
    primary.to_csv(out / "primary_statistics.csv", index=False)
    secondary.to_csv(out / "secondary_statistics.csv", index=False)

    runtime = time.perf_counter() - started
    decisions = {
        row["hypothesis"]: row["decision"]
        for row in primary.to_dict(orient="records")
    }
    summary = {
        "analysis_plan_sha256": plan_sha,
        "plan_version": PLAN_VERSION,
        "seeds": seeds,
        "n_seeds": len(seeds),
        "primary_family_size": len(PRIMARY_ENDPOINTS),
        "alpha": args.alpha,
        "recall_noninferiority_margin": args.recall_noninferiority_margin,
        "runtime_seconds": runtime,
        "decisions": decisions,
        "primary_statistics": primary.to_dict(orient="records"),
    }
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    LOG.info("=" * 100)
    LOG.info("FINAL V9 CONFIRMATORY PRIMARY FAMILY")
    for row in primary.to_dict(orient="records"):
        LOG.info(
            "%s %-31s delta=%+.5f CI95=[%.5f,%.5f] raw-p=%.5g Holm=%.5g -> %s",
            row["hypothesis"], row["metric"], row["delta_v9_minus_static"],
            row["delta_ci_lo"], row["delta_ci_hi"],
            row["paired_signflip_p"], row["holm_p"], row["decision"],
        )
    LOG.info("Runtime %.1fs | output=%s", runtime, out)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
