"""Prospective exploratory falsification of leave-one-trust-origin-out scoring.

The score is an intentionally transparent *proxy*, not a trained one-class
model.  Its purpose is to test a structural failure mode of min-LOO aggregation
under one-origin evidence injection.  The counterfactual worlds are identical
except for explicit CONSISTENT -> INCONSISTENT report changes.

No attack labels, simulator latent operational state or compromise ground truth
are read by the scoring or conformal-calibration functions.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "5g"))

from sba_semantic_v5 import ORIGIN_ORDER  # noqa: E402
from sba_semantic_v8 import (  # noqa: E402
    EvidenceState,
    OperationalState,
    SemanticFact,
    SemanticSbaSimulator,
)

FACT = SemanticFact.TOKEN
ORIGINS = tuple(ORIGIN_ORDER[FACT][:5])
ARMS = ("raw_pooled", "raw_context", "loo_pooled", "loo_context", "q3")
CONTEXTS = ("normal_indirect", "diurnal_peak")


def event_reports(window, event_id):
    """Last observed report per independent origin for one event and fact."""
    reports = {}
    for report in window.observations:
        if (
            report.event_id == event_id
            and report.fact == FACT
            and report.origin in ORIGINS
        ):
            prior = reports.get(report.origin)
            if prior is None or report.observed_at > prior.observed_at:
                reports[report.origin] = report
    return reports


def choose_target(window):
    """Predefined label-free eligibility rule, applied identically to null/attack."""
    for event in window.events:
        reports = event_reports(window, event.event_id)
        consistent = [
            origin for origin in ORIGINS
            if origin in reports and reports[origin].state == EvidenceState.CONSISTENT
        ]
        if len(consistent) >= 3:
            return reports, consistent
    return None


def score(reports):
    """Raw count and min leave-one-independent-origin-out count."""
    raw = sum(
        int(report.state == EvidenceState.INCONSISTENT)
        for report in reports.values()
    )
    loo = min(
        sum(
            int(report.state == EvidenceState.INCONSISTENT)
            for origin, report in reports.items()
            if origin != excluded
        )
        for excluded in ORIGINS
    )
    return int(raw), int(loo)


def intervene(reports, consistent, n_origins):
    """Change only already observed CONSISTENT evidence, never UNKNOWN."""
    out = dict(reports)
    changed = consistent[:n_origins]
    for origin in changed:
        old = out[origin]
        assert old.state == EvidenceState.CONSISTENT
        out[origin] = replace(old, state=EvidenceState.INCONSISTENT)
    assert len(changed) == n_origins
    for origin, old in reports.items():
        if old.state == EvidenceState.UNKNOWN:
            assert out[origin].state == EvidenceState.UNKNOWN
        if origin not in changed:
            assert out[origin] == old
    return out


def conformal_p(calibration, observed):
    """Conservative finite-sample rank p-value (no exchangeability claim)."""
    if not calibration:
        raise ValueError("empty calibration stratum")
    return (1 + sum(x >= observed for x in calibration)) / (len(calibration) + 1)


def wilson_interval(hits, total, z=1.959963984540054):
    if not total:
        return (float("nan"), float("nan"))
    p = hits / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    margin = z * math.sqrt(
        (p * (1 - p) + z * z / (4 * total)) / total
    ) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def run_one_seed(seed, n_calib, n_eval, alpha):
    sim = SemanticSbaSimulator(seed=seed)
    window_id = 0
    attempts = defaultdict(int)

    def next_eligible(context):
        nonlocal window_id
        while True:
            if attempts[context] >= 20 * (n_calib + n_eval):
                raise RuntimeError("too few eligible windows")
            attempts[context] += 1
            window = sim.generate_window(
                window_id,
                context=context,
                operational_state=OperationalState.NORMAL,
                false_marker=False,
            )
            window_id += 1
            target = choose_target(window)
            if target is not None:
                return target

    calibration = {}
    for context in CONTEXTS:
        raw_values, loo_values = [], []
        for _ in range(n_calib):
            reports, _consistent = next_eligible(context)
            raw, loo = score(reports)
            raw_values.append(raw)
            loo_values.append(loo)
        calibration[context] = {"raw": raw_values, "loo": loo_values}
    pooled = {
        kind: [v for context in CONTEXTS for v in calibration[context][kind]]
        for kind in ("raw", "loo")
    }

    counts = defaultdict(lambda: [0, 0])
    structural_counterexamples = 0
    for context in CONTEXTS:
        for _ in range(n_eval):
            reports, consistent = next_eligible(context)
            base_raw, _base_loo = score(reports)
            for injected in (0, 1, 2, 3):
                modified = intervene(reports, consistent, injected)
                raw, loo = score(modified)
                if injected == 1 and base_raw == 0:
                    # Single-origin evidence attack is fully erased by min-LOO.
                    assert raw == 1 and loo == 0
                    structural_counterexamples += 1
                alarms = {
                    "raw_pooled": conformal_p(pooled["raw"], raw) < alpha,
                    "raw_context": conformal_p(calibration[context]["raw"], raw) < alpha,
                    "loo_pooled": conformal_p(pooled["loo"], loo) < alpha,
                    "loo_context": conformal_p(calibration[context]["loo"], loo) < alpha,
                    "q3": raw >= 3,
                }
                for arm in ARMS:
                    entry = counts[(context, injected, arm)]
                    entry[0] += int(alarms[arm])
                    entry[1] += 1

    if structural_counterexamples <= 0:
        raise AssertionError("no eligible clean one-origin counterexamples observed")

    rows = []
    for (context, injected, arm), (hits, n) in sorted(counts.items()):
        low, high = wilson_interval(hits, n)
        rows.append({
            "seed": seed,
            "context": context,
            "injected_origins": injected,
            "arm": arm,
            "alerts": hits,
            "n": n,
            "alert_rate": hits / n,
            "wilson95_lo": low,
            "wilson95_hi": high,
        })
    return rows, {
        "seed": seed,
        "structural_single_origin_counterexamples": structural_counterexamples,
        "generated_attempts_by_context": dict(attempts),
        "calibration_per_context": n_calib,
        "evaluation_per_context": n_eval,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", nargs="+", type=int, default=[111, 112, 113, 114, 115])
    p.add_argument("--calib", type=int, default=600)
    p.add_argument("--eval", type=int, default=300)
    p.add_argument("--alpha", type=float, default=0.01)
    p.add_argument("--out-dir", default="runs/sba_loo_falsification/111_115")
    args = p.parse_args()
    if not (0 < args.alpha < 1):
        raise ValueError("alpha must be between 0 and 1")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("seeds must be unique")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    plan = {
        "seeds": args.seeds,
        "contexts": list(CONTEXTS),
        "origins": [x.value for x in ORIGINS],
        "fact": FACT.value,
        "calib_per_seed_context": args.calib,
        "eval_per_seed_context": args.eval,
        "alpha": args.alpha,
        "score": "per-event TOKEN inconsistent count; min leave-one-origin-out",
        "attack": "paired consistent-to-inconsistent report injection in 1/2/3 origins",
        "status": "exploratory_falsification_not_trained_one_class",
    }
    canonical = json.dumps(plan, sort_keys=True, separators=(",", ":"))
    plan["sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    (out / "analysis_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")

    all_rows, audits = [], []
    for seed in args.seeds:
        rows, audit = run_one_seed(seed, args.calib, args.eval, args.alpha)
        all_rows.extend(rows)
        audits.append(audit)
        print(f"seed={seed}: structural counterexamples={audit['structural_single_origin_counterexamples']}", flush=True)

    columns = [
        "seed", "context", "injected_origins", "arm", "alerts", "n",
        "alert_rate", "wilson95_lo", "wilson95_hi",
    ]
    with (out / "per_seed.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(all_rows)

    combined = defaultdict(lambda: [0, 0])
    for row in all_rows:
        for context in (row["context"], "ALL"):
            x = combined[(context, row["injected_origins"], row["arm"])]
            x[0] += row["alerts"]
            x[1] += row["n"]
    aggregate = []
    for (context, injected, arm), (hits, n) in sorted(combined.items()):
        lo, hi = wilson_interval(hits, n)
        aggregate.append({
            "context": context,
            "injected_origins": injected,
            "arm": arm,
            "alerts": hits,
            "n": n,
            "alert_rate": hits / n,
            "wilson95_lo": lo,
            "wilson95_hi": hi,
        })
    with (out / "aggregate.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns[1:])
        writer.writeheader()
        writer.writerows(aggregate)
    summary = {
        "plan_sha256": plan["sha256"],
        "git_sha": os.getenv("GITHUB_SHA", "unknown"),
        "audits": audits,
        "aggregate": aggregate,
        "limits": [
            "Transparent handcrafted proxy, not a trained s_theta model.",
            "Event selection conditions on at least three consistent origin reports.",
            "Synthetic observation-level injection, not an actual SBA network attack.",
            "Within-seed events are dependent; Wilson intervals are descriptive only.",
            "Conditional conformal exchangeability and 0.1% FPR are not established.",
        ],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({
        "plan_sha256": plan["sha256"],
        "structural_counterexamples": sum(a["structural_single_origin_counterexamples"] for a in audits),
        "pooled": [x for x in aggregate if x["context"] == "ALL"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
