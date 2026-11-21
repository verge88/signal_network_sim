"""Unchanged frozen V9: cross-scenario Open5GS NF-window audit.

The model is an exact artifact from the pre-test 91201 V9 freeze.
All 3,600-window features are scored *before* labels are read.
Archived V5 samples are posthoc diagnostics, not held-out prospective evidence.
Fresh 913xx observations are prospective only when acquired after model freeze.
No use of attack labels to choose thresholds, models or signal features.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from experiments.live_5g.evaluate import load_frozen, score_frame

SCENARIOS = ("S1", "S2", "S3", "S4", "S5", "C", "D", "F", "G", "R")
METHODS = ("TG_V4", "TG_V8", "TG_V9")
KEYS = ["run_id", "window_id", "nf_id"]
ARCHIVED = (90401, 90402, 90405)
PROSPECTIVE = (91301, 91302, 91303)
WINDOWS = 3600
ACTORS = 32


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metrics(truth, pred):
    truth = np.asarray(truth, dtype=bool)
    pred = np.asarray(pred, dtype=bool)
    tp = int((truth & pred).sum())
    fn = int((truth & ~pred).sum())
    fp = int((~truth & pred).sum())
    tn = int((~truth & ~pred).sum())
    return {
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "recall": tp / (tp + fn) if tp + fn else None,
        "fpr": fp / (fp + tn) if fp + tn else None,
        "precision": tp / (tp + fp) if tp + fp else None,
    }


def summarize(root, frozen, output, phase, online_check=False):
    root, frozen, output = Path(root), Path(frozen), Path(output)
    expected = ARCHIVED if phase == "archival" else PROSPECTIVE
    manifest = json.loads((root / "test" / "manifest.json").read_text())
    seed = int(manifest["seed"])
    if seed not in expected or manifest["windows"] != WINDOWS or manifest["actors"] != ACTORS:
        raise ValueError("Not a preregistered 3600x32 source seed")
    if manifest.get("v9_structure_experiment"):
        raise ValueError("Cannot substitute an S9 result for a cross-scenario result")
    if manifest["phase"] != "test":
        raise ValueError("Not a held-out test acquisition")

    bundle = load_frozen(frozen)
    frozen_manifest = bundle["manifest"]
    if frozen_manifest.get("ml_method") != "TG_V9":
        raise ValueError("Model is not frozen TG_V9")
    meta = frozen_manifest.get("temporal_graph_v9", {})
    if meta.get("heldout_v9_test_used_for_selection") is not False:
        raise ValueError("Improperly tuned V9 bundle")
    if abs(float(meta["fusion"]["pair_support"]) - 0.62) > 1e-12:
        raise ValueError("V9 pair-support mismatch")
    if abs(float(meta["pair_branch"]["decay"]) - 0.75) > 1e-12:
        raise ValueError("V9 decay mismatch")
    if abs(float(meta["k4_branch"]["threshold_floor"]) - 0.91) > 1e-12:
        raise ValueError("V9 k4 floor mismatch")

    features_path = root / "test" / "features.csv"
    features = pd.read_csv(features_path, dtype={"run_id": str, "nf_id": str},
                           float_precision="round_trip")
    if len(features) != WINDOWS * ACTORS or features.duplicated(KEYS).any():
        raise ValueError("Invalid feature coverage")
    if set(features.window_id.astype(int)) != set(range(WINDOWS)):
        raise ValueError("Missing test windows")
    if set(features.nf_id.astype(str)) != {f"slot-{i}" for i in range(ACTORS)}:
        raise ValueError("Unexpected NF identities")
    if set(features.run_id) != {str(manifest["run_id"])}:
        raise ValueError("Unexpected test run ID")
    if set(features.run_id) & set(frozen_manifest.get("train_runs", []) +
                                  frozen_manifest.get("calibration_runs", [])):
        raise ValueError("Test run appears in frozen training/calibration")
    if phase == "prospective":
        if float(features.timestamp.min()) <= float(frozen_manifest["created_at"]):
            raise ValueError("Fresh prospective test must begin after frozen V9 model creation")
        if int(seed) not in PROSPECTIVE:
            raise ValueError("Unexpected prospective seed")
    else:
        # Archived 904xx acquisitions PREDATE this model's 912xx freeze.
        # Do not misrepresent archival analysis as independent prospective validation.
        if float(features.timestamp.max()) >= float(frozen_manifest["created_at"]):
            raise ValueError("Expected archival V5 sample predating V9 freeze")

    # Run full chronological inference before any labels are opened.
    features = features.sort_values(["run_id", "window_id", "nf_id"]).reset_index(drop=True)
    predictions = score_frame(bundle, features)
    prediction_frame = predictions.loc[predictions.method.isin(METHODS),
                                       KEYS + ["method", "alarm"]]
    if (set(prediction_frame.method) != set(METHODS)
            or len(prediction_frame) != len(features) * len(METHODS)
            or prediction_frame.duplicated(KEYS + ["method"]).any()):
        raise ValueError("Incomplete V4/V8/V9 replay")
    if not prediction_frame.alarm.isin((0, 1)).all():
        raise ValueError("Non-binary replay decisions")

    if online_check:
        online = pd.read_csv(root / "test" / "online_predictions.csv",
                             usecols=KEYS + ["method", "alarm"],
                             dtype={"run_id": str, "nf_id": str})
        original = online.loc[online.method.isin(METHODS)]
        joined = prediction_frame.merge(original, on=KEYS + ["method"],
                                        validate="one_to_one", suffixes=("_replay", "_online"))
        if len(joined) != len(prediction_frame) or not (
            joined.alarm_replay.astype(int) == joined.alarm_online.astype(int)).all():
            raise ValueError("Prospective online/offline V9 alarm mismatch")

    # Only now are attack truth and scenario IDs opened.
    labels_path = root / "test" / "labels.csv"
    labels = pd.read_csv(labels_path,
                         dtype={"run_id": str, "nf_id": str, "episode_id": str})
    if (len(labels) != len(features) or labels.duplicated(KEYS).any()
            or not labels.attack.isin((0, 1)).all()):
        raise ValueError("Invalid attack ground truth")
    attacked_scenarios = set(labels.loc[labels.attack.astype(bool), "scenario"])
    if not set(SCENARIOS).issubset(attacked_scenarios):
        raise ValueError(f"Required attack scenarios absent: {set(SCENARIOS)-attacked_scenarios}")
    paired = prediction_frame.merge(labels, on=KEYS, validate="many_to_one")
    if len(paired) != len(prediction_frame):
        raise ValueError("Inference and label keys disagree")

    totals = {}
    by_scenario = {}
    for method in METHODS:
        portion = paired.loc[paired.method == method]
        totals[method] = metrics(portion.attack, portion.alarm)
        by_scenario[method] = {}
        for scenario in SCENARIOS:
            subset = portion.loc[(portion.scenario == scenario) &
                                 portion.attack.astype(bool)]
            if not len(subset):
                raise ValueError("Empty attack scenario")
            by_scenario[method][scenario] = {
                "tp": int(subset.alarm.sum()),
                "positive": int(len(subset)),
                "recall": float(subset.alarm.mean()),
                "episodes": int(subset.episode_id.nunique()),
                "episodes_detected": int(subset.groupby("episode_id").alarm.max().sum()),
            }

    output.mkdir(parents=True, exist_ok=True)
    report = {
        "seed": seed, "phase": phase, "source_run_id": str(manifest["run_id"]),
        "frozen_v9_seed": int(frozen_manifest["seed"]),
        "model_sha256": digest(frozen / "model.joblib"),
        "features_sha256": digest(features_path),
        "labels_sha256": digest(labels_path),
        "totals": totals, "by_scenario": by_scenario,
        "fpr_all_methods_below_0_001": all(totals[m]["fpr"] <= 0.001 for m in METHODS),
        "v9_fpr_below_0_001": totals["TG_V9"]["fpr"] <= 0.001,
        "heldout_after_freeze": phase == "prospective",
        "scope": "S1-S5, C,D,F,G,R across 3600x32; independent only for prospective 913xx",
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps(report, allow_nan=False), flush=True)


def aggregate(root, output, phase):
    root, output = Path(root), Path(output)
    expected = set(ARCHIVED if phase == "archival" else PROSPECTIVE)
    reports = [json.loads(p.read_text()) for p in sorted(root.rglob("report.json"))]
    if len(reports) != len(expected) or {r["seed"] for r in reports} != expected:
        raise ValueError(f"Require exactly {sorted(expected)} seed reports")
    if any(r["phase"] != phase or r["heldout_after_freeze"] != (phase == "prospective")
           for r in reports):
        raise ValueError("Mixed archival/prospective reports")
    if len({r["model_sha256"] for r in reports}) != 1:
        raise ValueError("Frozen model drift across sources")

    pooled = {}
    by_scenario = {}
    for method in METHODS:
        counts = {key: sum(r["totals"][method][key] for r in reports)
                  for key in ("tp", "fn", "fp", "tn")}
        tp, fn, fp, tn = (counts[k] for k in ("tp", "fn", "fp", "tn"))
        pooled[method] = {**counts,
                          "recall": tp/(tp+fn), "fpr": fp/(fp+tn),
                          "precision": tp/(tp+fp) if tp+fp else None}
        by_scenario[method] = {}
        for s in SCENARIOS:
            tp = sum(r["by_scenario"][method][s]["tp"] for r in reports)
            pos = sum(r["by_scenario"][method][s]["positive"] for r in reports)
            by_scenario[method][s] = {
                "tp": tp, "positive": pos, "recall": tp/pos if pos else None,
                "episodes": sum(r["by_scenario"][method][s]["episodes"] for r in reports),
                "episodes_detected": sum(r["by_scenario"][method][s]["episodes_detected"] for r in reports),
            }
    expected_total = WINDOWS * ACTORS * len(expected)
    if any(sum(pooled[m][k] for k in ("tp", "fn", "fp", "tn")) != expected_total
           for m in METHODS):
        raise ValueError("Wrong aggregation denominators")
    if any(sum(by_scenario[m][s]["positive"] for s in SCENARIOS)
           != pooled[m]["tp"]+pooled[m]["fn"] for m in METHODS):
        raise ValueError("Attack scenarios not exhaustive")

    result = {
        "phase": phase, "source_seeds": sorted(expected),
        "model_sha256": reports[0]["model_sha256"],
        "model_seed": reports[0]["frozen_v9_seed"],
        "pooled": pooled, "by_scenario": by_scenario,
        "per_seed": {str(r["seed"]): r["totals"] for r in reports},
        "criteria": {
            "all_seed_v9_fpr_le_0_001": all(r["v9_fpr_below_0_001"] for r in reports),
            "v9_pooled_recall_ge_0_90": pooled["TG_V9"]["recall"] >= .9,
        },
        "not_S9": True,
        "independent_prospective": phase == "prospective",
        "limitations": ("Only 3600-window simulated Open5GS scenarios; "
                        "archived 904xx analyses are retrospective; "
                        "V9 trained/calibrated for S9 and frozen before fresh 913xx samples."),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "aggregate.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps(result, allow_nan=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="action", required=True)
    for action in ("audit", "aggregate"):
        p = sub.add_parser(action)
        p.add_argument("root", type=Path)
        p.add_argument("output", type=Path)
        p.add_argument("--phase", choices=("archival", "prospective"), required=True)
        if action == "audit":
            p.add_argument("--frozen", type=Path, required=True)
            p.add_argument("--online-check", action="store_true")
    args = parser.parse_args()
    if args.action == "audit":
        summarize(args.root, args.frozen, args.output, args.phase,
                  online_check=args.online_check)
    else:
        aggregate(args.root, args.output, args.phase)


if __name__ == "__main__":
    main()
