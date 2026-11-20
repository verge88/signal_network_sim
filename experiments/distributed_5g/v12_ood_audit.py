"""Posthoc OOD diagnostic for frozen S9-specific V12 on V5 S1-S5/C/D/F/G/R.

Only features enter the prefix-causal inference loops. Labels are opened after
both detectors have returned predictions for every NF and window. No training,
threshold optimization, or changes to the frozen V12 method are performed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.distributed_5g.v11_staircase import CausalStaircaseDetector
from experiments.distributed_5g.v12_staircase import ConfirmedStaircaseDetector

SEEDS = (90401, 90402, 90405)
WINDOWS = 3600
NF_COUNT = 32
KEYS = ["run_id", "window_id", "nf_id"]
SCENARIOS = ("S1", "S2", "S3", "S4", "S5", "C", "D", "F", "G", "R")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metrics(truth, predicted):
    truth = np.asarray(truth, dtype=bool)
    predicted = np.asarray(predicted, dtype=bool)
    tp = int((truth & predicted).sum())
    fn = int((truth & ~predicted).sum())
    fp = int((~truth & predicted).sum())
    tn = int((~truth & ~predicted).sum())
    return dict(tp=tp, fp=fp, fn=fn, tn=tn,
                recall=tp / (tp + fn) if tp + fn else None,
                fpr=fp / (fp + tn) if fp + tn else None,
                precision=tp / (tp + fp) if tp + fp else None)


def feature_only_replay(features):
    """Use only complete chronologically observed features; no labels/schedules."""
    if not set(KEYS + ["request_count"]).issubset(features):
        raise ValueError("Missing feature replay columns")
    if len(features) != WINDOWS * NF_COUNT or features.duplicated(KEYS).any():
        raise ValueError("Expected complete 3600x32 unique feature rows")
    if features.run_id.nunique() != 1:
        raise ValueError("Multiple run IDs in one replay")
    ordered = features.sort_values(["window_id", "nf_id"])
    nfs = tuple(sorted(ordered.nf_id.astype(str).unique()))
    if set(nfs) != {f"slot-{n}" for n in range(NF_COUNT)}:
        raise ValueError("NF identifier coverage mismatch")
    if set(ordered.window_id.unique()) != set(range(WINDOWS)):
        raise ValueError("Missing or unexpected windows")
    zones = {nf: int(nf[5:]) // 8 for nf in nfs}
    v11 = CausalStaircaseDetector(zones)
    v12 = ConfirmedStaircaseDetector(zones)
    decisions = []
    for window, part in ordered.groupby("window_id", sort=True):
        if len(part) != NF_COUNT:
            raise ValueError("Incomplete single-window coverage")
        counts = dict(zip(part.nf_id.astype(str), part.request_count))
        old = v11.step(int(window), counts)
        new = v12.step(int(window), counts)
        if any(new[nf] > old[nf] for nf in nfs):
            raise ValueError("Frozen confirmed V12 must be a subset of V11")
        run_id = str(part.run_id.iloc[0])
        decisions.extend((run_id, int(window), nf, old[nf], new[nf]) for nf in nfs)
    return pd.DataFrame(decisions, columns=KEYS + ["TG_V11", "TG_V12"])


def audit(root, output):
    root, output = Path(root), Path(output)
    capture = root / "test"
    manifest = json.loads((capture / "manifest.json").read_text())
    seed = int(manifest["seed"])
    if seed not in SEEDS or manifest["phase"] != "test":
        raise ValueError("Not a preregistered archived V5 final test seed")
    if manifest["windows"] != WINDOWS or manifest["actors"] != NF_COUNT:
        raise ValueError("Unexpected V5 experiment dimensions")
    feature_path = capture / "features.csv"
    features = pd.read_csv(feature_path, dtype={"run_id": str, "nf_id": str})
    # Both streams are irrevocably predicted before accessing the label file.
    predicted = feature_only_replay(features)

    label_path = capture / "labels.csv"
    labels = pd.read_csv(label_path, dtype={"run_id": str, "nf_id": str,
                                            "episode_id": str}).fillna({"episode_id": ""})
    if len(labels) != len(predicted) or labels.duplicated(KEYS).any():
        raise ValueError("Incomplete or duplicated label coverage")
    if not set(SCENARIOS).issubset(set(labels.loc[labels.attack == 1, "scenario"])):
        raise ValueError("Archived scenarios are missing")
    if bool(labels.loc[~labels.attack.astype(bool), "scenario"].ne("N1").any()):
        raise ValueError("Normal labels are not N1")
    frame = labels.merge(predicted, on=KEYS, validate="one_to_one")
    if len(frame) != WINDOWS * NF_COUNT:
        raise ValueError("Feature/label key mismatch")
    total = {method: metrics(frame.attack, frame[method])
             for method in ("TG_V11", "TG_V12")}
    scenarios = {}
    for scenario in SCENARIOS:
        attacked = frame.loc[(frame.scenario == scenario) & frame.attack.astype(bool)]
        episodes = attacked.episode_id.astype(str)
        scenarios[scenario] = {
            "attacked_nf_windows": len(attacked),
            "episodes": int(episodes.nunique()),
            **{
                method: {
                    "detected_nf_windows": int(attacked[method].sum()),
                    "window_recall": float(attacked[method].mean()),
                    "episodes_detected": int(
                        attacked.groupby("episode_id")[method].max().sum()),
                }
                for method in ("TG_V11", "TG_V12")
            },
        }
    report = dict(seed=seed, source_run_id=str(manifest["run_id"]),
                  phase="posthoc_ood_diagnostic", windows=WINDOWS, actors=NF_COUNT,
                  source_features_sha256=digest(feature_path),
                  source_labels_sha256=digest(label_path),
                  detector_source_sha256=digest("experiments/distributed_5g/v12_staircase.py"),
                  confusion=total, scenarios=scenarios,
                  no_label_leakage=True, no_retuning=True,
                  limitations="S9-specific detector; V5 scenarios are a separate older archive, not fresh prospective S9 confirmation")
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, allow_nan=False), flush=True)


def aggregate(root, output):
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    reports = [json.loads(p.read_text()) for p in sorted(root.rglob("report.json"))]
    if len(reports) != len(SEEDS) or {r["seed"] for r in reports} != set(SEEDS):
        raise ValueError("Aggregate requires exactly three preregistered V5 archived seeds")
    if any(r["phase"] != "posthoc_ood_diagnostic" for r in reports):
        raise ValueError("A V12 OOD report was mislabeled")
    if len({r["detector_source_sha256"] for r in reports}) != 1:
        raise ValueError("Detector changed across reruns")
    pooled = {}
    for method in ("TG_V11", "TG_V12"):
        values = [r["confusion"][method] for r in reports]
        pooled[method] = metrics(
            np.concatenate([
                np.concatenate([np.ones(v["tp"], dtype=bool),
                                np.ones(v["fn"], dtype=bool),
                                np.zeros(v["fp"] + v["tn"], dtype=bool)])
                for v in values
            ]),
            np.concatenate([
                np.concatenate([np.ones(v["tp"], dtype=bool),
                                np.zeros(v["fn"], dtype=bool),
                                np.ones(v["fp"], dtype=bool),
                                np.zeros(v["tn"], dtype=bool)])
                for v in values
            ]),
        )
    scenario_pooled = {}
    for scenario in SCENARIOS:
        scenario_pooled[scenario] = {
            "attacked_nf_windows": sum(r["scenarios"][scenario]["attacked_nf_windows"]
                                       for r in reports),
            "episodes": sum(r["scenarios"][scenario]["episodes"] for r in reports),
        }
        for method in ("TG_V11", "TG_V12"):
            detected = sum(r["scenarios"][scenario][method]["detected_nf_windows"]
                           for r in reports)
            total = scenario_pooled[scenario]["attacked_nf_windows"]
            scenario_pooled[scenario][method] = {
                "detected_nf_windows": detected,
                "window_recall": detected / total if total else None,
                "episodes_detected": sum(
                    r["scenarios"][scenario][method]["episodes_detected"]
                    for r in reports),
            }
    result = dict(phase="posthoc_ood_diagnostic", source_seeds=list(SEEDS),
                  pooled=pooled, by_scenario=scenario_pooled,
                  per_seed={str(r["seed"]):r["confusion"] for r in reports},
                  no_new_prospective_claim=True, no_retuning=True,
                  limitations="OOD results are diagnostic only; no ≥90% goal claimed outside S9")
    (output / "aggregate.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, allow_nan=False), flush=True)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="action", required=True)
    for action in ("audit", "aggregate"):
        sub.add_parser(action).add_argument("root", type=Path)
        # CLI has positional output as in V12.
    for action in sub.choices.values():
        action.add_argument("output", type=Path)
    args = p.parse_args()
    (audit if args.action == "audit" else aggregate)(args.root, args.output)


if __name__ == "__main__":
    main()
