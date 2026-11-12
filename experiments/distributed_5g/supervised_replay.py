"""Leave-one-seed-out supervised development study for the FULL_HYBRID ML branch.

Historical prospective captures from two seeds provide measured features and attack
labels for development only. The held-out seed contributes only its normal
calibration split before its test capture is scored. No run/window/NF identifier,
timestamp, scenario name, onset or episode identifier enters a classifier.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier, RandomForestClassifier

from experiments.live_5g.evaluate import CONTEXT, KEYS, conformal_threshold
from experiments.distributed_5g.paired import summarize


def load_test(root):
    features = pd.read_csv(root / "test" / "features.csv", float_precision="round_trip")
    labels = pd.read_csv(root / "test" / "labels.csv").fillna({"episode_id": ""})
    joined = features.merge(labels, on=KEYS, validate="one_to_one")
    return features, labels, joined


def models(seed):
    return {
        "hist_gbdt": HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=240, max_leaf_nodes=31,
            min_samples_leaf=20, l2_regularization=1.0,
            class_weight="balanced", early_stopping=False, random_state=seed),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=400, max_features="sqrt", min_samples_leaf=2,
            class_weight="balanced_subsample", n_jobs=-1, random_state=seed),
        "random_forest": RandomForestClassifier(
            n_estimators=400, max_features="sqrt", min_samples_leaf=2,
            class_weight="balanced_subsample", n_jobs=-1, random_state=seed),
    }


def stateful_without_mining(root, run_id):
    zones = json.loads((root / "zone_models.json").read_text())
    zone_for = {identity: model for model in zones.values() for identity in model["members"]}
    records = []
    with (root / "test" / "distributed_predictions.jsonl").open() as stream:
        for line in stream:
            window = json.loads(line)
            for identity, decision in window["decisions"].items():
                threshold = zone_for[identity]["threshold"]
                recovery = float(decision["recovery_accumulated"])
                without = bool(decision["local_security"] or recovery > threshold)
                records.append({
                    "run_id": run_id, "window_id": window["window"], "nf_id": identity,
                    "WITHOUT_MINING": without,
                    "recovery": recovery, "zone_threshold": float(threshold),
                })
    return pd.DataFrame(records)


def run(heldout, development, output, seed):
    heldout = Path(heldout)
    development = [Path(p) for p in development]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    train_parts = []
    development_runs = []
    for root in development:
        _, _, joined = load_test(root)
        development_runs.extend(sorted(joined.run_id.astype(str).unique()))
        train_parts.append(joined)
    train = pd.concat(train_parts, ignore_index=True)
    if set(train.run_id.astype(str)) & set(pd.read_csv(heldout / "test" / "features.csv").run_id.astype(str)):
        raise ValueError("Development and held-out run IDs overlap")

    test_features, test_labels, test = load_test(heldout)
    cal = pd.read_csv(heldout / "cal" / "features.csv", float_precision="round_trip")
    cal_labels = pd.read_csv(heldout / "cal" / "labels.csv")
    if cal_labels.attack.astype(bool).any():
        raise ValueError("Held-out threshold calibration must be normal-only")
    if set(cal.run_id.astype(str)) & set(train.run_id.astype(str)):
        raise ValueError("Calibration overlaps supervised development runs")

    run_id = str(test_features.run_id.iloc[0])
    state = stateful_without_mining(heldout, run_id)
    base = test_labels.merge(state, on=KEYS, validate="one_to_one")
    truth = base.attack.eq(1)
    normal = ~truth

    reports = []
    for name, model in models(seed).items():
        model.fit(train[CONTEXT].to_numpy(dtype=float), train.attack.astype(int).to_numpy())
        cal_scores = model.predict_proba(cal[CONTEXT].to_numpy(dtype=float))[:, 1]
        threshold = conformal_threshold(cal_scores, alpha=.001)
        test_scores = model.predict_proba(test_features[CONTEXT].to_numpy(dtype=float))[:, 1]
        alarm = test_scores > threshold if np.isfinite(threshold) else np.zeros(len(test_scores), dtype=bool)

        paired = base.copy()
        paired["ML_ALARM"] = alarm
        paired["FULL_HYBRID"] = paired.WITHOUT_MINING | (
            paired.ML_ALARM & (paired.recovery > .5 * paired.zone_threshold)
        )
        totals, _ = summarize(paired, ("WITHOUT_MINING", "FULL_HYBRID"), include_delay=False)
        full = next(x for x in totals if x["method"] == "FULL_HYBRID")
        without = next(x for x in totals if x["method"] == "WITHOUT_MINING")
        for row in (full, without):
            row["precision"] = row["tp"] / (row["tp"] + row["fp"]) if row["tp"] + row["fp"] else None
        reports.append({
            "model": name,
            "heldout_seed": seed,
            "heldout_run_id": run_id,
            "development_runs": development_runs,
            "features": CONTEXT,
            "calibration_rows": len(cal),
            "threshold": float(threshold) if np.isfinite(threshold) else None,
            "calibration_alarm_rate": float((cal_scores > threshold).mean()) if np.isfinite(threshold) else 0.0,
            "metrics": totals,
            "data_mining_increment": {
                "additional_tp": int((truth & paired.FULL_HYBRID & ~paired.WITHOUT_MINING).sum()),
                "additional_fp": int((normal & paired.FULL_HYBRID & ~paired.WITHOUT_MINING).sum()),
            },
            "fpr_target": .001,
            "fpr_target_met": bool(full["fpr"] <= .001),
            "scope": "Leave-one-seed-out supervised development; held-out normal calibration sets threshold; held-out attack labels used only after scoring.",
        })

    result = {
        "heldout_seed": seed,
        "rows": len(base),
        "development_rows": len(train),
        "reports": reports,
    }
    (output / "report.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--heldout", required=True)
    parser.add_argument("--development", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    run(args.heldout, args.development, args.output, args.seed)
