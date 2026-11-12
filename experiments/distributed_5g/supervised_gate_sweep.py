"""Development-only sweep of the stateful ML confirmation fraction.

The classifier and its probability threshold are unchanged from the leave-one-seed-out
study. Only the pre-prospective FULL_HYBRID confirmation fraction is evaluated on
historical held-out folds. Any selected fraction must be frozen before a new
prospective acquisition.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier, RandomForestClassifier

KEYS = ["run_id", "window_id", "nf_id"]
CONTEXT = ["request_count", "error_rate", "update_count", "interarrival_mean",
           "request_count_10s", "request_count_60s", "error_rate_10s", "error_rate_60s",
           "latency_ms", "heartbeat_age", "profile_changes", "forbidden_operation",
           "profile_policy_violation"]
FRACTIONS = (0.50, 0.60, 0.70, 0.80, 0.90, 0.95)


def conformal_threshold(scores, alpha=.001):
    values = np.asarray(scores, dtype=float)
    rank = math.ceil((len(values) + 1) * (1 - alpha))
    return float(np.sort(values)[rank - 1]) if rank <= len(values) else math.inf


def load_test(root):
    f = pd.read_csv(root / "test" / "features.csv", float_precision="round_trip")
    y = pd.read_csv(root / "test" / "labels.csv").fillna({"episode_id": ""})
    return f, y, f.merge(y, on=KEYS, validate="one_to_one")


def model_set(seed):
    return {
        "hist_gbdt": HistGradientBoostingClassifier(
            learning_rate=.05, max_iter=240, max_leaf_nodes=31, min_samples_leaf=20,
            l2_regularization=1., class_weight="balanced", early_stopping=False,
            random_state=seed),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=400, max_features="sqrt", min_samples_leaf=2,
            class_weight="balanced_subsample", n_jobs=-1, random_state=seed),
        "random_forest": RandomForestClassifier(
            n_estimators=400, max_features="sqrt", min_samples_leaf=2,
            class_weight="balanced_subsample", n_jobs=-1, random_state=seed),
    }


def hierarchy_state(root, run_id):
    zones = json.loads((root / "zone_models.json").read_text())
    zone_for = {identity: model for model in zones.values() for identity in model["members"]}
    rows = []
    with (root / "test" / "distributed_predictions.jsonl").open() as stream:
        for line in stream:
            window = json.loads(line)
            for identity, decision in window["decisions"].items():
                z = float(zone_for[identity]["threshold"])
                recovery = float(decision["recovery_accumulated"])
                rows.append({
                    "run_id": run_id, "window_id": window["window"], "nf_id": identity,
                    "recovery": recovery, "zone_threshold": z,
                    "WITHOUT_MINING": bool(decision["local_security"] or recovery > z),
                })
    return pd.DataFrame(rows)


def metrics(truth, pred):
    truth = np.asarray(truth, dtype=bool)
    pred = np.asarray(pred, dtype=bool)
    tp = int((truth & pred).sum()); fp = int((~truth & pred).sum())
    fn = int((truth & ~pred).sum()); tn = int((~truth & ~pred).sum())
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "recall": tp / (tp + fn) if tp + fn else None,
        "fpr": fp / (fp + tn) if fp + tn else None,
        "precision": tp / (tp + fp) if tp + fp else None,
    }


def run(heldout, development, output, seed):
    heldout = Path(heldout); development = [Path(x) for x in development]
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    parts = [load_test(root)[2] for root in development]
    train = pd.concat(parts, ignore_index=True)
    test_features, test_labels, _ = load_test(heldout)
    cal = pd.read_csv(heldout / "cal" / "features.csv", float_precision="round_trip")
    cal_labels = pd.read_csv(heldout / "cal" / "labels.csv")
    if cal_labels.attack.astype(bool).any():
        raise ValueError("Held-out calibration is not normal-only")
    state = hierarchy_state(heldout, str(test_features.run_id.iloc[0]))
    paired = test_labels.merge(state, on=KEYS, validate="one_to_one")
    truth = paired.attack.astype(bool).to_numpy()
    without = paired.WITHOUT_MINING.astype(bool).to_numpy()
    without_metrics = metrics(truth, without)

    reports = []
    for name, model in model_set(seed).items():
        model.fit(train[CONTEXT].to_numpy(float), train.attack.astype(int).to_numpy())
        cal_score = model.predict_proba(cal[CONTEXT].to_numpy(float))[:, 1]
        threshold = conformal_threshold(cal_score)
        score = model.predict_proba(test_features[CONTEXT].to_numpy(float))[:, 1]
        alarm = score > threshold if np.isfinite(threshold) else np.zeros(len(score), dtype=bool)
        sweep = []
        for fraction in FRACTIONS:
            full = without | (alarm & (paired.recovery.to_numpy(float) >
                                        fraction * paired.zone_threshold.to_numpy(float)))
            m = metrics(truth, full)
            sweep.append({
                "confirmation_fraction": fraction,
                "metrics": m,
                "additional_tp": int((truth & full & ~without).sum()),
                "additional_fp": int((~truth & full & ~without).sum()),
                "fpr_target_met": bool(m["fpr"] <= .001),
            })
        reports.append({
            "model": name, "threshold": threshold,
            "calibration_alarm_rate": float((cal_score > threshold).mean()),
            "without_mining": without_metrics, "sweep": sweep,
        })
    result = {"heldout_seed": seed, "development_rows": len(train), "reports": reports,
              "scope": "Historical development sweep only; selected confirmation fraction requires a fresh prospective confirmation run."}
    (output / "report.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--heldout", required=True)
    p.add_argument("--development", nargs="+", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, required=True)
    a = p.parse_args()
    run(a.heldout, a.development, a.output, a.seed)
