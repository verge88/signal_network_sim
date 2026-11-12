"""Paired alternative-detector replay on frozen observations from the prospective IF experiment.

This is not a new prospective acquisition. It refits only on the archived normal
train/cal partitions and evaluates the unchanged held-out test observations.
The stateful hierarchy excluding the ML branch is replayed from archived decisions;
its recovery accumulator is independent of the ML alarm.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from experiments.live_5g.evaluate import fit, load_frozen, score_frame, KEYS
from experiments.distributed_5g.paired import summarize


def read_json(path):
    return json.loads(Path(path).read_text())


def run(root, detector, seed):
    root = Path(root)
    frozen = root / "alt_frozen"
    manifest = fit(root / "train", root / "cal", frozen, alpha=.001, seed=seed, detector=detector)
    if not manifest["threshold_sufficient"]:
        raise ValueError("Calibration sample is insufficient for requested threshold")

    bundle = load_frozen(frozen)
    features = pd.read_csv(root / "test" / "features.csv", float_precision="round_trip")
    labels = pd.read_csv(root / "test" / "labels.csv").fillna({"episode_id": ""})
    predictions = score_frame(bundle, features)
    context = predictions[predictions.method == "IF_CONTEXT"].set_index(KEYS).alarm.to_dict()

    zones = read_json(root / "zone_models.json")
    zone_for = {identity: model for model in zones.values() for identity in model["members"]}
    records = []
    with (root / "test" / "distributed_predictions.jsonl").open() as stream:
        for line in stream:
            window = json.loads(line)
            for identity, decision in window["decisions"].items():
                threshold = zone_for[identity]["threshold"]
                recovery = decision["recovery_accumulated"]
                without_ml = bool(decision["local_security"] or recovery > threshold)
                ml = bool(context[(features.run_id.iloc[0], window["window"], identity)])
                full = bool(without_ml or (ml and recovery > .5 * threshold))
                records.append({
                    "run_id": features.run_id.iloc[0],
                    "window_id": window["window"],
                    "nf_id": identity,
                    "WITHOUT_MINING": without_ml,
                    "FULL_HYBRID": full,
                })

    distributed = pd.DataFrame(records)
    paired = labels.merge(distributed, on=KEYS, validate="one_to_one")
    totals, episodes = summarize(paired, ("WITHOUT_MINING", "FULL_HYBRID"), include_delay=False)
    truth = paired.attack.eq(1)
    full = next(x for x in totals if x["method"] == "FULL_HYBRID")
    without = next(x for x in totals if x["method"] == "WITHOUT_MINING")
    for row in (full, without):
        row["precision"] = row["tp"] / (row["tp"] + row["fp"]) if row["tp"] + row["fp"] else None

    result = {
        "detector": detector,
        "seed": seed,
        "source_run_id": features.run_id.iloc[0],
        "rows": len(paired),
        "model_sha256": manifest["model_sha256"],
        "metrics": totals,
        "data_mining_increment": {
            "additional_tp": int((truth & paired.FULL_HYBRID & ~paired.WITHOUT_MINING).sum()),
            "additional_fp": int((~truth & paired.FULL_HYBRID & ~paired.WITHOUT_MINING).sum()),
        },
        "fpr_target": .001,
        "fpr_target_met": full["fpr"] <= .001,
        "scope": "Paired retrospective replay on observations captured by prospective IF run; no held-out labels used for fitting or threshold calibration.",
    }
    out = root / "alt_summary"
    out.mkdir(exist_ok=True)
    (out / "report.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--detector", required=True, choices=["ocsvm_rbf", "lof", "elliptic"])
    parser.add_argument("--seed", required=True, type=int)
    args = parser.parse_args()
    run(args.root, args.detector, args.seed)
