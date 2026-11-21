"""Fit a passive NF-window detector and compare it with recorded decisions on paired windows."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from .cross_scenario_detector import (
    MEASUREMENTS, METHOD, SLOTS, CrossScenarioDetector, WindowScorer, feature_panel, flags,
    operating_threshold,
)
from .v9_cross_scenario_audit import KEYS, SCENARIOS, metrics


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def aligned(frame, index):
    keyed = frame.set_index(KEYS, verify_integrity=True)
    if len(keyed) != len(index) or not index.isin(keyed.index).all():
        raise ValueError("Mismatched NF-window identities")
    return keyed.reindex(index)


def read_csv(path, columns):
    return pd.read_csv(path, usecols=list(columns), dtype={"run_id": str, "nf_id": str},
                       float_precision="round_trip")


class Capture:
    def __init__(self, root):
        self.path = Path(root) / "test"
        manifest = json.loads((self.path / "manifest.json").read_text())
        self.seed, self.run = manifest["seed"], str(manifest["run_id"])
        self.windows = manifest["windows"]
        if manifest["phase"] != "test" or manifest["actors"] != 32 or self.windows < 1:
            raise ValueError("Expected a complete 32-NF capture")
        self.index = pd.MultiIndex.from_product([[self.run], range(self.windows), SLOTS], names=KEYS)
        frame = read_csv(self.path / "features.csv", [*KEYS, *MEASUREMENTS])
        self.values = aligned(frame, self.index)[list(MEASUREMENTS)].to_numpy().reshape(self.windows, 32, -1)
        self.local = np.empty((self.windows, 32), bool)
        self.allowed = np.empty_like(self.local)
        self.full = np.empty_like(self.local)
        self.recorded_ml = np.empty_like(self.local)
        self.availability = np.empty_like(self.local)
        self.unknown = np.empty_like(self.local)
        columns = ("local_security", "allowed", "security", "if_context_alarm", "availability", "unknown")
        count = 0
        with (self.path / "distributed_predictions.jsonl").open() as stream:
            for window, line in enumerate(stream):
                item = json.loads(line)
                decisions = item["decisions"]
                if (window >= self.windows or item["window"] != window or item["complete"] is not True
                        or set(decisions) != set(SLOTS)):
                    raise ValueError("Incomplete or out-of-order hierarchy recording")
                observations = [[decisions[nf][field] for field in columns] for nf in SLOTS]
                block = flags(observations, (32, len(columns)))
                (self.local[window], self.allowed[window], self.full[window], self.recorded_ml[window],
                 self.availability[window], self.unknown[window]) = block.T
                count += 1
        if count != self.windows:
            raise ValueError("Missing decision windows")

    def labels(self):
        frame = read_csv(self.path / "labels.csv", [*KEYS, "attack", "scenario"])
        frame = aligned(frame, self.index)
        truth = flags(frame.attack.to_numpy(), (len(self.index),)).reshape(self.windows, 32)
        return truth, frame.scenario.to_numpy().reshape(self.windows, 32)

    def original_alarm(self, method):
        columns = [*KEYS, "method", "alarm"]
        selected = []
        for part in pd.read_csv(self.path / "online_predictions.csv", usecols=columns,
                                dtype={"run_id": str, "nf_id": str}, chunksize=160000):
            selected.append(part.loc[part.method == method, KEYS + ["alarm"]])
        frame = aligned(pd.concat(selected), self.index)
        result = flags(frame.alarm.to_numpy(), (len(self.index),)).reshape(self.windows, 32)
        if not np.array_equal(result, self.recorded_ml):
            raise ValueError(f"Recorded {method} alarms differ from the hierarchy's ML input")
        return result


def fit(train_paths, calibration_paths, output):
    training, calibration, seeds = [], [], set()
    matrix, targets = [], []

    def register(capture, group):
        if capture.seed in seeds or capture.run in training + calibration:
            raise ValueError("Fit and calibration acquisitions must be distinct")
        seeds.add(capture.seed)
        group.append(capture.run)

    for path in train_paths:
        capture = Capture(path)
        register(capture, training)
        truth, _ = capture.labels()
        eligible = ~capture.local & ~capture.allowed
        matrix.append(feature_panel(capture.values)[eligible])
        targets.append(truth[eligible])
    x, y = np.concatenate(matrix), np.concatenate(targets)
    if len(np.unique(y)) != 2:
        raise ValueError("Training needs both benign and anomalous eligible windows")
    classifier = HistGradientBoostingClassifier(
        learning_rate=.07, max_iter=160, max_leaf_nodes=15, min_samples_leaf=30,
        l2_regularization=3., early_stopping=False, random_state=731,
    )
    classifier.fit(x, y, sample_weight=np.where(y, 32., 1.))
    model = CrossScenarioDetector(classifier, 0., tuple(training), ())
    cuts = []
    for path in calibration_paths:
        capture = Capture(path)
        register(capture, calibration)
        scores = model.score_features(feature_panel(capture.values))
        truth, _ = capture.labels()
        cuts.append(operating_threshold(scores, truth, capture.local, capture.allowed, budget=.0005))
    model.threshold = max(cuts)
    model.calibration_runs = tuple(calibration)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output / "model.joblib", compress=3)
    metadata = dict(method=METHOD, model_sha256=sha256(output / "model.joblib"),
                    train_runs=training, calibration_runs=calibration, consumed_seeds=sorted(seeds),
                    calibration_budget=.0005, per_run_thresholds=dict(zip(calibration, cuts)),
                    threshold=model.threshold, measurements=list(MEASUREMENTS),
                    decision="local_security OR (NOT allowed AND score > threshold)")
    write_json(output / "manifest.json", metadata)
    print(json.dumps(metadata), flush=True)


def load_model(directory):
    path = Path(directory)
    meta = json.loads((path / "manifest.json").read_text())
    if meta["method"] != METHOD or sha256(path / "model.joblib") != meta["model_sha256"]:
        raise ValueError("Unexpected detector or model digest")
    model = joblib.load(path / "model.joblib")
    if (tuple(meta["train_runs"]) != model.fit_runs or tuple(meta["calibration_runs"]) != model.calibration_runs
            or model.threshold != meta["threshold"]):
        raise ValueError("Frozen model and manifest disagree")
    return model, meta


def verify_streaming(model, capture, candidate):
    scorer = WindowScorer(model)
    for window, values in enumerate(capture.values):
        rows = [dict(run_id=capture.run, window_id=window, nf_id=nf, **dict(zip(MEASUREMENTS, row)))
                for nf, row in zip(SLOTS, values)]
        decisions = {nf: {"local_security": local, "allowed": allowed}
                     for nf, local, allowed in zip(SLOTS, capture.local[window], capture.allowed[window])}
        alarms = scorer.score(rows, decisions)
        different = [nf for nf, expected in zip(SLOTS, candidate[window]) if alarms[nf] != expected]
        if different:
            raise ValueError(f"Streaming replay differs in {capture.run} at window {window}: {different}")


def evaluate(directory, root, reference, output):
    model, meta = load_model(directory)
    capture = Capture(root)
    if capture.seed in meta["consumed_seeds"] or capture.run in model.fit_runs + model.calibration_runs:
        raise ValueError("Evaluation acquisition was used for fitting or calibration")
    # Complete inference before accessing the reference decisions file or test labels.
    scores = model.score_features(feature_panel(capture.values))
    candidate = model.decide(scores, capture.local, capture.allowed)
    verify_streaming(model, capture, candidate)
    predictions = {METHOD: candidate, "STATISTICAL_ONLY": ~capture.allowed & (scores > model.threshold),
                   "LOCAL_ONLY": capture.local, "FULL_HYBRID": capture.full,
                   reference: capture.original_alarm(reference)}
    truth, scenarios = capture.labels()
    if set(np.unique(scenarios[truth])) != set(SCENARIOS):
        raise ValueError("Incomplete cross-scenario label coverage")
    totals = {name: metrics(truth, alarm) for name, alarm in predictions.items()}
    strata = {}
    for scenario in SCENARIOS:
        selected = truth & (scenarios == scenario)
        strata[scenario] = dict(positive=int(selected.sum()), unknown=int(capture.unknown[selected].sum()),
                                availability=int(capture.availability[selected].sum()),
                                tp={name: int(alarm[selected].sum()) for name, alarm in predictions.items()})
    report = dict(seed=capture.seed, run_id=capture.run, model_sha256=meta["model_sha256"],
                  windows=capture.windows, streaming_matches=int(candidate.size),
                  comparison=reference, totals=totals, by_scenario=strata,
                  fpr_pass=totals[METHOD]["fpr"] <= .001,
                  sources={name: sha256(capture.path / name) for name in
                           ("features.csv", "labels.csv", "distributed_predictions.jsonl", "online_predictions.csv")})
    write_json(output, report)
    print(json.dumps({"seed": capture.seed, "metrics": totals, "by_scenario": strata}), flush=True)
    return report


def aggregate(paths, seeds, output):
    reports = [json.loads(Path(p).read_text()) for p in paths]
    if (len(reports) != len(seeds) or len(set(seeds)) != len(seeds)
            or {r["seed"] for r in reports} != set(seeds)
            or len({r["run_id"] for r in reports}) != len(reports)
            or len({r["model_sha256"] for r in reports}) != 1
            or any(r["comparison"] != "TG_V9" or r["windows"] != 3600
                   or r["streaming_matches"] != 32 * r["windows"] for r in reports)):
        raise ValueError("Expected a single frozen model and all distinct 3600-window V9 recordings")
    totals = {}
    for method in reports[0]["totals"]:
        counts = {k: sum(r["totals"][method][k] for r in reports) for k in ("tp", "fp", "fn", "tn")}
        tp, fp, fn, tn = (counts[k] for k in ("tp", "fp", "fn", "tn"))
        totals[method] = dict(**counts, recall=tp/(tp+fn), fpr=fp/(fp+tn),
                              precision=tp/(tp+fp) if tp+fp else None)
    strata = {}
    for scenario in SCENARIOS:
        rows = [r["by_scenario"][scenario] for r in reports]
        strata[scenario] = dict(positive=sum(r["positive"] for r in rows), unknown=sum(r["unknown"] for r in rows),
                                availability=sum(r["availability"] for r in rows),
                                tp={m: sum(r["tp"][m] for r in rows) for m in totals})
    result = dict(model_sha256=reports[0]["model_sha256"], seeds=seeds, totals=totals, by_scenario=strata,
                  per_seed={r["seed"]: r["totals"] for r in reports},
                  streaming_matches=sum(r["streaming_matches"] for r in reports),
                  all_seeds_fpr_within_budget=all(r["fpr_pass"] for r in reports),
                  recall_90_percent=totals[METHOD]["recall"] >= .90,
                  scope="Retrospective paired comparison on independently recorded acquisitions.")
    write_json(output, result)
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    train = commands.add_parser("fit")
    train.add_argument("--train", nargs="+", required=True)
    train.add_argument("--calibration", nargs="+", required=True)
    train.add_argument("--output", required=True)
    test = commands.add_parser("evaluate")
    test.add_argument("model")
    test.add_argument("root")
    test.add_argument("--reference", required=True, choices=("TG_V5", "TG_V9"))
    test.add_argument("--output", required=True)
    pool = commands.add_parser("aggregate")
    pool.add_argument("reports", nargs="+")
    pool.add_argument("--seeds", nargs="+", required=True, type=int)
    pool.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.action == "fit":
        fit(args.train, args.calibration, args.output)
    elif args.action == "evaluate":
        evaluate(args.model, args.root, args.reference, args.output)
    else:
        aggregate(args.reports, args.seeds, args.output)


if __name__ == "__main__":
    main()
