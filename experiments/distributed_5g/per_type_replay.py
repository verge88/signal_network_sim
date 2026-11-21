"""Train, calibrate and audit the per-type detector on immutable recorded telemetry."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from .cross_scenario_detector import MEASUREMENTS, SLOTS, feature_panel
from .cross_scenario_replay import Capture, load_model as load_previous, sha256, write_json
from .per_type_detector import CONTROL_NF, METHOD, OnlineScorer, PerTypeModel, ProfileModel, missing_live_receipt, shape_panel
from .v9_cross_scenario_audit import SCENARIOS, metrics


CALIBRATION_FPR = .0006
FPR_LIMIT = .001
RECALL_TARGET = .90
PREVIOUS = "CROSS_SCENARIO"


class Observations:
    def __init__(self, root):
        self.capture = cap = Capture(root)
        self.receipts, self.times = [], []
        self.loss = np.zeros_like(cap.local)
        with (cap.path / "control_events.jsonl").open() as controls, (cap.path / "distributed_predictions.jsonl").open() as decisions:
            for w, (line, snapshot) in enumerate(zip(controls, decisions, strict=True)):
                item, snap = json.loads(line), json.loads(snapshot)
                if w >= cap.windows or snap["window"] != w:
                    raise ValueError("Receipt and hierarchy windows disagree")
                # Do not expose source-side reports, configuration ground truth or intervention logs.
                receipt = dict(window_id=item["window_id"], completed=item["completed"],
                               observer_alive=item["observer_alive"],
                               delivered=None if item["delivered"] is None else True)
                at = float(snap["decision_time"])
                self.loss[w, SLOTS.index(CONTROL_NF)] = missing_live_receipt(receipt, w, at)
                self.receipts.append(receipt)
                self.times.append(at)
        if len(self.receipts) != cap.windows:
            raise ValueError("Incomplete measured control-receipt stream")

    def annotated(self):
        truth, scenarios = self.capture.labels()
        return np.where(truth, scenarios, "N1")


def thresholds(scores, normal):
    rates = np.linspace(0, .0012, 25)
    cuts = [np.quantile(s[m], 1 - rates, method="higher") for s, m in zip(scores, normal, strict=True)]
    boundary = np.max(cuts, axis=0)
    return np.unique(np.r_[1., boundary, np.nextafter(boundary, -np.inf)])


def alarm_grid(cuts, scores, fixed, labels):
    """Exact alarm counts for each triple of strict thresholds, including tied scores."""
    shape = tuple(len(c) for c in cuts)
    bins = [np.searchsorted(c, s, side="left") for c, s in zip(cuts, scores, strict=True)]
    flat = np.ravel_multi_index(tuple(bins), shape)
    for name in ("N1", *SCENARIOS):
        selected = labels == name
        n = int(selected.sum())
        if not n:
            raise ValueError(f"Missing calibration exposure for {name}")
        counts = np.bincount(flat[selected & ~fixed], minlength=int(np.prod(shape))).reshape(shape)
        silent = counts.cumsum(axis=0).cumsum(axis=1).cumsum(axis=2)
        yield name, n, n - silent


def operating_point(populations):
    normal = [(label == "N1") & ~fixed for _, fixed, label in populations]
    cuts = [thresholds([scores[d] for scores, _, _ in populations], normal) for d in range(3)]
    shape = tuple(len(c) for c in cuts)
    worst = np.ones(shape)
    max_fpr = np.zeros(shape)
    hits = {s: np.zeros(shape, dtype=int) for s in SCENARIOS}
    support = dict.fromkeys(SCENARIOS, 0)
    for scores, fixed, labels in populations:
        for name, n, alarms in alarm_grid(cuts, scores, fixed, labels):
            if name == "N1":
                np.maximum(max_fpr, alarms / n, out=max_fpr)
            else:
                np.minimum(worst, alarms / n, out=worst)
                hits[name] += alarms
                support[name] += n
    feasible = max_fpr <= CALIBRATION_FPR
    if not feasible.any():
        raise ValueError("Even the fixed evidence exceeds the calibration FPR budget")
    rates = np.stack([hits[s] / support[s] for s in SCENARIOS])
    pooled_min, macro = rates.min(axis=0), rates.mean(axis=0)
    for criterion in (worst, pooled_min, macro, -max_fpr):
        feasible &= criterion == criterion[feasible].max()
    index = np.unravel_index(np.flatnonzero(feasible)[-1], shape)
    selected = tuple(float(c[i]) for c, i in zip(cuts, index, strict=True))
    quality = dict(worst_type_and_seed=float(worst[index]), worst_pooled_type=float(pooled_min[index]),
                   macro_recall=float(macro[index]), max_fpr=float(max_fpr[index]))
    return selected, quality


def choose_cuts(model, calibration):
    panels = [model.raw_panel(obs.capture.values, obs.capture.allowed) for obs in calibration]
    labels = [obs.annotated() for obs in calibration]
    choices = []
    for weight in (0., .65, 1.):
        model.support_weight = weight
        populations = []
        for obs, (raw, single, group, support), label in zip(calibration, panels, labels, strict=True):
            cap = obs.capture
            graph = np.where(cap.allowed, -1., model.corroborated_graph(raw, support))
            populations.append(((graph, single, group), cap.local | obs.loss, label))
        cuts, quality = operating_point(populations)
        rank = (quality["worst_type_and_seed"], quality["worst_pooled_type"], quality["macro_recall"],
                -quality["max_fpr"], -weight)
        choices.append((rank, weight, cuts, quality))
    _, model.support_weight, cuts, quality = max(choices, key=lambda x: x[0])
    return cuts, quality


def fit(train_paths, calibration_paths, output):
    train = [Observations(p) for p in train_paths]
    cal = [Observations(p) for p in calibration_paths]
    captures = [s.capture for s in train + cal]
    if (len({c.seed for c in captures}) != len(captures)
            or len({c.run for c in captures}) != len(captures)):
        raise ValueError("Training and calibration runs must be disjoint")
    profile_sets, matrices, targets, weights = [], [], [], []
    for obs in train:
        c = obs.capture
        labels = obs.annotated()
        profile_sets.append((c.values[:, :, 0], c.allowed, labels))
        usable = ~(c.local | obs.loss | c.allowed) & np.isin(labels, ["N1", "G", "S5"])
        ty = labels[usable]
        matrices.append(shape_panel(c.values)[usable])
        targets.append(ty != "N1")
        weights.append(np.where(ty == "S5", 42., np.where(ty == "G", 32., 1.)))
    profiles = ProfileModel.fit(profile_sets)
    classifier = HistGradientBoostingClassifier(learning_rate=.075, max_iter=190, max_leaf_nodes=23,
                                                min_samples_leaf=18, l2_regularization=2.,
                                                early_stopping=False, random_state=731)
    classifier.fit(np.concatenate(matrices), np.concatenate(targets), sample_weight=np.concatenate(weights))
    del matrices, targets, weights
    model = PerTypeModel(classifier, profiles, 1., 1., tuple(x.capture.run for x in train),
                         tuple(x.capture.run for x in cal))
    (model.graph_cut, model.profile_cut, model.cohort_cut), criterion = choose_cuts(model, cal)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out / "model.joblib", compress=3)
    meta = dict(method=METHOD, model_sha256=sha256(out / "model.joblib"), train_runs=model.fit_runs,
                calibration_runs=model.calibration_runs, consumed_seeds=sorted(c.seed for c in captures),
                graph_cut=model.graph_cut, profile_cut=model.profile_cut, cohort_cut=model.cohort_cut,
                graph_support_weight=model.support_weight,
                repeated_paths=len(profiles.paths), repeated_cohorts=len(profiles.cohorts),
                selection=criterion, calibration_fpr_cap=CALIBRATION_FPR, recall_target=RECALL_TARGET)
    write_json(out / "manifest.json", meta)
    print(json.dumps(meta), flush=True)


def load_model(directory):
    root = Path(directory)
    meta = json.loads((root / "manifest.json").read_text())
    if meta["method"] != METHOD or sha256(root / "model.joblib") != meta["model_sha256"]:
        raise ValueError("Unexpected per-type checkpoint or digest")
    model = joblib.load(root / "model.joblib")
    if (tuple(meta["train_runs"]) != model.fit_runs or tuple(meta["calibration_runs"]) != model.calibration_runs
            or (meta["graph_cut"], meta["profile_cut"], meta.get("cohort_cut", 1.),
                meta.get("graph_support_weight", 0.)) !=
               (model.graph_cut, model.profile_cut, model.cohort_cut, model.support_weight)):
        raise ValueError("Per-type manifest and checkpoint differ")
    return model, meta


def confirm_stream(model, obs, batch):
    scorer, c = OnlineScorer(model), obs.capture
    for w, values in enumerate(c.values):
        rows = [dict(run_id=c.run, window_id=w, nf_id=nf, **dict(zip(MEASUREMENTS, row)))
                for nf, row in zip(SLOTS, values)]
        decisions = {nf: {"local_security": loc, "allowed": grant}
                     for nf, loc, grant in zip(SLOTS, c.local[w], c.allowed[w])}
        online = scorer.score(rows, decisions, obs.receipts[w], obs.times[w])
        if any(online[nf] != b for nf, b in zip(SLOTS, batch[w])):
            raise ValueError(f"Streaming decision differs at {c.run} window {w}")


def evaluate(model_path, previous_path, root, output):
    model, meta = load_model(model_path)
    baseline, old_meta = load_previous(previous_path)
    obs = Observations(root)
    c = obs.capture
    if (c.seed in meta["consumed_seeds"] or c.seed in old_meta["consumed_seeds"]
            or c.run in model.fit_runs + model.calibration_runs
            or c.run in baseline.fit_runs + baseline.calibration_runs):
        raise ValueError("Refusing to evaluate an acquisition used in model selection")
    graph, profile, cohort = model.score_panel(c.values, c.allowed)
    candidate = model.decide(c.local, obs.loss, c.allowed, graph, profile, cohort)
    confirm_stream(model, obs, candidate)
    old = baseline.decide(baseline.score_features(feature_panel(c.values)), c.local, c.allowed)
    alarms = {METHOD: candidate, PREVIOUS: old, "FULL_HYBRID": c.full,
              "RECEIPT_ONLY": obs.loss, "PROFILE_ONLY": profile > model.profile_cut,
              "GRAPH_ONLY": ~c.allowed & (graph > model.graph_cut),
              "COHORT_ONLY": cohort > model.cohort_cut}
    # The only access to held-out labels follows both batch and exact per-window inference.
    truth, label = c.labels()
    if set(np.unique(label[truth])) != set(SCENARIOS):
        raise ValueError("Incomplete evaluation scenario coverage")
    by_type = {}
    for scenario in SCENARIOS:
        subset = truth & (label == scenario)
        by_type[scenario] = dict(n=int(subset.sum()), hits={m:int(a[subset].sum()) for m,a in alarms.items()})
    normal = ~truth
    subgroups = {}
    monitored = np.broadcast_to(np.arange(32) == SLOTS.index(CONTROL_NF), normal.shape)
    for group, mask in {"permitted": c.allowed, "control_nf": monitored, "unknown": c.unknown}.items():
        exposed = normal & mask
        subgroups[group] = dict(n=int(exposed.sum()), fp=int((candidate & exposed).sum()),
                                receipt_fp=int((obs.loss & exposed).sum()))
    out = dict(seed=c.seed, run=c.run, model=meta["model_sha256"], previous_model=old_meta["model_sha256"],
               streaming_matches=int(candidate.size), totals={m:metrics(truth,a) for m,a in alarms.items()},
               by_type=by_type, normal_subgroups=subgroups,
               inputs={name:sha256(c.path/name) for name in ("features.csv","distributed_predictions.jsonl",
                                                              "control_events.jsonl","labels.csv")})
    write_json(output, out)
    print(json.dumps(out), flush=True)


def aggregate(paths, seeds, output):
    rows = [json.loads(Path(p).read_text()) for p in paths]
    for key in ("model", "previous_model"):
        if len({r[key] for r in rows}) != 1:
            raise ValueError("Mismatched frozen model digests")
    if (len(rows) != len(set(seeds)) or len(seeds) != len(set(seeds))
            or {r["seed"] for r in rows} != set(seeds) or len({r["run"] for r in rows}) != len(rows)
            or any(r["streaming_matches"] != 3600 * 32 for r in rows)):
        raise ValueError("Missing or duplicate complete acquisitions")
    by_type = {}
    for scenario in SCENARIOS:
        n = sum(r["by_type"][scenario]["n"] for r in rows)
        hits = {m:sum(r["by_type"][scenario]["hits"][m] for r in rows) for m in rows[0]["totals"]}
        by_type[scenario] = dict(n=n, hits=hits, recall={m:h/n for m,h in hits.items()})
    totals = {}
    for m in rows[0]["totals"]:
        tp, fn, fp, tn = (sum(r["totals"][m][k] for r in rows) for k in ("tp","fn","fp","tn"))
        totals[m] = dict(tp=tp,fn=fn,fp=fp,tn=tn,recall=tp/(tp+fn),fpr=fp/(fp+tn),
                         precision=tp/(tp+fp) if tp+fp else None)
    per_seed = {r["seed"]: dict(fpr=r["totals"][METHOD]["fpr"],
                                recall={s:v["hits"][METHOD]/v["n"] for s,v in r["by_type"].items()})
                for r in rows}
    result = dict(seeds=sorted(seeds), model=rows[0]["model"], previous_model=rows[0]["previous_model"],
                  by_type=by_type, totals=totals, per_seed=per_seed,
                  worst_type=min(v["recall"][METHOD] for v in by_type.values()),
                  worst_type_single_seed=min(min(x["recall"].values()) for x in per_seed.values()),
                  max_seed_fpr=max(x["fpr"] for x in per_seed.values()),
                  streaming_matches=sum(r["streaming_matches"] for r in rows))
    result["passes"] = result["worst_type"] >= RECALL_TARGET and result["max_seed_fpr"] <= FPR_LIMIT
    write_json(output, result)
    print(json.dumps(result), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="action", required=True)
    train = subs.add_parser("fit")
    train.add_argument("--train", nargs="+", required=True)
    train.add_argument("--calibration", nargs="+", required=True)
    train.add_argument("--output", required=True)
    test = subs.add_parser("evaluate")
    test.add_argument("model")
    test.add_argument("previous")
    test.add_argument("capture")
    test.add_argument("output")
    pool = subs.add_parser("aggregate")
    pool.add_argument("reports", nargs="+")
    pool.add_argument("--seeds", nargs="+", required=True, type=int)
    pool.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.action == "fit":
        fit(args.train, args.calibration, args.output)
    elif args.action == "evaluate":
        evaluate(args.model, args.previous, args.capture, args.output)
    else:
        aggregate(args.reports, args.seeds, args.output)
