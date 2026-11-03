"""Replay historical Diameter training, freeze it, and evaluate new simulations.

Historical run 40538956475 did not serialize fitted estimators. Replayed models
are explicitly replicas; agreement with archived predictions is audited per model.
No fitting, threshold selection, or model selection uses independent test data.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib
import json
import os
from pathlib import Path
import pickle
import sys

SOURCE_SHA = "32755d6fb47e9df590934b20e5413afa583ab479"
SOURCE_RUN = 40538956475
TEST_SEEDS = (1001, 1002, 1003)
TRAIN_SEEDS = tuple(range(42, 52))


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")


def unique_file(root, pattern):
    paths = sorted(Path(root).rglob(pattern))
    if len(paths) != 1:
        raise ValueError(f"expected one {pattern}, found {paths}")
    return paths[0]


def check_test_seed(seed):
    if seed in TRAIN_SEEDS or seed not in TEST_SEEDS:
        raise ValueError(f"test seed {seed} is outside the preregistered independent set")


def validate_features(columns):
    forbidden = ("gt_", "true_", "label", "attack_", "is_anomaly", "split_group")
    leaks = [c for c in columns if c.startswith(forbidden)]
    if leaks:
        raise ValueError(f"ground-truth features: {leaks}")


def shift_scenarios(scenarios, mapping, offset):
    return [dataclasses.replace(s, target_nodes=[mapping.get(n, n) for n in s.target_nodes],
                                start_interval=s.start_interval + offset,
                                end_interval=s.end_interval + offset)
            for s in scenarios]


def instrument_export(source):
    """Add serialization only; keep the historical fit/score operations unchanged."""
    patches = [
        ('    ms = [c for c in pipe.feature_cols if c in mod.MS_FEATURES]',
         '    replay_ablations = {}\n    ms = [c for c in pipe.feature_cols if c in mod.MS_FEATURES]'),
        ('        m.fit(X_train[:, idx], y_train)',
         '        m.fit(X_train[:, idx], y_train)\n        replay_ablations[tag] = (m, idx)'),
        ('    # ── сохранение ──\n    df_res = pd.DataFrame(rows)',
         '    # ── сохранение ──\n    df_res = pd.DataFrame(rows)\n'
         '    frozen = dict(pipe=pipe, iso=iso, rf=rf, gb=gb, casc=casc, meta=meta,\n'
         '                  keys=keys, base_val=base_val, ablations=replay_ablations,\n'
         '                  metrics=rows, ae=locals().get("ae_tr"), vae=locals().get("vae_tr"))\n'
         '    with open(os.path.join(out, "frozen_models.pkl"), "wb") as replay_file:\n'
         '        pickle.dump(frozen, replay_file)')]
    for before, after in patches:
        if source.count(before) != 1:
            raise ValueError(f"historical source anchor changed: {before}")
        source = source.replace(before, after)
    return source


def replay(original, seed, out, neural=False):
    import numpy as np
    import pandas as pd

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    data = unique_file(original, f"diameter_open5gs_calibrated_dataset_seed{seed}.csv")
    historical = unique_file(original, f"artifacts_seed{seed}.pkl")
    manifest = unique_file(original, f"manifest_seed{seed}.json")
    provenance = json.loads(manifest.read_text(encoding="utf-8"))
    if provenance["seed"] != seed or provenance["operating_point"]["test_labels_used_for_threshold"]:
        raise ValueError("invalid historical provenance")
    core_path = Path("diameter/ms_multiseed_core.py")
    source = core_path.read_text(encoding="utf-8")
    core_path.write_text(instrument_export(source), encoding="utf-8")
    core = importlib.import_module("ms_multiseed_core")
    spec = importlib.import_module("run_seeds_diameter_calibrated").DIAMETER_CALIBRATED
    core.run_one_seed(spec, seed, csv=str(data), out=str(out), include_sequential=False,
                      include_svm=False, include_neural=neural)
    with open(historical, "rb") as stream:
        old = pickle.load(stream)
    with open(out / f"artifacts_seed{seed}.pkl", "rb") as stream:
        new = pickle.load(stream)
    if old["feature_cols"] != new["feature_cols"] or not np.array_equal(old["y_test"], new["y_test"]):
        raise ValueError("historical features/split are not reproducible")
    rows = []
    for name, scores in new["model_scores"].items():
        reference = np.asarray(old["model_scores"][name])
        delta = float(np.max(np.abs(reference - scores)))
        rows.append(dict(model=name, max_absolute_score_difference=delta,
                         historical_predictions_match=bool(np.allclose(reference, scores, atol=1e-8, rtol=1e-7))))
    pd.DataFrame(rows).to_csv(out / "replay_audit.csv", index=False)
    write_json(out / "provenance.json", dict(source_sha=SOURCE_SHA, source_run=SOURCE_RUN,
               historical_data_sha256=digest(data), historical_artifact_sha256=digest(historical),
               historical_weights_available=False, replay_seed=seed, neural=neural,
               interpretation="reconstructed replicas; audit historical prediction agreement per model",
               test_seeds=TEST_SEEDS, threshold_reference="replayed historical validation negatives",
               test_labels_used_for_fit=False, test_labels_used_for_threshold=False))
    # Preserve a small original reference to make the audit reviewable.
    write_json(out / "historical_manifest.json", provenance)
    return out / "frozen_models.pkl"


def generate(seed, shifted, out):
    import numpy as np
    import pandas as pd

    check_test_seed(seed)
    sim = importlib.import_module("diameter_sim_v1")
    spec = importlib.import_module("run_seeds_diameter_calibrated").DIAMETER_CALIBRATED
    cfg_class, sim_class = sim.SimulationConfig, sim.DiameterSimulator
    offset = (137, 271, 389)[TEST_SEEDS.index(seed)] if shifted else 0
    info = dict(test_seed=seed, shifted=shifted, scenario_offset_intervals=offset,
                independent_randomness=True, external_real_traffic=False,
                same_simulator_family=True)

    def config(*args, **kwargs):
        kwargs.update(spec.sim_config_overrides)
        kwargs["seed"] = seed
        return cfg_class(*args, **kwargs)

    def simulator(cfg, attacks, normal_events):
        topology = sim.DiameterNetworkTopology(cfg)
        groups = {}
        for node_id, node in topology.nodes.items():
            groups.setdefault((node.node_type, node.role), []).append(node_id)
        rng = np.random.default_rng(seed + 90000)
        mapping = {}
        if shifted:
            for nodes in groups.values():
                if len(nodes) > 1:
                    # Nonzero cyclic permutation guarantees remapping within type/role.
                    targets = np.roll(nodes, int(rng.integers(1, len(nodes))))
                    mapping.update(zip(nodes, map(int, targets)))
        changed_attacks = shift_scenarios(attacks, mapping, offset)
        changed_normal = shift_scenarios(normal_events, mapping, offset)
        info["node_mapping"] = mapping
        info["attacks"] = [dataclasses.asdict(a) for a in changed_attacks]
        return sim_class(cfg, changed_attacks, changed_normal)

    sim.SimulationConfig, sim.DiameterSimulator = config, simulator
    try:
        frame = sim.main()
    finally:
        sim.SimulationConfig, sim.DiameterSimulator = cfg_class, sim_class
    # Match the historical CSV round trip before inference.
    path = Path(out) / f"test_{seed}_{'shifted' if shifted else 'iid'}.csv"
    frame.to_csv(path, index=False)
    info["dataset_sha256"] = digest(path)
    info["n_rows"] = len(frame)
    write_json(path.with_suffix(".json"), info)
    return pd.read_csv(path), info


def scores_for(frozen, frame):
    import numpy as np
    import ms_multiseed_core as core

    pipe = frozen["pipe"]
    validate_features(pipe.feature_cols)
    validate_features(pipe.compromise_feature_cols)
    missing = set(pipe.feature_cols + pipe.compromise_feature_cols) - set(frame.columns)
    if missing:
        raise ValueError(f"missing measured features: {sorted(missing)}")
    # Only observable columns reach preprocessing or estimators.
    X = pipe.scaler.transform(pipe.imputer.transform(frame[pipe.feature_cols]))
    C = pipe.compromise_scaler.transform(pipe.compromise_imputer.transform(frame[pipe.compromise_feature_cols]))
    scores = {
        "Isolation Forest": -frozen["iso"].score_samples(X),
        "Random Forest": frozen["rf"].predict_proba(X)[:, 1],
        "Gradient Boosting": frozen["gb"].predict_proba(X)[:, 1],
        "Cascade": frozen["casc"].predict_proba(X, C),
        "Cascade-Stage1": frozen["casc"].predict_proba_stage1(X),
        "Cascade-Stage2": frozen["casc"].predict_proba_stage2(C),
    }
    if frozen["ae"] is not None:
        scores["Autoencoder"] = frozen["ae"].reconstruction_error(X)
    if frozen["vae"] is not None:
        scores["VAE"] = frozen["vae"].anomaly_score(X)
    by_key = {"iso": "Isolation Forest", "rf": "Random Forest", "gb": "Gradient Boosting",
              "casc": "Cascade", "ae": "Autoencoder", "vae": "VAE"}
    meta_X = np.column_stack([core.normalize_score01_from_reference(
        frozen["base_val"][key], scores[by_key[key]]) for key in frozen["keys"]])
    scores["Meta-Ensemble"] = frozen["meta"].predict_proba(meta_X)[:, 1]
    for name, (model, indices) in frozen["ablations"].items():
        scores[name] = model.predict_proba(X[:, indices])[:, 1]
    return scores


def fixed_metrics(labels, scores, threshold):
    import numpy as np
    from sklearn.metrics import average_precision_score, roc_auc_score

    labels, scores = np.asarray(labels), np.asarray(scores)
    if not np.isfinite(scores).all() or not np.isfinite(threshold):
        raise ValueError("nonfinite scores or threshold")
    predicted = scores >= threshold
    positive, negative = labels == 1, labels == 0
    return dict(roc_auc=float(roc_auc_score(labels, scores)),
                average_precision=float(average_precision_score(labels, scores)),
                recall=float(predicted[positive].mean()), fpr=float(predicted[negative].mean()),
                threshold=float(threshold), n=int(len(labels)), n_positive=int(positive.sum()),
                n_negative=int(negative.sum()))


def evaluate(bundle_path, seed, out):
    import numpy as np
    import pandas as pd
    import torch
    import ms_multiseed_core as core

    out = Path(out)
    before = digest(bundle_path)
    with open(bundle_path, "rb") as stream:
        frozen = pickle.load(stream)
    audit = pd.read_csv(Path(bundle_path).parent / "replay_audit.csv").set_index("model")
    references = {r["name"]: r for r in frozen["metrics"]}
    rows, attack_rows = [], []
    for test_seed in TEST_SEEDS:
        for shifted in (False, True):
            frame, info = generate(test_seed, shifted, out)
            labels = frame["is_anomaly"].to_numpy(int)
            torch.manual_seed(test_seed + 70000)
            for name, scores in scores_for(frozen, frame).items():
                reference = references[name]
                # Meta labels fitted on validation; use its prespecified 0.5 only.
                threshold = reference["threshold"] if name == "Meta-Ensemble" else reference["val_1pct_fpr_threshold"]
                row = fixed_metrics(labels, scores, threshold)
                pred = (np.asarray(scores) >= threshold).astype(int)
                debounced = core.debounce_consecutive(frame, pred, required=2)
                row.update(model=name, train_seed=seed, test_seed=test_seed,
                           scenario="shifted" if shifted else "iid",
                           historical_predictions_match=bool(audit.loc[name, "historical_predictions_match"]),
                           threshold_reference="fixed_0.5" if name == "Meta-Ensemble" else "historical_training_validation_negatives",
                           debounce2_recall=float(debounced[labels == 1].mean()),
                           debounce2_fpr=float(debounced[labels == 0].mean()),
                           dataset_sha256=info["dataset_sha256"], frozen_sha256=before)
                rows.append(row)
                for attack, group in frame.loc[labels == 1].groupby("attack_type"):
                    idx = group.index.to_numpy()
                    attack_rows.append(dict(model=name, train_seed=seed, test_seed=test_seed,
                                            scenario=row["scenario"], attack=attack, n=len(idx),
                                            recall=float(pred[idx].mean()), debounce2_recall=float(debounced[idx].mean())))
            pd.DataFrame(rows).to_csv(out / "independent_metrics.csv", index=False)
            pd.DataFrame(attack_rows).to_csv(out / "per_attack.csv", index=False)
    if before != digest(bundle_path):
        raise ValueError("frozen artifact changed during independent evaluation")
    write_json(out / "evaluation_contract.json", dict(source_run=SOURCE_RUN, source_sha=SOURCE_SHA,
               frozen_sha256=before, test_seeds=TEST_SEEDS, test_labels_used_for_fit=False,
               test_labels_used_for_threshold=False, test_data_used_for_model_selection=False,
               observations="6 synthetic datasets: 3 new seeds, each with original and shifted/remapped schedules",
               external_real_traffic=False, historical_weights_available=False))
    print(pd.DataFrame(rows).groupby(["model", "scenario"])[["roc_auc", "recall", "fpr"]].mean().to_string())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--original", required=True)
    p.add_argument("--seed", type=int, required=True, choices=TRAIN_SEEDS)
    p.add_argument("--out", required=True)
    p.add_argument("--neural", action="store_true")
    args = p.parse_args()
    sys.path.insert(0, str(Path("diameter").resolve()))
    bundle = replay(args.original, args.seed, args.out, args.neural)
    evaluate(bundle, args.seed, args.out)


if __name__ == "__main__":
    main()
