"""Frozen V12 confirmed-staircase S9 audit; feature-only streaming precedes labels."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.distributed_5g.v12_staircase import ConfirmedStaircaseDetector
from experiments.distributed_5g.v11_staircase_trial import offline_replay as v11_offline_replay
from experiments.distributed_5g.v9_design import V9_WINDOWS, v9_attack_schedule, v9_members

NF_COUNT = 32
DEVELOPMENT_SEEDS = (91001, 91002, 91003, 91004, 91005, 91006)
PROSPECTIVE_SEEDS = (91101, 91102, 91103, 91104, 91105, 91106)
KEYS = ["run_id", "window_id", "nf_id"]

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def metric(truth, pred):
    truth = np.asarray(truth, bool)
    pred = np.asarray(pred, bool)
    tp = int((truth & pred).sum())
    fp = int((~truth & pred).sum())
    fn = int((truth & ~pred).sum())
    tn = int((~truth & ~pred).sum())
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, recall=tp/(tp+fn),
                fpr=fp/(fp+tn), precision=tp/(tp+fp) if tp+fp else None)

def offline_replay(features):
    """Complete, sequential observed-feature-only inference; no labels or schedule."""
    required = set(KEYS) | {"request_count"}
    if not required.issubset(features):
        raise ValueError(f"Missing feature columns: {required-set(features)}")
    if features.duplicated(KEYS).any() or features.run_id.nunique() != 1:
        raise ValueError("Duplicate features or multiple runs")
    if len(features) != V9_WINDOWS * NF_COUNT:
        raise ValueError("Incomplete 700x32 features")
    ordered = features.sort_values(["window_id", "nf_id"])
    if set(ordered.window_id.astype(int).unique()) != set(range(V9_WINDOWS)):
        raise ValueError("Missing or non-contiguous windows")
    nfs = sorted(ordered.nf_id.astype(str).unique().tolist())
    if set(nfs) != {f"slot-{n}" for n in range(NF_COUNT)}:
        raise ValueError("Unknown NF names")
    # Topology-known zone; NOT labels, attack membership, timing or future data.
    detector = ConfirmedStaircaseDetector({nf: int(nf[5:])//8 for nf in nfs})
    rows = []
    for window, part in ordered.groupby("window_id", sort=True):
        if len(part) != NF_COUNT:
            raise ValueError("Incomplete per-window NF coverage")
        counts = dict(zip(part.nf_id.astype(str), part.request_count))
        alarms = detector.step(int(window), counts)
        for nf, alarm in alarms.items():
            rows.append((str(part.run_id.iloc[0]), int(window), nf, alarm))
    return pd.DataFrame(rows, columns=KEYS+["TG_V12"])

def audit(root, output, phase):
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    capture = root / "test"
    manifest = json.loads((capture/"manifest.json").read_text())
    seed = int(manifest["seed"])
    expected = DEVELOPMENT_SEEDS if phase == "development" else PROSPECTIVE_SEEDS
    if seed not in expected:
        raise ValueError(f"Seed {seed} not preregistered for {phase}")
    if not manifest.get("v10_structure_experiment") or manifest["windows"] != V9_WINDOWS or manifest["actors"] != NF_COUNT:
        raise ValueError("Not an unchanged S9 acquisition")
    schedule = v9_attack_schedule(V9_WINDOWS, seed)
    if manifest["schedule"] != [list(e) for e in schedule]:
        raise ValueError("S9 schedule mismatch")
    frozen = json.loads((root/"frozen"/"manifest.json").read_text())
    if frozen.get("ml_method") != "TG_V10" or frozen.get("temporal_graph_v10",{}).get("heldout_v10_test_used_for_selection") is not False:
        raise ValueError("Invalid frozen V10 paired comparator")

    features_path = capture/"features.csv"
    features = pd.read_csv(features_path, dtype={"run_id":str,"nf_id":str})
    if float(features.timestamp.min()) <= float(frozen["created_at"]):
        raise ValueError("Test acquisition predates frozen V10 baseline")
    # Enforced separation: run both streaming detectors before opening any labels.
    predicted = offline_replay(features)
    v11_predicted = v11_offline_replay(features).rename(columns={"TG_V11":"TG_V11"})
    if not (predicted.merge(v11_predicted,on=KEYS,validate="one_to_one")
            .eval("TG_V12 <= TG_V11").all()):
        raise ValueError("V12 alerts must be a subset of causal V11 alerts")

    labels_path = capture/"labels.csv"
    labels = pd.read_csv(labels_path, dtype={"run_id":str,"nf_id":str,"episode_id":str}).fillna(
        {"episode_id":"","variant":"normal"})
    if len(labels) != len(predicted) or labels.duplicated(KEYS).any():
        raise ValueError("Bad label coverage")
    expected_labels = {}
    for e in schedule:
        for w in range(e[0],e[1]):
            for nf in v9_members(e):
                expected_labels[(w,f"slot-{nf}")] = e[5]
    for r in labels.itertuples():
        actual = expected_labels.get((int(r.window_id),str(r.nf_id)))
        if bool(r.attack) != bool(actual) or (actual is not None and str(r.variant) != actual):
            raise ValueError("S9 labels mismatch")

    paired = labels.merge(predicted,on=KEYS,validate="one_to_one").merge(
        v11_predicted,on=KEYS,validate="one_to_one")
    online = pd.read_csv(capture/"online_predictions.csv",usecols=KEYS+["method","alarm"],
                         dtype={"run_id":str,"nf_id":str})
    old = online.loc[online.method=="TG_V10",KEYS+["alarm"]].rename(columns={"alarm":"TG_V10"})
    paired = paired.merge(old,on=KEYS,validate="one_to_one")
    if len(paired) != V9_WINDOWS*NF_COUNT:
        raise ValueError("Missing paired online V10 predictions")
    totals = {name:metric(paired.attack,paired[name]) for name in ("TG_V10","TG_V11","TG_V12")}
    episode_rows = []
    for episode, part in paired.loc[paired.attack.astype(bool)].groupby("episode_id"):
        for method in ("TG_V10","TG_V11","TG_V12"):
            episode_rows.append(dict(seed=seed,episode=str(episode),method=method,
                variant=str(part.variant.iloc[0]),
                k=int(str(part.variant.iloc[0]).split("_",1)[0][1:]),
                detected=bool(part[method].astype(bool).any()),
                window_recall=float(part[method].astype(bool).mean())))
    report = dict(seed=seed, phase=phase, source_run_id=str(manifest["run_id"]),
        features_sha256=digest(features_path), labels_sha256=digest(labels_path),
        frozen_manifest_sha256=digest(root/"frozen"/"manifest.json"),
        totals=totals, recall_gain=totals["TG_V12"]["recall"]-totals["TG_V10"]["recall"],
        target_met=totals["TG_V12"]["recall"]>=.90 and totals["TG_V12"]["fpr"]<=.001,
        causal_no_lookahead=True, scenario_specific=True,
        caveat="V12 suppresses unconfirmed third-window alerts; S9-specific, not general 5G")
    (output/"report.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    pd.DataFrame(episode_rows).to_csv(output/"episodes.csv",index=False)
    print(json.dumps(report,allow_nan=False),flush=True)

def aggregate(root, output, phase):
    root,output=Path(root),Path(output)
    output.mkdir(parents=True,exist_ok=True)
    expected = set(DEVELOPMENT_SEEDS if phase=="development" else PROSPECTIVE_SEEDS)
    reports = [json.loads(p.read_text()) for p in sorted(root.rglob("report.json"))]
    if len(reports) != len(expected) or {r["seed"] for r in reports} != expected or any(r["phase"]!=phase for r in reports):
        raise ValueError(f"Require exactly {sorted(expected)} reports for {phase}")
    pooled = {}
    for method in ("TG_V10","TG_V11","TG_V12"):
        vals=[r["totals"][method] for r in reports]
        tp,fp,fn,tn=(sum(v[k] for v in vals) for k in ("tp","fp","fn","tn"))
        pooled[method]=dict(tp=tp,fp=fp,fn=fn,tn=tn,
            recall=tp/(tp+fn), fpr=fp/(fp+tn),
            precision=tp/(tp+fp) if tp+fp else None)
    criteria=dict(pooled_window_recall_ge_0_90=pooled["TG_V12"]["recall"]>=.90,
        all_seed_fpr_le_0_001=all(r["totals"]["TG_V12"]["fpr"]<=.001 for r in reports),
        pooled_gain_vs_same_run_v10=pooled["TG_V12"]["recall"]>pooled["TG_V10"]["recall"],
        pooled_fpr_reduction_vs_same_run_v11=pooled["TG_V12"]["fpr"]<pooled["TG_V11"]["fpr"])
    report=dict(phase=phase,seeds=sorted(expected),pooled=pooled,
        per_seed={str(r["seed"]):r["totals"] for r in reports},
        preregistered_success_criteria=criteria,scientific_success=bool(all(criteria.values())),
        confirmation=phase=="prospective",
        limitations="S9-specific deterministic count waveform; no assertion of general 5G performance")
    (output/"aggregate.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps(report,allow_nan=False),flush=True)

def main():
    p=argparse.ArgumentParser()
    s=p.add_subparsers(dest="action",required=True)
    for action in ("audit","aggregate"):
        q=s.add_parser(action)
        q.add_argument("root",type=Path)
        q.add_argument("output",type=Path)
        q.add_argument("--phase",required=True,choices=("development","prospective"))
    a=p.parse_args()
    (audit if a.action=="audit" else aggregate)(a.root,a.output,a.phase)

if __name__=="__main__":
    main()
