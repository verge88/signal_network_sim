"""Fail-closed V10 mixed-scenario and compromised-research-NF independent audit.

Labels are opened ONLY after per-window causal re-inference and original online
alarm matching.  The control trust report is independently recomputed from
retained measured report envelopes with the lab-only HMAC key at each original
decision time; never derived from S10 labels.  NF-window security alarms are
kept distinct from availability/UNKNOWN warnings.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.live_5g.evaluate import load_frozen, score_frame
from experiments.distributed_5g.v10_design import (
    ALL_SCENARIOS, V10_COMPROMISE_VARIANTS, V10_WINDOWS,
    v10_attack_schedule, v10_members,
)

KEYS = ["run_id", "window_id", "nf_id"]
METHODS = ("TG_V8", "TG_V9", "TG_V10")
ACTORS = 32


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_jsonlines(path):
    with Path(path).open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def confusion(truth, alarm):
    a=np.asarray(truth,dtype=bool);p=np.asarray(alarm,dtype=bool)
    if len(a)!=len(p) or not len(a):
        raise ValueError("Incomplete truth/prediction coverage")
    tp=int((a & p).sum());fn=int((a & ~p).sum())
    fp=int((~a & p).sum());tn=int((~a & ~p).sum())
    return {
        "tp":tp,"fn":fn,"fp":fp,"tn":tn,
        "recall":tp/(tp+fn) if tp+fn else None,
        "fpr":fp/(fp+tn) if fp+tn else None,
        "precision":tp/(tp+fp) if tp+fp else None,
    }


def audit(root, frozen_root, output):
    root,frozen_root,output=map(Path,(root,frozen_root,output))
    test=root/"test"
    manifest=json.loads((test/"manifest.json").read_text())
    if not (manifest.get("complete") and manifest["phase"]=="test"
            and manifest.get("v10_compromise_experiment") is True
            and manifest["windows"]==V10_WINDOWS and manifest["actors"]==ACTORS):
        raise ValueError("Not a finished, full V10 3600x32 test capture")
    expected_schedule=v10_attack_schedule(V10_WINDOWS,int(manifest["seed"]))
    if manifest["schedule"] != [list(entry) for entry in expected_schedule]:
        raise ValueError("V10 schedule is missing or was modified after preregistration")

    bundle=load_frozen(frozen_root)
    frozen_meta=bundle["manifest"]
    if (frozen_meta.get("ml_method") != "TG_V10" or
            frozen_meta.get("temporal_graph_v10", {}).get("new_training") is not False or
            "v10_multithreat" not in bundle):
        raise ValueError("Frozen TG_V10 derivation or provenance is invalid")
    if (manifest["run_id"] in frozen_meta.get("train_runs", []) or
            manifest["run_id"] in frozen_meta.get("calibration_runs", [])):
        raise ValueError("Test included in frozen training or calibration")

    feature_path=test/"features.csv"
    features=pd.read_csv(feature_path,dtype={"run_id":str,"nf_id":str},
                         float_precision="round_trip")
    if (len(features)!=V10_WINDOWS*ACTORS or features.duplicated(KEYS).any()
            or set(features.run_id)!={str(manifest["run_id"])}
            or set(features.window_id.astype(int))!=set(range(V10_WINDOWS))):
        raise ValueError("Invalid V10 feature identity grid")
    if float(features.timestamp.min()) <= float(frozen_meta["created_at"]):
        raise ValueError("V10 heldout acquisition did not occur after checkpoint freeze")
    expected_slots=[f"slot-{slot}" for slot in range(ACTORS)]
    if not features.window_id.is_monotonic_increasing:
        raise ValueError("V10 observed windows are out of order")

    # Exact online collector invocation: each window of 32 rows, numeric slot
    # order, one score_frame call. No giant batch or lexicographic slot sort.
    replay=[]
    for window, part in features.groupby("window_id", sort=True):
        if int(window)!=len(replay) or part.nf_id.tolist()!=expected_slots:
            raise ValueError("Noncausal/misaligned V10 input window")
        pred=score_frame(bundle,part.reset_index(drop=True))
        matched=pred.loc[pred.method.isin(METHODS),KEYS+["method","alarm"]]
        if len(matched)!=ACTORS*len(METHODS):
            raise ValueError("Missing TG_V8/TG_V9/TG_V10 replay output")
        replay.append(matched)
    replayed=pd.concat(replay,ignore_index=True)
    online_path=test/"online_predictions.csv"
    online=pd.read_csv(online_path,
                       usecols=KEYS+["method","alarm"],
                       dtype={"run_id":str,"nf_id":str,"method":str})
    online=online.loc[online.method.isin(METHODS)]
    if online.duplicated(KEYS+["method"]).any() or len(online)!=len(replayed):
        raise ValueError("Online V10 decision coverage invalid")
    paired=replayed.merge(online,on=KEYS+["method"],validate="one_to_one",
                          suffixes=("_replay","_online"))
    different=paired.loc[paired.alarm_replay.astype(int)!=paired.alarm_online.astype(int)]
    if len(paired)!=len(replayed) or len(different):
        example=different.head(1)[KEYS+["method","alarm_replay","alarm_online"]].to_dict("records")
        raise ValueError("Online/offline replay mismatch "
                         f"counts={different.groupby('method').size().to_dict()} first={example}")

    # Restore the exact independent control witness event for each single-NF
    # slot-4 at original decision time, verify HMAC and provenance from scratch.
    control=list(read_jsonlines(test/"control_events.jsonl"))
    secure=list(read_jsonlines(test/"v10_secure_decisions.jsonl"))
    if (len(control)!=V10_WINDOWS or len(secure)!=V10_WINDOWS
            or [int(x["window_id"]) for x in control]!=list(range(V10_WINDOWS))
            or [int(x["window_id"]) for x in secure]!=list(range(V10_WINDOWS))):
        raise ValueError("Missing/duplicated trusted control windows")
    secret=bytes.fromhex((test/"lab_control_key.hex").read_text().strip())
    traffic=online.loc[(online.method=="TG_V10")&(online.nf_id=="slot-4")]
    by_window=traffic.set_index("window_id")
    from experiments.distributed_5g.v10_model import V10MultiThreatFusion
    guard=V10MultiThreatFusion()
    checked=[]
    for w,(observation,stored) in enumerate(zip(control,secure)):
        if (stored["run_id"]!=manifest["run_id"] or
                stored["nf_id"]!="slot-4" or
                w not in by_window.index):
            raise ValueError(f"Incomplete V10 research NF control at window {w}")
        expected=guard.assess_external(
            bool(by_window.loc[w,"alarm"]),observation,secret,
            float(stored["decision_time"]))
        for field in ("traffic_alarm","control_state","security_alarm",
                      "availability_unknown","trusted_external_scope"):
            if expected[field]!=stored[field]:
                raise ValueError(f"Independent external control mismatch at {w}: {field}")
        checked.append(expected)

    # **Only after** predictions and report authenticity are verified,
    # open the truth, schedule labels, and episode variants for assessment.
    label_path=test/"labels.csv"
    labels=pd.read_csv(label_path,dtype={"run_id":str,"nf_id":str,
                                         "episode_id":str}).fillna({"episode_id":""})
    if len(labels)!=len(features) or labels.duplicated(KEYS).any() or not labels.attack.isin((0,1)).all():
        raise ValueError("Invalid V10 attack label coverage")
    expected_attack={}
    for entry in expected_schedule:
        for w in range(int(entry[0]),int(entry[1])):
            for slot in v10_members(entry):
                key=(w,f"slot-{slot}")
                if key in expected_attack:
                    raise ValueError("Overlapping V10 attacking NF windows")
                expected_attack[key]=(entry[2],entry[5])
    for row in labels.itertuples(index=False):
        tag=expected_attack.get((int(row.window_id),str(row.nf_id)))
        if bool(row.attack)!=bool(tag):
            raise ValueError("Scenario attack windows not supported by actual preregistration")
        if tag and (str(row.scenario),str(row.variant))!=tag:
            raise ValueError("Source S10/legacy attack variant was mislabelled")
    if set(labels.loc[labels.attack.astype(bool),"scenario"])!=ALL_SCENARIOS:
        raise ValueError("Not all original + S10 attack types were actually measured")

    label_keys=labels[KEYS+["attack","scenario","variant","episode_id"]]
    original_online=online.merge(label_keys,on=KEYS,validate="many_to_one")
    if len(original_online)!=len(online):
        raise ValueError("V10 labels and methods have nonidentical grids")
    totals={};scenario_stats={}
    for name in METHODS:
        part=original_online.loc[original_online.method==name]
        totals[name]=confusion(part.attack,part.alarm)
        scenario_stats[name]={}
        for scenario in sorted(ALL_SCENARIOS):
            positive=part.loc[(part.scenario==scenario)&part.attack.astype(bool)]
            scenario_stats[name][scenario]={
                "tp":int(positive.alarm.sum()),
                "positive":int(len(positive)),
                "recall":float(positive.alarm.mean()),
                "episodes":int(positive.episode_id.nunique()),
                "episodes_detected":int(positive.groupby("episode_id").alarm.max().sum()),
            }
    # Combined security policy: for slot-4 additionally trust *only* the
    # independent authenticated configuration report. Every other NF retains
    # its feature-only TG_V10 alarm. Report unknown separately, not as TP.
    fused=original_online.loc[original_online.method=="TG_V10"].copy()
    secure_alarm={w:item["security_alarm"] for w,item in enumerate(checked)}
    four=fused.nf_id.eq("slot-4")
    fused.loc[four,"alarm"]=fused.loc[four,"window_id"].map(secure_alarm).astype(int)
    totals["V10_SECURE"]=confusion(fused.attack,fused.alarm)
    scenario_stats["V10_SECURE"]={}
    for scenario in sorted(ALL_SCENARIOS):
        positive=fused.loc[(fused.scenario==scenario)&fused.attack.astype(bool)]
        scenario_stats["V10_SECURE"][scenario]={
            "tp":int(positive.alarm.sum()),
            "positive":int(len(positive)),
            "recall":float(positive.alarm.mean()),
            "episodes":int(positive.episode_id.nunique()),
            "episodes_detected":int(positive.groupby("episode_id").alarm.max().sum()),
        }
    s10_variants={}
    for variant in V10_COMPROMISE_VARIANTS:
        attack=fused.loc[(fused.scenario=="S10")&fused.attack.astype(bool)
                         & fused.variant.eq(variant)]
        if len(attack)!=30:
            raise ValueError(f"S10 variant missing: {variant}")
        s10_variants[variant]={
            "tp":int(attack.alarm.sum()),"positive":int(len(attack)),
            "recall":float(attack.alarm.mean()),
        }
    report={
        "phase":"prospective_v10_new_914xx", "seed":int(manifest["seed"]),
        "source_run_id":manifest["run_id"],
        "method":"TG_V10 + independent V10_SECURE on research NF",
        "source_checkpoint_sha256":frozen_meta["temporal_graph_v10"]["parent_frozen_model_sha256"],
        "model_sha256":digest(frozen_root/"model.joblib"),
        "features_sha256":digest(feature_path),
        "labels_sha256":digest(label_path),
        "online_predictions_sha256":digest(online_path),
        "control_event_count":len(control),
        "secure_decision_count":len(checked),
        "unknown_research_nf_windows":int(sum(x["availability_unknown"] for x in checked)),
        "all_alarm_replays_equal":True,
        "all_external_hmac_checks_equal":True,
        "total_nf_windows":len(features),
        "attack_nf_windows":int(labels.attack.sum()),
        "normal_nf_windows":int((~labels.attack.astype(bool)).sum()),
        "totals":totals,"by_scenario":scenario_stats,
        "s10_variants_secure":s10_variants,
        "criteria_observed":{
            "v10_fpr_le_0_001":totals["TG_V10"]["fpr"]<=0.001,
            "v10_secure_fpr_le_0_001":totals["V10_SECURE"]["fpr"]<=0.001,
            "v10_secure_recall_ge_0_90":totals["V10_SECURE"]["recall"]>=0.9,
        },
        "limitations":(
            "Only an isolated research NF's configuration and local report delivery are tampered; "
            "no operating-system exploit, native Open5GS takeover, trusted external key compromise, "
            "or operator-network generalization claim. V10_SECURE requires a trusted external witness."
        ),
    }
    output.mkdir(parents=True,exist_ok=True)
    (output/"report.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps(report,allow_nan=False),flush=True)
    return report


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--frozen",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    a=parser.parse_args()
    audit(a.root,a.frozen,a.out)


if __name__=="__main__":
    main()
