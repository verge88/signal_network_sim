"""Audit fresh S10 observations against online decisions before joining truth."""
import json
import sys
import time
from pathlib import Path
import numpy as np
from experiments.integrated_5g.control import combine
from .cross_scenario_detector import SLOTS
from .cross_scenario_replay import aligned, read_csv, sha256, write_json
from .per_type_detector import CONTROL_NF, METHOD
from .per_type_replay import Observations, load_model, confirm_stream
from .s10_repeat_design import SEEDS, VARIANTS, schedule_for
from .v9_cross_scenario_audit import KEYS, metrics


def audit():
    obs = Observations("fresh")
    cap = obs.capture
    model, meta = load_model("frozen")
    manifest = json.loads((cap.path/"manifest.json").read_text())
    start = float(Path("not_before").read_text())
    if not (manifest["complete"] and start <= manifest["started_at"] < manifest["completed_at"] <= time.time()
            and cap.windows == 3600 and cap.seed not in meta["consumed_seeds"]
            and manifest["actor_http2_requests"] > 0 and manifest["observer_http2_requests"] > 0):
        raise ValueError("Incomplete or stale acquisition")
    base = model.decide(cap.local, obs.loss, cap.allowed, *model.score_panel(cap.values,cap.allowed))
    confirm_stream(model, obs, base)
    key = bytes.fromhex((cap.path/"lab_control_key.hex").read_text().strip())
    states, mismatch, modified = [], [], []
    slot = SLOTS.index(CONTROL_NF)
    with (cap.path/"control_events.jsonl").open() as controls, (cap.path/"distributed_predictions.jsonl").open() as predictions:
        for w,(c,p) in enumerate(zip(controls,predictions,strict=True)):
            c,p=json.loads(c),json.loads(p)
            now=p["decision_time"]
            if not (c["window_id"] == w == p["window"] and start <= c["started"] <= c["completed"] <= now):
                raise ValueError("Invalid observation identity or timing")
            if any(p["decisions"][nf][METHOD] != base[w,j] for j,nf in enumerate(SLOTS)):
                raise ValueError("Model differs from online decisions")
            states.append(combine(False,c["delivered"],c["nonce"],c["expected"],c["instance"],key,now)["K"])
            mismatch.append(c["ground_truth"]["config_hash"] != c["expected"])
            modified.append(c["original"] != c["delivered"])
    states=np.array(states)
    fused=base.copy()
    fused[:,slot] |= states=="alarm"
    catalog=np.zeros(cap.windows,int)
    with (cap.path/"events.jsonl").open() as stream:
        for line in stream:
            row=json.loads(line)
            if row.get("endpoint")=="isolated_empty_test_catalog":
                if row["slot"]!=CONTROL_NF or row["status"]!=200:
                    raise ValueError("Invalid catalog receipt")
                catalog[row["window_id"]] += 1
    truth, labels=cap.labels()
    variants=aligned(read_csv(cap.path/"labels.csv",[*KEYS,"variant"]),cap.index).variant.to_numpy().reshape(truth.shape)
    target=truth[:,slot]
    planned=np.zeros(cap.windows,bool)
    results={}
    for lo,hi,_,_,_,name in schedule_for(cap.seed):
        planned[lo:hi]=True
        if not np.all(variants[lo:hi,slot]==name):
            raise ValueError("Variant labels differ from protocol")
        hits=np.flatnonzero(fused[lo:hi,slot])
        changes=sum(modified[lo:hi])
        if bool(changes)!=(name=="report_integrity"):
            raise ValueError("Envelope observations differ from intervention")
        results[name]=dict(n=hi-lo,tp=len(hits),first_offset=int(hits[0]) if len(hits) else None,
                           base_tp=int(base[lo:hi,slot].sum()),unknown=int((states[lo:hi]=="unknown").sum()),
                           external_tp=int((states[lo:hi]=="alarm").sum()),modified_envelopes=changes)
    if (not np.array_equal(planned,target) or not np.array_equal(planned,mismatch)
            or truth.sum()!=90 or set(labels[truth])!={"S10"}
            or not np.array_equal(catalog,variants[:,slot]=="catalog_policy")):
        raise ValueError("Observed events, interventions and truth disagree")
    result=dict(seed=cap.seed,run=cap.run,started=manifest["started_at"],ended=manifest["completed_at"],
                frozen=meta["model_sha256"],source=sha256(cap.path/"features.csv"),controls=sha256(cap.path/"control_events.jsonl"),
                node_windows=int(truth.size),normal_target=int((~target).sum()),target_fp=int(fused[~target,slot].sum()),
                base=metrics(truth,base),fused=metrics(truth,fused),variants=results)
    Path("summary").mkdir(exist_ok=True)
    write_json(Path("summary")/f"{cap.seed}.json",result)
    print(json.dumps(result))


def aggregate(paths):
    rows=sorted([json.loads(Path(p).read_text()) for p in paths],key=lambda r:r["seed"])
    if ([r["seed"] for r in rows]!=list(SEEDS) or len({r["frozen"] for r in rows})!=1
            or any(len({r[k] for r in rows})!=3 for k in ("run","source","controls"))):
        raise ValueError("Missing or duplicate acquisitions")
    totals={k:sum(r["fused"][k] for r in rows) for k in ("tp","fn","fp","tn")}
    combined=dict(totals=totals,variants={v:sum(r["variants"][v]["tp"] for r in rows) for v in VARIANTS},
                  target_fp=sum(r["target_fp"] for r in rows),normal_target=sum(r["normal_target"] for r in rows),runs=rows)
    write_json("combined.json",combined)
    print(json.dumps(combined))


if __name__=="__main__":
    if sys.argv[1:]:
        aggregate(sys.argv[1:])
    else:
        audit()
