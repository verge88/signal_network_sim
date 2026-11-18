"""Audit and aggregate fresh V8 pair-sensitive prospective structural trials."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from experiments.distributed_5g.v8_design import v8_attack_schedule, v8_members, V8_WINDOWS

METHODS=(
    "TG_V4","TG_V7_GROUP","TG_V7","TG_V8_GROUP","TG_V8",
    "BUDGETED_WITHOUT_MINING","FULL_HYBRID",
)


def metric(truth,pred):
    truth=np.asarray(truth,dtype=bool);pred=np.asarray(pred,dtype=bool)
    tp=int((truth&pred).sum());fp=int((~truth&pred).sum())
    fn=int((truth&~pred).sum());tn=int((~truth&~pred).sum())
    return dict(tp=tp,fp=fp,fn=fn,tn=tn,
        recall=tp/(tp+fn) if tp+fn else None,
        fpr=fp/(fp+tn) if fp+tn else None,
        precision=tp/(tp+fp) if tp+fp else None)


def _k(variant):
    text=str(variant)
    if not text.startswith("k"):
        return None
    return int(text.split("_",1)[0][1:])


def post_episode_tail_fp(frame,method,horizon=5):
    alarms=frame[method].astype(bool).to_numpy()
    attacks=frame.attack.astype(bool).to_numpy()
    windows=frame.window_id.astype(int).to_numpy()
    nfs=frame.nf_id.astype(str).to_numpy()
    count=0
    for nf in sorted(set(nfs)):
        idx=np.flatnonzero(nfs==nf)
        attacked_windows=set(windows[idx[attacks[idx]]])
        for row in idx:
            if attacks[row] or not alarms[row]:
                continue
            w=windows[row]
            if any((w-d) in attacked_windows for d in range(1,horizon+1)):
                count+=1
    return int(count)


def audit(root,output):
    from experiments.live_5g.evaluate import KEYS, load_frozen, score_frame, sha256

    root=Path(root);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    capture=root/"test"
    manifest=json.loads((capture/"manifest.json").read_text())
    if not manifest.get("v8_structure_experiment"):
        raise ValueError("Not a V8 structural acquisition")
    if manifest["windows"]!=V8_WINDOWS or manifest["actors"]!=32:
        raise ValueError("Unexpected V8 structural dimensions")
    expected_schedule=v8_attack_schedule(V8_WINDOWS,manifest["seed"])
    if manifest["schedule"] != [list(x) for x in expected_schedule]:
        raise ValueError("V8 structural schedule mismatch")

    bundle=load_frozen(root/"frozen")
    if bundle["manifest"].get("ml_method")!="TG_V8":
        raise ValueError("Frozen model is not V8")
    v8meta=bundle["manifest"].get("temporal_graph_v8",{})
    if v8meta.get("heldout_v8_test_used_for_selection") is not False:
        raise ValueError("V8 manifest does not prove held-out exclusion")
    branch=v8meta.get("group_branch",{})
    if float(branch.get("pair_decay",-1))!=0.55 or float(branch.get("group_decay_k3_to_k5",-1))!=0.25:
        raise ValueError("Unexpected V8 frozen temporal parameters")

    features=pd.read_csv(capture/"features.csv",dtype={"run_id":str,"nf_id":str},
                         float_precision="round_trip")
    labels=pd.read_csv(capture/"labels.csv",dtype={"run_id":str,"nf_id":str,"episode_id":str})
    labels=labels.fillna({"episode_id":"","variant":"normal","scenario":"N1"})
    if len(features)!=V8_WINDOWS*32 or features.duplicated(KEYS).any() or labels.duplicated(KEYS).any():
        raise ValueError("Unexpected V8 feature/label coverage")
    if features.timestamp.min() <= bundle["manifest"]["created_at"]:
        raise ValueError("V8 test must be acquired after model freeze")

    expected={}
    for entry in expected_schedule:
        for window in range(entry[0],entry[1]):
            for member in v8_members(entry):
                expected[(window,f"slot-{member}")]=(entry[2],entry[5])
    for row in labels.itertuples():
        item=expected.get((int(row.window_id),str(row.nf_id)))
        if bool(row.attack)!=bool(item):
            raise ValueError("V8 attack label mismatch")
        if item and (str(row.scenario),str(row.variant))!=item:
            raise ValueError("V8 scenario/variant mismatch")

    recomputed=score_frame(bundle,features)
    online=pd.read_csv(capture/"online_predictions.csv",
                       dtype={"run_id":str,"nf_id":str},float_precision="round_trip")
    check=recomputed.merge(online,on=KEYS+["method"],validate="one_to_one",
                           suffixes=("","_live"))
    if len(check)!=len(recomputed) or len(check)!=len(online):
        raise ValueError("Online V8 coverage mismatch")
    if not np.array_equal(check.alarm.to_numpy(),check.alarm_live.to_numpy()):
        raise ValueError("Online V8 alarm mismatch")
    if not np.allclose(check.score,check.score_live,rtol=1e-10,atol=1e-12):
        raise ValueError("Online V8 score mismatch")

    online_methods=("TG_V4","TG_V7_GROUP","TG_V7","TG_V8_GROUP","TG_V8")
    pivot=online.loc[online.method.isin(online_methods),
                     KEYS+["method","alarm"]].pivot(
                         index=KEYS,columns="method",values="alarm").reset_index()
    if set(online_methods)-set(pivot):
        raise ValueError("Missing V8 online comparison methods")

    dist=[]
    with (capture/"distributed_predictions.jsonl").open() as stream:
        for line in stream:
            item=json.loads(line)
            if not item["complete"] or len(item["decisions"])!=32:
                raise ValueError("Incomplete hierarchy result")
            for nf,decision in item["decisions"].items():
                dist.append(dict(
                    run_id=manifest["run_id"],window_id=int(item["window"]),nf_id=str(nf),
                    BUDGETED_WITHOUT_MINING=int(bool(decision.get("budgeted_without_ml",
                        decision.get("accum_security",False)))),
                    FULL_HYBRID=int(bool(decision["security"]))))
    dist=pd.DataFrame(dist)
    paired=labels.merge(pivot,on=KEYS,validate="one_to_one").merge(
        dist,on=KEYS,validate="one_to_one")

    totals={}
    episodes=[]
    for method in METHODS:
        totals[method]=metric(paired.attack.astype(bool),paired[method].astype(bool))
        totals[method]["post_episode_tail_fp_5"]=post_episode_tail_fp(paired,method,5)
        for episode,part in paired.loc[paired.attack.astype(bool)].groupby("episode_id",sort=True):
            detected=bool(part[method].astype(bool).any())
            variant=str(part.variant.iloc[0]);members=sorted(part.nf_id.unique().tolist())
            episodes.append(dict(seed=int(manifest["seed"]),episode=str(episode),
                variant=variant,k=_k(variant),members=",".join(members),
                method=method,detected=detected,
                window_recall=float(part[method].astype(bool).mean())))

    episode_frame=pd.DataFrame(episodes)
    by_k=[]
    for (method,k),part in episode_frame.groupby(["method","k"],sort=True):
        by_k.append(dict(method=method,k=int(k),episodes=int(len(part)),
            detected=int(part.detected.sum()),episode_recall=float(part.detected.mean()),
            mean_window_recall=float(part.window_recall.mean())))

    def kmean(method,k):
        part=episode_frame[(episode_frame.method==method)&(episode_frame.k==k)]
        return float(part.window_recall.mean())

    v8=totals["TG_V8"]
    report=dict(
        seed=int(manifest["seed"]),run_id=manifest["run_id"],
        frozen_model_sha256=sha256(root/"frozen"/"model.joblib"),
        test_features_sha256=sha256(capture/"features.csv"),
        totals=totals,by_k=by_k,
        v8_fpr_target=0.001,
        v8_fpr_met=bool(v8["fpr"]<=0.001),
        k2_window_recall_tg_v7=kmean("TG_V7",2),
        k2_window_recall_tg_v8=kmean("TG_V8",2),
        k2_window_recall_delta=float(kmean("TG_V8",2)-kmean("TG_V7",2)),
        prospective=True,
        heldout_after_freeze=True,
        schedule="four randomized k=2 and two each k=3,4,5 within-zone episodes; no adjacency constraint",
    )
    (output/"report.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    episode_frame.to_csv(output/"episodes.csv",index=False)
    pd.DataFrame(by_k).to_csv(output/"by_k.csv",index=False)
    print(json.dumps(report,allow_nan=False),flush=True)


def _wilcoxon(values,alternative):
    values=np.asarray(values,dtype=float)
    if np.allclose(values,0):
        return dict(statistic=0.0,p=1.0)
    test=wilcoxon(values,alternative=alternative,zero_method="wilcox",method="auto")
    return dict(statistic=float(test.statistic),p=float(test.pvalue))


def aggregate(root,output):
    root=Path(root);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    reports=[];episodes=[]
    for path in sorted(root.rglob("report.json")):
        report=json.loads(path.read_text())
        if "seed" in report and "by_k" in report:
            reports.append(report)
            ep=path.parent/"episodes.csv"
            if ep.exists():
                episodes.append(pd.read_csv(ep))
    if not reports:
        raise ValueError("No V8 structural reports")
    if len({r["seed"] for r in reports})!=len(reports):
        raise ValueError("Duplicate V8 seed reports")
    frame=pd.concat(episodes,ignore_index=True)

    pooled={}
    for method in METHODS:
        vals=[r["totals"][method] for r in reports]
        tp=sum(x["tp"] for x in vals);fp=sum(x["fp"] for x in vals)
        fn=sum(x["fn"] for x in vals);tn=sum(x["tn"] for x in vals)
        pooled[method]=dict(tp=tp,fp=fp,fn=fn,tn=tn,
            recall=tp/(tp+fn),fpr=fp/(fp+tn),
            precision=tp/(tp+fp) if tp+fp else None,
            post_episode_tail_fp_5=sum(x["post_episode_tail_fp_5"] for x in vals),
            fpr_target_seeds=sum(x["fpr"]<=0.001 for x in vals),
            seeds=len(vals))

    by_k=[]
    for (method,k),part in frame.groupby(["method","k"],sort=True):
        by_k.append(dict(method=method,k=int(k),episodes=int(len(part)),
            detected=int(part.detected.astype(bool).sum()),
            episode_recall=float(part.detected.astype(bool).mean()),
            mean_window_recall=float(part.window_recall.mean())))

    v8_by_k={row["k"]:row for row in by_k if row["method"]=="TG_V8"}
    k2_delta=np.asarray([r["k2_window_recall_delta"] for r in reports],dtype=float)
    criteria=dict(
        all_seed_fpr=all(r["totals"]["TG_V8"]["fpr"]<=0.001 for r in reports),
        k2_all_24_of_24=(
            2 in v8_by_k and v8_by_k[2]["episodes"]==24 and v8_by_k[2]["detected"]==24),
        k2_mean_window_recall_at_least_0_25=(
            2 in v8_by_k and v8_by_k[2]["mean_window_recall"]>=0.25),
        k3_to_k5_each_at_least_10_of_12=all(
            k in v8_by_k and v8_by_k[k]["episodes"]==12 and v8_by_k[k]["detected"]>=10
            for k in (3,4,5)),
        positive_mean_k2_window_recall_gain_vs_v7=float(k2_delta.mean())>0,
    )
    report=dict(
        seeds=sorted(r["seed"] for r in reports),
        pooled=pooled,by_k=by_k,
        paired_k2_window_recall_v8_minus_v7=dict(
            per_seed=[float(x) for x in k2_delta],mean=float(k2_delta.mean()),
            wilcoxon_greater=_wilcoxon(k2_delta,"greater")),
        preregistered_success_criteria=criteria,
        scientific_success=bool(all(criteria.values())),
        interpretation=(
            "V8 fixes the V7 k=2 limitation only if all six fresh seeds satisfy "
            "FPR <= 0.001, all 24 fresh k=2 episodes are detected, mean k=2 "
            "window recall is >=0.25, k=3..5 each retain >=10/12 detections, "
            "and mean same-run k=2 window recall improves over frozen V7."
        ),
    )
    (output/"aggregate.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    pd.DataFrame(by_k).to_csv(output/"aggregate_by_k.csv",index=False)
    print(json.dumps(report,allow_nan=False),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    sub=parser.add_subparsers(dest="action",required=True)
    one=sub.add_parser("audit");one.add_argument("root");one.add_argument("output")
    agg=sub.add_parser("aggregate");agg.add_argument("root");agg.add_argument("output")
    args=parser.parse_args()
    audit(args.root,args.output) if args.action=="audit" else aggregate(args.root,args.output)
