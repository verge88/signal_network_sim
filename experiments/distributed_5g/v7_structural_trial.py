"""Audit and aggregate fresh V7 stable-set prospective structural trials."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from experiments.distributed_5g.v7_design import v7_attack_schedule, v7_members, V7_WINDOWS

METHODS=(
    "TG_V4","TG_V6_GROUP","TG_V6","TG_V7_GROUP","TG_V7",
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
    """Count false alarms 1..horizon windows after an attacked NF leaves an episode."""
    alarms=frame[method].astype(bool).to_numpy()
    attacks=frame.attack.astype(bool).to_numpy()
    windows=frame.window_id.astype(int).to_numpy()
    nfs=frame.nf_id.astype(str).to_numpy()
    count=0
    for nf in sorted(set(nfs)):
        idx=np.flatnonzero(nfs==nf)
        pos=idx[attacks[idx]]
        if not len(pos):
            continue
        attacked_windows=set(windows[pos])
        for row in idx:
            if attacks[row] or not alarms[row]:
                continue
            w=windows[row]
            if any((w-d) in attacked_windows for d in range(1,horizon+1)):
                count+=1
    return int(count)


def audit(root,output):
    # Heavy detector dependencies are needed only for acquisition audit, not aggregation.
    from experiments.live_5g.evaluate import KEYS, load_frozen, score_frame, sha256

    root=Path(root);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    capture=root/"test"
    manifest=json.loads((capture/"manifest.json").read_text())
    if not manifest.get("v7_structure_experiment"):
        raise ValueError("Not a V7 structural acquisition")
    if manifest["windows"]!=V7_WINDOWS or manifest["actors"]!=32:
        raise ValueError("Unexpected V7 structural dimensions")
    expected_schedule=v7_attack_schedule(V7_WINDOWS,manifest["seed"])
    if manifest["schedule"] != [list(x) for x in expected_schedule]:
        raise ValueError("V7 structural schedule mismatch")

    bundle=load_frozen(root/"frozen")
    if bundle["manifest"].get("ml_method")!="TG_V7":
        raise ValueError("Frozen model is not V7")
    if bundle["manifest"].get("temporal_graph_v7",{}).get("heldout_v7_test_used_for_selection") is not False:
        raise ValueError("V7 manifest does not prove held-out exclusion")

    features=pd.read_csv(capture/"features.csv",dtype={"run_id":str,"nf_id":str},
                         float_precision="round_trip")
    labels=pd.read_csv(capture/"labels.csv",dtype={"run_id":str,"nf_id":str,"episode_id":str})
    labels=labels.fillna({"episode_id":"","variant":"normal","scenario":"N1"})
    if len(features)!=V7_WINDOWS*32 or features.duplicated(KEYS).any() or labels.duplicated(KEYS).any():
        raise ValueError("Unexpected V7 feature/label coverage")
    if features.timestamp.min() <= bundle["manifest"]["created_at"]:
        raise ValueError("V7 test must be acquired after model freeze")

    expected={}
    for entry in expected_schedule:
        for window in range(entry[0],entry[1]):
            for member in v7_members(entry):
                expected[(window,f"slot-{member}")]=(entry[2],entry[5])
    for row in labels.itertuples():
        item=expected.get((int(row.window_id),str(row.nf_id)))
        if bool(row.attack)!=bool(item):
            raise ValueError("V7 attack label mismatch")
        if item and (str(row.scenario),str(row.variant))!=item:
            raise ValueError("V7 scenario/variant mismatch")

    recomputed=score_frame(bundle,features)
    online=pd.read_csv(capture/"online_predictions.csv",
                       dtype={"run_id":str,"nf_id":str},float_precision="round_trip")
    check=recomputed.merge(online,on=KEYS+["method"],validate="one_to_one",
                           suffixes=("","_live"))
    if len(check)!=len(recomputed) or len(check)!=len(online):
        raise ValueError("Online V7 coverage mismatch")
    if not np.array_equal(check.alarm.to_numpy(),check.alarm_live.to_numpy()):
        raise ValueError("Online V7 alarm mismatch")
    if not np.allclose(check.score,check.score_live,rtol=1e-10,atol=1e-12):
        raise ValueError("Online V7 score mismatch")

    online_methods=("TG_V4","TG_V6_GROUP","TG_V6","TG_V7_GROUP","TG_V7")
    pivot=online.loc[online.method.isin(online_methods),
                     KEYS+["method","alarm"]].pivot(
                         index=KEYS,columns="method",values="alarm").reset_index()
    if set(online_methods)-set(pivot):
        raise ValueError("Missing V7 online comparison methods")

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

    v7=totals["TG_V7"]
    report=dict(
        seed=int(manifest["seed"]),run_id=manifest["run_id"],
        frozen_model_sha256=sha256(root/"frozen"/"model.joblib"),
        test_features_sha256=sha256(capture/"features.csv"),
        totals=totals,by_k=by_k,
        v7_fpr_target=0.001,
        v7_fpr_met=bool(v7["fpr"]<=0.001),
        tg_v7_minus_v4_recall=float(v7["recall"]-totals["TG_V4"]["recall"]),
        tg_v7_minus_v6_recall=float(v7["recall"]-totals["TG_V6"]["recall"]),
        tg_v6_minus_v7_fpr=float(totals["TG_V6"]["fpr"]-v7["fpr"]),
        prospective=True,
        heldout_after_freeze=True,
        schedule="two randomized within-zone episodes for each k=2,3,4,5; no adjacency constraint",
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
        raise ValueError("No V7 structural reports")
    if len({r["seed"] for r in reports})!=len(reports):
        raise ValueError("Duplicate V7 seed reports")
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

    recall_diff=np.asarray([
        r["totals"]["TG_V7"]["recall"]-r["totals"]["TG_V4"]["recall"]
        for r in reports],dtype=float)
    fpr_reduction=np.asarray([
        r["totals"]["TG_V6"]["fpr"]-r["totals"]["TG_V7"]["fpr"]
        for r in reports],dtype=float)

    v7_by_k={row["k"]:row for row in by_k if row["method"]=="TG_V7"}
    criteria=dict(
        all_seed_fpr=all(r["totals"]["TG_V7"]["fpr"]<=0.001 for r in reports),
        each_k_at_least_10_of_12=(
            set(v7_by_k)=={2,3,4,5} and all(
                row["episodes"]>=12 and row["detected"]>=10 for row in v7_by_k.values())),
        pooled_nf_window_recall_at_least_0_50=pooled["TG_V7"]["recall"]>=0.50,
    )
    report=dict(
        seeds=sorted(r["seed"] for r in reports),
        pooled=pooled,by_k=by_k,
        paired_v7_minus_v4_recall=dict(
            per_seed=[float(x) for x in recall_diff],mean=float(recall_diff.mean()),
            wilcoxon_greater=_wilcoxon(recall_diff,"greater")),
        paired_v6_minus_v7_fpr=dict(
            per_seed=[float(x) for x in fpr_reduction],mean=float(fpr_reduction.mean()),
            wilcoxon_greater=_wilcoxon(fpr_reduction,"greater")),
        preregistered_success_criteria=criteria,
        scientific_success=bool(all(criteria.values())),
        interpretation=(
            "V7 success requires FPR <= 0.001 on every fresh seed, >=10/12 detected "
            "episodes separately for k=2,3,4,5, and pooled NF-window recall >=0.50."
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
