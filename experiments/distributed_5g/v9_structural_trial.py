"""Audit and aggregate fresh V9 specialized-fusion prospective trials."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from experiments.distributed_5g.v9_design import v9_attack_schedule,v9_members,V9_WINDOWS

METHODS=("TG_V4","TG_V8","TG_V9_PAIR","TG_V9_K4","TG_V9_K5","TG_V9",
         "BUDGETED_WITHOUT_MINING","FULL_HYBRID")

def metric(truth,pred):
    truth=np.asarray(truth,bool);pred=np.asarray(pred,bool)
    tp=int((truth&pred).sum());fp=int((~truth&pred).sum());fn=int((truth&~pred).sum());tn=int((~truth&~pred).sum())
    return dict(tp=tp,fp=fp,fn=fn,tn=tn,recall=tp/(tp+fn),fpr=fp/(fp+tn),
                precision=tp/(tp+fp) if tp+fp else None)

def _k(v): return int(str(v).split("_",1)[0][1:])

def audit(root,output):
    from experiments.live_5g.evaluate import KEYS,load_frozen,score_frame,sha256
    root=Path(root);output=Path(output);output.mkdir(parents=True,exist_ok=True);capture=root/"test"
    manifest=json.loads((capture/"manifest.json").read_text())
    if not manifest.get("v9_structure_experiment") or manifest["windows"]!=V9_WINDOWS or manifest["actors"]!=32:
        raise ValueError("Not a valid V9 acquisition")
    expected=v9_attack_schedule(V9_WINDOWS,manifest["seed"])
    if manifest["schedule"] != [list(x) for x in expected]: raise ValueError("V9 schedule mismatch")
    bundle=load_frozen(root/"frozen");meta=bundle["manifest"].get("temporal_graph_v9",{})
    if bundle["manifest"].get("ml_method")!="TG_V9" or meta.get("heldout_v9_test_used_for_selection") is not False:
        raise ValueError("Frozen model is not valid V9")
    if abs(float(meta["fusion"]["pair_support"])-.62)>1e-12: raise ValueError("Unexpected pair support")
    if abs(float(meta["pair_branch"]["decay"])-.75)>1e-12: raise ValueError("Unexpected pair decay")
    if abs(float(meta["k4_branch"]["threshold_floor"])-.91)>1e-12: raise ValueError("Unexpected k4 floor")

    features=pd.read_csv(capture/"features.csv",dtype={"run_id":str,"nf_id":str},float_precision="round_trip")
    labels=pd.read_csv(capture/"labels.csv",dtype={"run_id":str,"nf_id":str,"episode_id":str}).fillna(
        {"episode_id":"","variant":"normal","scenario":"N1"})
    if len(features)!=V9_WINDOWS*32 or features.duplicated(KEYS).any() or labels.duplicated(KEYS).any():
        raise ValueError("Unexpected V9 coverage")
    if features.timestamp.min()<=bundle["manifest"]["created_at"]: raise ValueError("V9 test predates freeze")
    expected_labels={}
    for e in expected:
        for w in range(e[0],e[1]):
            for m in v9_members(e): expected_labels[(w,f"slot-{m}")]=(e[2],e[5])
    for row in labels.itertuples():
        item=expected_labels.get((int(row.window_id),str(row.nf_id)))
        if bool(row.attack)!=bool(item): raise ValueError("V9 attack label mismatch")
        if item and (str(row.scenario),str(row.variant))!=item: raise ValueError("V9 scenario mismatch")

    recomputed=score_frame(bundle,features)
    online=pd.read_csv(capture/"online_predictions.csv",dtype={"run_id":str,"nf_id":str},float_precision="round_trip")
    chk=recomputed.merge(online,on=KEYS+["method"],validate="one_to_one",suffixes=("","_live"))
    if len(chk)!=len(recomputed) or len(chk)!=len(online): raise ValueError("Online V9 coverage mismatch")
    if not np.array_equal(chk.alarm.to_numpy(),chk.alarm_live.to_numpy()): raise ValueError("V9 alarm replay mismatch")
    if not np.allclose(chk.score,chk.score_live,rtol=1e-10,atol=1e-12): raise ValueError("V9 score replay mismatch")

    om=("TG_V4","TG_V8","TG_V9_PAIR","TG_V9_K4","TG_V9_K5","TG_V9")
    pivot=online.loc[online.method.isin(om),KEYS+["method","alarm"]].pivot(
        index=KEYS,columns="method",values="alarm").reset_index()
    if set(om)-set(pivot): raise ValueError("Missing V9 online methods")

    dist=[]
    with (capture/"distributed_predictions.jsonl").open() as stream:
        for line in stream:
            item=json.loads(line)
            if not item["complete"] or len(item["decisions"])!=32: raise ValueError("Incomplete hierarchy result")
            for nf,d in item["decisions"].items():
                dist.append(dict(run_id=manifest["run_id"],window_id=int(item["window"]),nf_id=str(nf),
                    BUDGETED_WITHOUT_MINING=int(bool(d.get("budgeted_without_ml",d.get("accum_security",False)))),
                    FULL_HYBRID=int(bool(d["security"]))))
    paired=labels.merge(pivot,on=KEYS,validate="one_to_one").merge(pd.DataFrame(dist),on=KEYS,validate="one_to_one")
    totals={};episodes=[]
    for method in METHODS:
        totals[method]=metric(paired.attack.astype(bool),paired[method].astype(bool))
        for eid,part in paired.loc[paired.attack.astype(bool)].groupby("episode_id",sort=True):
            v=str(part.variant.iloc[0]);episodes.append(dict(seed=int(manifest["seed"]),episode=str(eid),
                variant=v,k=_k(v),members=",".join(sorted(part.nf_id.unique())),
                method=method,detected=bool(part[method].astype(bool).any()),
                window_recall=float(part[method].astype(bool).mean())))
    ep=pd.DataFrame(episodes);by_k=[]
    for (method,k),part in ep.groupby(["method","k"],sort=True):
        by_k.append(dict(method=method,k=int(k),episodes=len(part),detected=int(part.detected.sum()),
                         episode_recall=float(part.detected.mean()),mean_window_recall=float(part.window_recall.mean())))
    def kmean(method,k):
        p=ep[(ep.method==method)&(ep.k==k)];return float(p.window_recall.mean())
    report=dict(seed=int(manifest["seed"]),run_id=manifest["run_id"],
        frozen_model_sha256=sha256(root/"frozen"/"model.joblib"),totals=totals,by_k=by_k,
        v9_fpr_met=bool(totals["TG_V9"]["fpr"]<=.001),
        k2_window_recall_tg_v8=kmean("TG_V8",2),k2_window_recall_tg_v9=kmean("TG_V9",2),
        k2_window_recall_delta=float(kmean("TG_V9",2)-kmean("TG_V8",2)),
        prospective=True,heldout_after_freeze=True)
    (output/"report.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    ep.to_csv(output/"episodes.csv",index=False);pd.DataFrame(by_k).to_csv(output/"by_k.csv",index=False)
    print(json.dumps(report,allow_nan=False),flush=True)

def _wilcoxon(v):
    v=np.asarray(v,float)
    if np.allclose(v,0): return dict(statistic=0.0,p=1.0)
    t=wilcoxon(v,alternative="greater",zero_method="wilcox",method="auto")
    return dict(statistic=float(t.statistic),p=float(t.pvalue))

def aggregate(root,output):
    root=Path(root);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    reports=[];eps=[]
    for p in sorted(root.rglob("report.json")):
        r=json.loads(p.read_text())
        if "seed" in r and "by_k" in r:
            reports.append(r);q=p.parent/"episodes.csv"
            if q.exists():eps.append(pd.read_csv(q))
    if len(reports)!=6 or len({r["seed"] for r in reports})!=6: raise ValueError("Expected six unique V9 reports")
    ep=pd.concat(eps,ignore_index=True);pooled={}
    for method in METHODS:
        vals=[r["totals"][method] for r in reports]
        tp=sum(x["tp"] for x in vals);fp=sum(x["fp"] for x in vals);fn=sum(x["fn"] for x in vals);tn=sum(x["tn"] for x in vals)
        pooled[method]=dict(tp=tp,fp=fp,fn=fn,tn=tn,recall=tp/(tp+fn),fpr=fp/(fp+tn),
            precision=tp/(tp+fp) if tp+fp else None,
            fpr_target_seeds=sum(x["fpr"]<=.001 for x in vals),seeds=6)
    by_k=[]
    for (method,k),part in ep.groupby(["method","k"],sort=True):
        by_k.append(dict(method=method,k=int(k),episodes=len(part),detected=int(part.detected.astype(bool).sum()),
            episode_recall=float(part.detected.astype(bool).mean()),mean_window_recall=float(part.window_recall.mean())))
    v9={x["k"]:x for x in by_k if x["method"]=="TG_V9"}
    diff=np.asarray([r["k2_window_recall_delta"] for r in reports],float)
    criteria=dict(
        all_seed_fpr=all(r["totals"]["TG_V9"]["fpr"]<=.001 for r in reports),
        k2_all_24_of_24=(2 in v9 and v9[2]["episodes"]==24 and v9[2]["detected"]==24),
        k2_mean_window_recall_at_least_0_30=(2 in v9 and v9[2]["mean_window_recall"]>=.30),
        k3_to_k5_each_at_least_10_of_12=all(k in v9 and v9[k]["episodes"]==12 and v9[k]["detected"]>=10 for k in (3,4,5)),
        pooled_recall_at_least_0_50=pooled["TG_V9"]["recall"]>=.50,
        positive_mean_k2_gain_vs_v8=float(diff.mean())>0)
    report=dict(seeds=sorted(r["seed"] for r in reports),pooled=pooled,by_k=by_k,
        paired_k2_window_recall_v9_minus_v8=dict(per_seed=[float(x) for x in diff],mean=float(diff.mean()),
                                                  wilcoxon_greater=_wilcoxon(diff)),
        preregistered_success_criteria=criteria,scientific_success=bool(all(criteria.values())),
        interpretation="V9 fixes the pair limitation only if all six FPR targets, 24/24 k2 detection, k2 mean window recall >=0.30, retained k3..5 detection, pooled recall >=0.50 and positive same-run gain versus V8 all hold.")
    (output/"aggregate.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    pd.DataFrame(by_k).to_csv(output/"aggregate_by_k.csv",index=False);print(json.dumps(report,allow_nan=False),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();s=p.add_subparsers(dest="action",required=True)
    a=s.add_parser("audit");a.add_argument("root");a.add_argument("output")
    g=s.add_parser("aggregate");g.add_argument("root");g.add_argument("output")
    x=p.parse_args();audit(x.root,x.output) if x.action=="audit" else aggregate(x.root,x.output)
