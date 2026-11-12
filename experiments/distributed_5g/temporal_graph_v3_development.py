"""Temporal-graph V3 development screen.

V3 addresses the two failure modes observed in prospective V2:
1) hard benign synchrony is explicitly up-weighted during classifier fitting;
2) a point score must form a localized multi-node graph motif before its evidence
   is accumulated over time.

All configuration selection is restricted to already-consumed development seeds
89901-90003. Any selected configuration must be frozen before new prospective
seeds are acquired.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from experiments.distributed_5g.temporal_graph_v2 import FEATURES, LocalSynchronyTransformer

KEYS=["run_id","window_id","nf_id"]
ACCUM_MULTIPLIERS=(1.50,1.60,1.75)
ALPHAS=(0.0001,0.00015,0.0002,0.00025)
DECAYS=(0.60,0.65,0.70,0.75)
CLUSTER_ORDERS=(2,)
PRIMARY_FPR_MARGIN=0.0007
SECONDARY_FPR_MARGIN=0.00085


def conformal_threshold(scores,alpha):
    values=np.asarray(scores,dtype=float)
    rank=math.ceil((len(values)+1)*(1-alpha))
    return float(np.sort(values)[rank-1]) if rank<=len(values) else math.inf


def load_phase(root,phase):
    root=Path(root)/phase
    features=pd.read_csv(root/"features.csv",dtype={"run_id":str,"nf_id":str},
                         float_precision="round_trip")
    labels=pd.read_csv(root/"labels.csv",dtype={"run_id":str,"nf_id":str,"episode_id":str})
    labels=labels.fillna({"episode_id":""})
    return features,labels


def graph_frame(features,labels):
    graph=LocalSynchronyTransformer(history=8).transform(features)
    result=pd.DataFrame(graph,columns=FEATURES)
    return pd.concat([
        features[KEYS].reset_index(drop=True),
        result,
        labels[["attack","scenario","episode_id"]].reset_index(drop=True),
    ],axis=1)


def hierarchy_state(root,features):
    root=Path(root)
    zones=json.loads((root/"zone_models.json").read_text())
    zone_for={identity:model for model in zones.values() for identity in model["members"]}
    rows=[]
    with (root/"test"/"distributed_predictions.jsonl").open() as stream:
        for line in stream:
            window=json.loads(line)
            for identity,decision in window["decisions"].items():
                rows.append({
                    "run_id":str(features.run_id.iloc[0]),
                    "window_id":int(window["window"]),
                    "nf_id":identity,
                    "local_security":bool(decision["local_security"]),
                    "recovery":float(decision["recovery_accumulated"]),
                    "zone_threshold":float(zone_for[identity]["threshold"]),
                })
    return pd.DataFrame(rows)


def fit_hard_negative_logistic(frames,seed):
    train=pd.concat(frames,ignore_index=True)
    target=train.scenario.astype(str).eq("S5")
    keep=target | train.attack.eq(0)
    x=train.loc[keep,list(FEATURES)].to_numpy(float)
    y=target.loc[keep].astype(int).to_numpy()
    if not y.any() or y.all():
        raise ValueError("V3 development requires both S5 and normal examples")

    stage1=make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0,class_weight="balanced",max_iter=1000,
                           random_state=seed))
    stage1.fit(x,y)
    probability=stage1.predict_proba(x)[:,1]
    normal_probability=probability[y==0]
    hard_cut=float(np.quantile(normal_probability,0.98))
    weights=np.ones(len(y),dtype=float)
    weights[(y==0)&(probability>=hard_cut)]=8.0

    model=make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0,class_weight="balanced",max_iter=1000,
                           random_state=seed))
    model.fit(x,y,logisticregression__sample_weight=weights)
    return model,dict(hard_negative_quantile=0.98,hard_negative_weight=8.0,
                      hard_negative_cut=hard_cut,
                      training_rows=int(len(y)),s5_rows=int(y.sum()),
                      hard_negative_rows=int(((y==0)&(probability>=hard_cut)).sum()))


def slot_number(nf_id):
    return int(str(nf_id).split("-",1)[1])


def structural_score(frame,probability,cluster_order):
    """Require a localized 2- or 3-node probability cluster inside each zone.

    The top-k minus zone median term is an explicit global-synchrony veto:
    if most of a zone rises together, the localized cluster contrast collapses.
    """
    work=frame[KEYS+list(FEATURES)].copy()
    work["probability"]=np.asarray(probability,dtype=float)
    work["_pos"]=np.arange(len(work))
    out=np.zeros(len(work),dtype=float)
    for (_,window_id),window in work.groupby(["run_id","window_id"],sort=False):
        for zone in range(4):
            part=window[window.nf_id.map(slot_number).floordiv(8).eq(zone)]
            if len(part)!=8:
                raise ValueError(f"Expected 8 NF in zone {zone} window {window_id}")
            p=part.probability.to_numpy(float)
            kth=float(np.sort(p)[-cluster_order])
            median=float(np.median(p))
            cluster_contrast=max(0.0,kth-median)
            local=np.maximum(0.,part.localized_corr_contrast.to_numpy(float))
            triangle=np.maximum(0.,part.triangle_contrast.to_numpy(float))
            motif=np.clip(1.0+local+triangle,1.0,3.0)
            values=np.minimum(p,kth)*cluster_contrast*motif
            out[part._pos.to_numpy(int)]=values
    return out


def accumulated_evidence(frame,score,decay):
    evidence=np.zeros(len(frame),dtype=float)
    state={}
    for pos,row in enumerate(frame.itertuples(index=False)):
        key=str(row.nf_id)
        value=decay*state.get(key,0.0)+float(score[pos])
        state[key]=value
        evidence[pos]=value
    return evidence


def metrics(truth,pred):
    truth=np.asarray(truth,dtype=bool); pred=np.asarray(pred,dtype=bool)
    tp=int((truth&pred).sum()); fp=int((~truth&pred).sum())
    fn=int((truth&~pred).sum()); tn=int((~truth&~pred).sum())
    return {
        "tp":tp,"fp":fp,"fn":fn,"tn":tn,
        "recall":tp/(tp+fn) if tp+fn else None,
        "fpr":fp/(fp+tn) if fp+tn else None,
        "precision":tp/(tp+fp) if tp+fp else None,
    }


def episode_recall(frame,pred,scenario="S5"):
    truth=frame.attack.astype(bool).to_numpy()
    scenario_mask=frame.scenario.astype(str).eq(scenario).to_numpy()
    ids=sorted(set(frame.loc[truth&scenario_mask,"episode_id"].astype(str))-{"","nan"})
    pred=np.asarray(pred,dtype=bool)
    detected=0
    for episode in ids:
        mask=frame.episode_id.astype(str).eq(episode).to_numpy()
        detected+=int(pred[mask].any())
    return {"episodes":len(ids),"detected":detected,
            "recall":detected/len(ids) if ids else None}


def prepare_seed(root):
    test_f,test_y=load_phase(root,"test")
    cal_f,cal_y=load_phase(root,"cal")
    if cal_y.attack.astype(bool).any():
        raise ValueError("Calibration must be normal-only")
    test=graph_frame(test_f,test_y)
    cal=graph_frame(cal_f,cal_y)
    state=hierarchy_state(root,test_f)
    test=test.merge(state,on=KEYS,validate="one_to_one")
    return {"root":Path(root),"test":test,"cal":cal}


def run(roots,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    prepared={int(seed):prepare_seed(root) for seed,root in roots.items()}
    fold_results=[]

    for heldout in sorted(prepared):
        development=[prepared[s]["test"] for s in sorted(prepared) if s!=heldout]
        model,training_meta=fit_hard_negative_logistic(development,heldout)
        test=prepared[heldout]["test"]
        test_probability=model.predict_proba(test[list(FEATURES)].to_numpy(float))[:,1]
        # Conservative multi-run calibration: every available calibration phase
        # is fully normal and independent of the held-out test. The operational
        # threshold is the maximum per-run conformal threshold, which protects
        # against run-to-run normal drift instead of trusting one 600-window cal.
        calibration_probability={
            seed:model.predict_proba(item["cal"][list(FEATURES)].to_numpy(float))[:,1]
            for seed,item in prepared.items()
        }
        truth=test.attack.astype(bool).to_numpy()
        original=(test.local_security | (test.recovery>test.zone_threshold)).to_numpy()
        original_metric=metrics(truth,original)
        original_s5=episode_recall(test,original)

        reports=[]
        for order in CLUSTER_ORDERS:
            test_structural=structural_score(test,test_probability,order)
            calibration_structural={
                seed:structural_score(prepared[seed]["cal"],probability,order)
                for seed,probability in calibration_probability.items()
            }
            for decay in DECAYS:
                test_evidence=accumulated_evidence(test,test_structural,decay)
                calibration_evidence={
                    seed:accumulated_evidence(prepared[seed]["cal"],score,decay)
                    for seed,score in calibration_structural.items()
                }
                for alpha in ALPHAS:
                    per_run_threshold={
                        seed:conformal_threshold(evidence,alpha)
                        for seed,evidence in calibration_evidence.items()
                    }
                    threshold=max(per_run_threshold.values())
                    graph_alarm=(test_evidence>threshold if math.isfinite(threshold)
                                 else np.zeros(len(test),dtype=bool))
                    calibration_alarm_rates={
                        seed:(float((evidence>threshold).mean())
                              if math.isfinite(threshold) else 0.0)
                        for seed,evidence in calibration_evidence.items()
                    }
                    for multiplier in ACCUM_MULTIPLIERS:
                        budgeted=(test.local_security |
                                  (test.recovery>multiplier*test.zone_threshold)).to_numpy()
                        full=budgeted|graph_alarm
                        total=metrics(truth,full)
                        s5=episode_recall(test,full)
                        reports.append({
                            "cluster_order":order,
                            "decay":decay,
                            "alpha":alpha,
                            "accumulator_multiplier":multiplier,
                            "threshold":threshold if math.isfinite(threshold) else None,
                            "per_run_calibration_thresholds":{
                                str(seed):(value if math.isfinite(value) else None)
                                for seed,value in per_run_threshold.items()},
                            "calibration_graph_alarm_rates":{
                                str(seed):value for seed,value in calibration_alarm_rates.items()},
                            "metrics":total,
                            "budgeted_metrics":metrics(truth,budgeted),
                            "graph_metrics":metrics(truth,graph_alarm),
                            "s5_episode":s5,
                            "graph_s5_episode":episode_recall(test,graph_alarm),
                            "tp_delta_vs_original":total["tp"]-original_metric["tp"],
                            "fp_delta_vs_original":total["fp"]-original_metric["fp"],
                            "s5_episode_delta_vs_original":s5["detected"]-original_s5["detected"],
                        })
        fold_results.append({
            "heldout_seed":heldout,
            "training_meta":training_meta,
            "original":original_metric,
            "original_s5":original_s5,
            "reports":reports,
        })

    # Aggregate the same configuration across every held-out seed.
    index={}
    for fold in fold_results:
        for report in fold["reports"]:
            key=(report["cluster_order"],report["decay"],report["alpha"],
                 report["accumulator_multiplier"])
            index.setdefault(key,[]).append((fold["heldout_seed"],report))

    candidates=[]
    for key,items in index.items():
        if len(items)!=len(fold_results):
            raise AssertionError("Missing fold for candidate")
        worst_fpr=max(r["metrics"]["fpr"] for _,r in items)
        min_tp_delta=min(r["tp_delta_vs_original"] for _,r in items)
        min_s5_delta=min(r["s5_episode_delta_vs_original"] for _,r in items)
        pooled={
            name:sum(r["metrics"][name] for _,r in items)
            for name in ("tp","fp","fn","tn")
        }
        pooled["recall"]=pooled["tp"]/(pooled["tp"]+pooled["fn"])
        pooled["fpr"]=pooled["fp"]/(pooled["fp"]+pooled["tn"])
        pooled["precision"]=pooled["tp"]/(pooled["tp"]+pooled["fp"])
        s5_detected=sum(r["s5_episode"]["detected"] for _,r in items)
        s5_total=sum(r["s5_episode"]["episodes"] for _,r in items)
        candidates.append({
            "cluster_order":key[0],"decay":key[1],"alpha":key[2],
            "accumulator_multiplier":key[3],
            "worst_seed_fpr":worst_fpr,
            "min_tp_delta":min_tp_delta,
            "min_s5_episode_delta":min_s5_delta,
            "pooled":pooled,
            "s5_episodes_detected":s5_detected,
            "s5_episodes_total":s5_total,
            "primary_eligible":bool(worst_fpr<=PRIMARY_FPR_MARGIN and
                                    min_tp_delta>=0 and min_s5_delta>=1),
            "secondary_eligible":bool(worst_fpr<=SECONDARY_FPR_MARGIN and
                                      min_tp_delta>=0 and min_s5_delta>=1),
            "folds":[{
                "seed":seed,
                "fpr":r["metrics"]["fpr"],
                "tp_delta":r["tp_delta_vs_original"],
                "fp_delta":r["fp_delta_vs_original"],
                "s5_detected":r["s5_episode"]["detected"],
                "s5_total":r["s5_episode"]["episodes"],
            } for seed,r in items],
        })

    def rank(item):
        return (-item["s5_episodes_detected"],
                -item["pooled"]["tp"],
                item["pooled"]["fp"],
                item["worst_seed_fpr"])
    primary=sorted([x for x in candidates if x["primary_eligible"]],key=rank)
    secondary=sorted([x for x in candidates if x["secondary_eligible"]],key=rank)
    result={
        "development_seeds":sorted(prepared),
        "primary_fpr_margin":PRIMARY_FPR_MARGIN,
        "secondary_fpr_margin":SECONDARY_FPR_MARGIN,
        "folds":fold_results,
        "candidate_summary":sorted(candidates,key=rank),
        "selected_primary":primary[0] if primary else None,
        "selected_secondary":secondary[0] if secondary else None,
        "calibration_policy":"Maximum conformal threshold across six fully-normal calibration runs; no test observations used.",
        "scope":"Development only. Any selected V3 configuration requires new prospective seeds.",
    }
    (output/"report.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps({
        "development_seeds":result["development_seeds"],
        "selected_primary":result["selected_primary"],
        "selected_secondary":result["selected_secondary"],
        "eligible_primary_count":len(primary),
        "eligible_secondary_count":len(secondary),
    },allow_nan=False),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for seed in (89901,89902,89903,90001,90002,90003):
        p.add_argument(f"--seed-{seed}",required=True)
    p.add_argument("--output",required=True)
    a=p.parse_args()
    roots={seed:getattr(a,f"seed_{seed}") for seed in (89901,89902,89903,90001,90002,90003)}
    run(roots,a.output)
