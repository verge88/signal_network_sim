"""Nine-seed development screen for common-mode residual temporal graph V4.

Selection objective is stricter than V3:
- worst-seed system FPR <= 0.07%;
- nonnegative total TP delta versus original WITHOUT_MINING on every fold;
- positive S5 episode delta on every fold;
- the cyclic S5 episode (window range 1800-2699) must be detected on every fold.

Regime/window position is used only for development evaluation and sample weighting.
It is never a runtime feature.
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

from experiments.distributed_5g.temporal_graph_v4 import (
    FEATURES_V4, CommonModeResidualTransformer
)

KEYS=["run_id","window_id","nf_id"]
SEEDS=(89901,89902,89903,90001,90002,90003,90101,90102,90103)
ACCUM_MULTIPLIERS=(1.50,1.60,1.75)
ALPHAS=(0.00005,0.000075,0.0001,0.00015,0.0002)
DECAYS=(0.65,0.70,0.75,0.80)
CLUSTER_ORDERS=(2,3)
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
    graph=CommonModeResidualTransformer(history=8).transform(features)
    result=pd.DataFrame(graph,columns=FEATURES_V4)
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


def fit_v4(frames,seed):
    train=pd.concat(frames,ignore_index=True)
    target=train.scenario.astype(str).eq("S5")
    normal=train.attack.eq(0)
    keep=target|normal
    x=train.loc[keep,list(FEATURES_V4)].to_numpy(float)
    y=target.loc[keep].astype(int).to_numpy()
    cyclic=train.loc[keep,"window_id"].astype(int).between(1800,2699).to_numpy()
    if not y.any() or y.all():
        raise ValueError("V4 development requires S5 and normal examples")

    def new_model():
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0,class_weight="balanced",max_iter=1000,
                               random_state=seed))

    stage1=new_model();stage1.fit(x,y)
    probability=stage1.predict_proba(x)[:,1]
    normal_probability=probability[y==0]
    hard_cut=float(np.quantile(normal_probability,.98))
    hard=(y==0)&(probability>=hard_cut)

    weights=np.ones(len(y),dtype=float)
    weights[hard]=8.0
    weights[(y==1)&cyclic]*=3.0
    weights[(y==0)&cyclic]*=2.0

    model=new_model()
    model.fit(x,y,logisticregression__sample_weight=weights)
    return model,{
        "training_rows":int(len(y)),
        "s5_rows":int(y.sum()),
        "cyclic_s5_rows":int(((y==1)&cyclic).sum()),
        "cyclic_normal_rows":int(((y==0)&cyclic).sum()),
        "hard_negative_rows":int(hard.sum()),
        "hard_negative_cut":hard_cut,
        "cyclic_positive_weight":3.0,
        "cyclic_normal_weight":2.0,
    }


def slot_number(nf_id):
    return int(str(nf_id).split("-",1)[1])


def structural_score(frame,probability,cluster_order):
    work=frame[KEYS+list(FEATURES_V4)].copy()
    work["probability"]=np.asarray(probability,dtype=float)
    work["_pos"]=np.arange(len(work))
    out=np.zeros(len(work),dtype=float)
    for (_,window_id),window in work.groupby(["run_id","window_id"],sort=False):
        for zone in range(4):
            part=window[window.nf_id.map(slot_number).floordiv(8).eq(zone)]
            if len(part)!=8:
                raise ValueError(f"Expected 8 NF in zone {zone}, window {window_id}")
            p=part.probability.to_numpy(float)
            kth=float(np.sort(p)[-cluster_order])
            median=float(np.median(p))
            cluster_contrast=max(0.,kth-median)
            local=np.maximum(0.,part.localized_residual_corr_contrast.to_numpy(float))
            triangle=np.maximum(0.,part.residual_triangle_contrast.to_numpy(float))
            # Zone-common movement explicitly suppresses ambiguous global shifts.
            common=np.maximum(0.,part.zone_common_delta_abs.to_numpy(float))
            veto=1.0/(1.0+0.20*common)
            motif=np.clip(1.+local+triangle,1.,3.)*veto
            out[part._pos.to_numpy(int)]=np.minimum(p,kth)*cluster_contrast*motif
    return out


def accumulated_evidence(frame,score,decay):
    output=np.zeros(len(frame),dtype=float)
    state={}
    for pos,row in enumerate(frame.itertuples(index=False)):
        nf=str(row.nf_id)
        value=decay*state.get(nf,0.)+float(score[pos])
        state[nf]=value;output[pos]=value
    return output


def metrics(truth,pred):
    truth=np.asarray(truth,dtype=bool);pred=np.asarray(pred,dtype=bool)
    tp=int((truth&pred).sum());fp=int((~truth&pred).sum())
    fn=int((truth&~pred).sum());tn=int((~truth&~pred).sum())
    return {
        "tp":tp,"fp":fp,"fn":fn,"tn":tn,
        "recall":tp/(tp+fn) if tp+fn else None,
        "fpr":fp/(fp+tn) if fp+tn else None,
        "precision":tp/(tp+fp) if tp+fp else None,
    }


def episode_recall(frame,pred,scenario="S5",cyclic_only=False):
    attack=frame.attack.astype(bool).to_numpy()
    mask=frame.scenario.astype(str).eq(scenario).to_numpy()
    if cyclic_only:
        mask &= frame.window_id.astype(int).between(1800,2699).to_numpy()
    ids=sorted(set(frame.loc[attack&mask,"episode_id"].astype(str))-{"","nan"})
    pred=np.asarray(pred,dtype=bool)
    detected=0
    for episode in ids:
        episode_mask=frame.episode_id.astype(str).eq(episode).to_numpy()
        detected+=int(pred[episode_mask].any())
    return {"episodes":len(ids),"detected":detected,
            "recall":detected/len(ids) if ids else None}


def prepare_seed(root):
    test_f,test_y=load_phase(root,"test")
    cal_f,cal_y=load_phase(root,"cal")
    if cal_y.attack.astype(bool).any():
        raise ValueError("Calibration must be normal-only")
    test=graph_frame(test_f,test_y)
    cal=graph_frame(cal_f,cal_y)
    test=test.merge(hierarchy_state(root,test_f),on=KEYS,validate="one_to_one")
    return {"root":Path(root),"test":test,"cal":cal}


def run(roots,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    prepared={int(seed):prepare_seed(root) for seed,root in roots.items()}
    fold_results=[]

    for heldout in sorted(prepared):
        development=[prepared[seed]["test"] for seed in sorted(prepared) if seed!=heldout]
        model,training_meta=fit_v4(development,heldout)
        test=prepared[heldout]["test"]
        truth=test.attack.astype(bool).to_numpy()
        original=(test.local_security|(test.recovery>test.zone_threshold)).to_numpy()
        original_metric=metrics(truth,original)
        original_s5=episode_recall(test,original)
        original_cyclic_s5=episode_recall(test,original,cyclic_only=True)

        test_probability=model.predict_proba(test[list(FEATURES_V4)].to_numpy(float))[:,1]
        calibration_probability={
            seed:model.predict_proba(item["cal"][list(FEATURES_V4)].to_numpy(float))[:,1]
            for seed,item in prepared.items()
        }

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
                    per_run={
                        seed:conformal_threshold(evidence,alpha)
                        for seed,evidence in calibration_evidence.items()
                    }
                    threshold=max(per_run.values())
                    graph=(test_evidence>threshold if math.isfinite(threshold)
                           else np.zeros(len(test),dtype=bool))
                    for multiplier in ACCUM_MULTIPLIERS:
                        budgeted=(test.local_security|
                                  (test.recovery>multiplier*test.zone_threshold)).to_numpy()
                        full=budgeted|graph
                        total=metrics(truth,full)
                        s5=episode_recall(test,full)
                        cyclic_s5=episode_recall(test,full,cyclic_only=True)
                        reports.append({
                            "cluster_order":order,"decay":decay,"alpha":alpha,
                            "accumulator_multiplier":multiplier,
                            "threshold":threshold if math.isfinite(threshold) else None,
                            "metrics":total,
                            "budgeted_metrics":metrics(truth,budgeted),
                            "graph_metrics":metrics(truth,graph),
                            "s5_episode":s5,
                            "cyclic_s5_episode":cyclic_s5,
                            "graph_s5_episode":episode_recall(test,graph),
                            "graph_cyclic_s5_episode":episode_recall(test,graph,cyclic_only=True),
                            "tp_delta_vs_original":total["tp"]-original_metric["tp"],
                            "fp_delta_vs_original":total["fp"]-original_metric["fp"],
                            "s5_episode_delta_vs_original":s5["detected"]-original_s5["detected"],
                            "cyclic_s5_episode_delta_vs_original":
                                cyclic_s5["detected"]-original_cyclic_s5["detected"],
                        })
        fold_results.append({
            "heldout_seed":heldout,
            "training_meta":training_meta,
            "original":original_metric,
            "original_s5":original_s5,
            "original_cyclic_s5":original_cyclic_s5,
            "reports":reports,
        })

    index={}
    for fold in fold_results:
        for report in fold["reports"]:
            key=(report["cluster_order"],report["decay"],report["alpha"],
                 report["accumulator_multiplier"])
            index.setdefault(key,[]).append((fold["heldout_seed"],report))

    candidates=[]
    for key,items in index.items():
        worst=max(r["metrics"]["fpr"] for _,r in items)
        min_tp=min(r["tp_delta_vs_original"] for _,r in items)
        min_s5=min(r["s5_episode_delta_vs_original"] for _,r in items)
        cyclic_detected=sum(r["cyclic_s5_episode"]["detected"] for _,r in items)
        cyclic_total=sum(r["cyclic_s5_episode"]["episodes"] for _,r in items)
        s5_detected=sum(r["s5_episode"]["detected"] for _,r in items)
        s5_total=sum(r["s5_episode"]["episodes"] for _,r in items)
        pooled={name:sum(r["metrics"][name] for _,r in items)
                for name in ("tp","fp","fn","tn")}
        pooled["recall"]=pooled["tp"]/(pooled["tp"]+pooled["fn"])
        pooled["fpr"]=pooled["fp"]/(pooled["fp"]+pooled["tn"])
        pooled["precision"]=pooled["tp"]/(pooled["tp"]+pooled["fp"])
        candidates.append({
            "cluster_order":key[0],"decay":key[1],"alpha":key[2],
            "accumulator_multiplier":key[3],
            "worst_seed_fpr":worst,
            "min_tp_delta":min,
            "min_s5_episode_delta":min_s5,
            "cyclic_s5_detected":cyclic_detected,
            "cyclic_s5_total":cyclic_total,
            "s5_detected":s5_detected,"s5_total":s5_total,
            "pooled":pooled,
            "primary_eligible":bool(
                worst<=PRIMARY_FPR_MARGIN and min_tp>=0 and min_s5>=1 and
                cyclic_detected==cyclic_total),
            "secondary_eligible":bool(
                worst<=SECONDARY_FPR_MARGIN and min_tp>=0 and min_s5>=1 and
                cyclic_detected==cyclic_total),
            "folds":[{
                "seed":seed,"fpr":r["metrics"]["fpr"],
                "tp_delta":r["tp_delta_vs_original"],
                "fp_delta":r["fp_delta_vs_original"],
                "s5_detected":r["s5_episode"]["detected"],
                "cyclic_s5_detected":r["cyclic_s5_episode"]["detected"],
            } for seed,r in items],
        })

    def rank(item):
        return (-item["cyclic_s5_detected"],-item["s5_detected"],
                -item["pooled"]["tp"],item["pooled"]["fp"],item["worst_seed_fpr"])
    primary=sorted([x for x in candidates if x["primary_eligible"]],key=rank)
    secondary=sorted([x for x in candidates if x["secondary_eligible"]],key=rank)
    result={
        "development_seeds":sorted(prepared),
        "primary_fpr_margin":PRIMARY_FPR_MARGIN,
        "secondary_fpr_margin":SECONDARY_FPR_MARGIN,
        "selected_primary":primary[0] if primary else None,
        "selected_secondary":secondary[0] if secondary else None,
        "eligible_primary_count":len(primary),
        "eligible_secondary_count":len(secondary),
        "candidate_summary":sorted(candidates,key=rank),
        "folds":fold_results,
        "scope":"Development only; selected V4 requires new prospective seeds.",
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
    for seed in SEEDS:
        p.add_argument(f"--seed-{seed}",required=True)
    p.add_argument("--output",required=True)
    args=p.parse_args()
    roots={seed:getattr(args,f"seed_{seed}") for seed in SEEDS}
    run(roots,args.output)
