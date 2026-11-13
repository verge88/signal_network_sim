"""Development screen for a common-mode residual triple rescue motif.

This module targets the remaining V4 blind spot: the S5 episode embedded in the
legitimate cyclic regime. It does not use the logistic probability. Instead, it
looks for a persistent contiguous three-node residual cluster after subtracting
the zone median. Threshold/decay are selected only on already-consumed evidence
and must be frozen before any new prospective run.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SEEDS=(89901,89902,89903,90001,90002,90003,90101,90102,90103,90201,90202,90203)
DECAYS=(0.50,0.60,0.70,0.80)
THRESHOLDS=(1.0,1.25,1.5,1.75,2.0)
IQR_LIMITS=(1.0,1.25,1.5)
MAX_RESCUE_FPR=0.00015


def slot_number(nf_id):
    return int(str(nf_id).split("-",1)[1])


def load(root):
    root=Path(root)/"test"
    f=pd.read_csv(root/"features.csv",dtype={"run_id":str,"nf_id":str},float_precision="round_trip")
    y=pd.read_csv(root/"labels.csv",dtype={"run_id":str,"nf_id":str,"episode_id":str}).fillna({"episode_id":""})
    return f.merge(y,on=["run_id","window_id","nf_id"],validate="one_to_one")


def triple_scores(frame):
    """Per-node contiguous-triple score plus zone residual IQR."""
    out=np.zeros(len(frame),dtype=float)
    iqr_out=np.zeros(len(frame),dtype=float)
    positions={idx:pos for pos,idx in enumerate(frame.index)}
    for (_,window_id),part in frame.groupby(["run_id","window_id"],sort=False):
        if len(part)!=32:
            raise ValueError(f"Expected 32 rows at window {window_id}")
        for zone in range(4):
            z=part[part.nf_id.map(slot_number).floordiv(8).eq(zone)].copy()
            z["_slot"]=z.nf_id.map(slot_number)%8
            z=z.sort_values("_slot")
            counts=z.request_count.to_numpy(float)
            common=float(np.median(counts))
            residual=counts-common
            q75,q25=np.quantile(residual,[.75,.25])
            zone_iqr=float(q75-q25)
            node_score=np.zeros(8,dtype=float)
            for start in range(6):
                triple=residual[start:start+3]
                rest=np.concatenate([residual[:start],residual[start+3:]])
                # Three adjacent nodes must all sit above the common-mode baseline;
                # subtract the strongest competing residual outside the motif.
                separation=float(np.min(triple)-max(0.0,float(np.max(rest))))
                coherence=1.0/(1.0+float(np.std(triple)))
                score=max(0.0,separation)*coherence
                node_score[start:start+3]=np.maximum(node_score[start:start+3],score)
            for row_idx,score in zip(z.index,node_score):
                pos=positions[row_idx]
                out[pos]=score
                iqr_out[pos]=zone_iqr
    return out,iqr_out


def evidence(frame,score,decay):
    out=np.zeros(len(frame),dtype=float)
    state={}
    for pos,row in enumerate(frame.itertuples(index=False)):
        nf=str(row.nf_id)
        value=decay*state.get(nf,0.0)+float(score[pos])
        state[nf]=value
        out[pos]=value
    return out


def episode_detect(frame,alarm,onset):
    mask=(frame.scenario.astype(str).eq("S5") &
          frame.window_id.astype(int).between(onset,onset+29))
    return bool(np.asarray(alarm,dtype=bool)[mask.to_numpy()].any())


def run(roots,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    prepared={}
    for seed,root in roots.items():
        frame=load(root)
        score,iqr=triple_scores(frame)
        prepared[int(seed)]=(frame,score,iqr)

    candidates=[]
    for iqr_limit in IQR_LIMITS:
      for decay in DECAYS:
        cached={
            seed:evidence(frame,score*(iqr<=iqr_limit),decay)
            for seed,(frame,score,iqr) in prepared.items()
        }
        for threshold in THRESHOLDS:
            folds=[]
            cyclic_detected=0
            all_s5_detected=0
            total_normal=0
            total_fp=0
            for seed,(frame,_,_) in prepared.items():
                alarm=cached[seed]>threshold
                normal=frame.attack.eq(0).to_numpy()
                fp=int((alarm&normal).sum())
                n=int(normal.sum())
                total_fp+=fp;total_normal+=n
                cyclic=episode_detect(frame,alarm,1820)
                cyclic_detected+=int(cyclic)
                detected=sum(int(episode_detect(frame,alarm,onset)) for onset in (20,920,1820,2720))
                all_s5_detected+=detected
                folds.append({
                    "seed":seed,
                    "normal_fp":fp,
                    "normal_rows":n,
                    "rescue_fpr":fp/n,
                    "cyclic_s5_detected":cyclic,
                    "s5_episodes_detected":detected,
                })
            worst=max(x["rescue_fpr"] for x in folds)
            pooled=total_fp/total_normal
            candidates.append({
                "iqr_limit":iqr_limit,"decay":decay,"threshold":threshold,
                "worst_seed_rescue_fpr":worst,
                "pooled_rescue_fpr":pooled,
                "cyclic_s5_detected":cyclic_detected,
                "cyclic_s5_total":len(SEEDS),
                "all_s5_detected":all_s5_detected,
                "all_s5_total":4*len(SEEDS),
                "eligible":bool(worst<=MAX_RESCUE_FPR and cyclic_detected==len(SEEDS)),
                "folds":folds,
            })

    eligible=[x for x in candidates if x["eligible"]]
    eligible.sort(key=lambda x:(-x["all_s5_detected"],x["worst_seed_rescue_fpr"],
                                x["pooled_rescue_fpr"],x["iqr_limit"],-x["threshold"]))
    result={
        "development_seeds":list(SEEDS),
        "max_rescue_fpr":MAX_RESCUE_FPR,
        "selected":eligible[0] if eligible else None,
        "eligible_count":len(eligible),
        "candidates":sorted(candidates,key=lambda x:(-x["cyclic_s5_detected"],
                                                     -x["all_s5_detected"],
                                                     x["worst_seed_rescue_fpr"])),
        "scope":"Development-only rescue motif. Selected parameters require fresh prospective confirmation."
    }
    (output/"report.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps({k:result[k] for k in ("development_seeds","max_rescue_fpr","selected","eligible_count")},
                     allow_nan=False),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for seed in SEEDS:
        p.add_argument(f"--seed-{seed}",required=True)
    p.add_argument("--output",required=True)
    a=p.parse_args()
    roots={seed:getattr(a,f"seed_{seed}") for seed in SEEDS}
    run(roots,a.output)
