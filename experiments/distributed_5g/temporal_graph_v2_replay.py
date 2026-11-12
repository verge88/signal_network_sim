"""Cross-seed development screen for invariant temporal-graph S5 detection.

Two historical S5 prospective captures train the classifier. The held-out capture
contributes only its normal calibration split for probability thresholding before
its held-out test is evaluated. This is development-only; any selected
configuration must be frozen and confirmed on new prospective seeds.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from experiments.distributed_5g.temporal_graph_v2 import LocalSynchronyTransformer, FEATURES

KEYS=["run_id","window_id","nf_id"]
ALPHAS=(.001,.002,.005,.01,.02)
GATES=((2,3),(3,5),(4,6),(5,8))


def conformal_threshold(scores,alpha):
    values=np.asarray(scores,dtype=float)
    rank=math.ceil((len(values)+1)*(1-alpha))
    return float(np.sort(values)[rank-1]) if rank<=len(values) else math.inf


def load(root,phase):
    root=Path(root)/phase
    f=pd.read_csv(root/"features.csv",dtype={"run_id":str,"nf_id":str})
    y=pd.read_csv(root/"labels.csv",dtype={"run_id":str,"nf_id":str,"episode_id":str}).fillna({"episode_id":""})
    return f,y


def graph_frame(features,labels):
    transformer=LocalSynchronyTransformer(history=8)
    x=transformer.transform(features)
    g=pd.DataFrame(x,columns=FEATURES)
    return pd.concat([features[KEYS].reset_index(drop=True),g,
                      labels[["attack","scenario","episode_id"]].reset_index(drop=True)],axis=1)


def model_set(seed):
    return {
      "logistic": make_pipeline(StandardScaler(),LogisticRegression(
          C=1.0,class_weight="balanced",max_iter=1000,random_state=seed)),
      "random_forest": RandomForestClassifier(
          n_estimators=500,max_features="sqrt",min_samples_leaf=2,
          class_weight="balanced_subsample",n_jobs=-1,random_state=seed),
      "extra_trees": ExtraTreesClassifier(
          n_estimators=500,max_features="sqrt",min_samples_leaf=2,
          class_weight="balanced_subsample",n_jobs=-1,random_state=seed),
      "hist_gbdt": HistGradientBoostingClassifier(
          learning_rate=.05,max_iter=240,max_leaf_nodes=31,min_samples_leaf=20,
          l2_regularization=1.0,class_weight="balanced",early_stopping=False,
          random_state=seed),
    }


def without_mining(root,features):
    zones=json.loads((Path(root)/"zone_models.json").read_text())
    zone_for={identity:model for model in zones.values() for identity in model["members"]}
    rows=[]
    with (Path(root)/"test"/"distributed_predictions.jsonl").open() as stream:
        for line in stream:
            window=json.loads(line)
            for identity,decision in window["decisions"].items():
                z=float(zone_for[identity]["threshold"])
                recovery=float(decision["recovery_accumulated"])
                rows.append(dict(
                    run_id=str(features.run_id.iloc[0]),window_id=int(window["window"]),nf_id=identity,
                    WITHOUT_MINING=bool(decision["local_security"] or recovery>z)))
    return pd.DataFrame(rows)


def persistent_flags(base,hits,width):
    base=np.asarray(base,dtype=bool)
    out=np.zeros(len(base),dtype=bool)
    history={}
    for i,row in enumerate(base):
        raise RuntimeError("persistent_flags requires frame-aware wrapper")


def apply_persistence(frame,alarm,hits,width):
    result=np.zeros(len(frame),dtype=bool)
    state={}
    for pos,(idx,row) in enumerate(frame.iterrows()):
        key=str(row.nf_id)
        hist=state.setdefault(key,[])
        hist.append(bool(alarm[pos]))
        del hist[:-width]
        result[pos]=sum(hist)>=hits
    return result


def metrics(truth,pred):
    truth=np.asarray(truth,dtype=bool); pred=np.asarray(pred,dtype=bool)
    tp=int((truth&pred).sum()); fp=int((~truth&pred).sum())
    fn=int((truth&~pred).sum()); tn=int((~truth&~pred).sum())
    return dict(tp=tp,fp=fp,fn=fn,tn=tn,
                recall=tp/(tp+fn) if tp+fn else None,
                fpr=fp/(fp+tn) if fp+tn else None,
                precision=tp/(tp+fp) if tp+fp else None)


def episode_recall(frame,pred,scenario="S5"):
    attack=frame.attack.astype(bool).to_numpy()
    scen=frame.scenario.astype(str).to_numpy()==scenario
    episodes=frame.episode_id.astype(str)
    ids=sorted(set(episodes[attack&scen])-{""})
    detected=0
    for episode in ids:
        mask=(episodes==episode).to_numpy()
        detected+=int(np.asarray(pred,dtype=bool)[mask].any())
    return dict(episodes=len(ids),detected=detected,
                recall=detected/len(ids) if ids else None)


def run(heldout,development,output,seed):
    heldout=Path(heldout); development=[Path(x) for x in development]
    output=Path(output); output.mkdir(parents=True,exist_ok=True)

    dev_parts=[]
    for root in development:
        f,y=load(root,"test")
        g=graph_frame(f,y)
        # S5 is the only positive target. Other attack families are excluded,
        # not relabelled as normal.
        dev_parts.append(g[(g.scenario=="S5") | (g.attack.eq(0))])
    dev=pd.concat(dev_parts,ignore_index=True)
    train_y=dev.scenario.eq("S5").astype(int).to_numpy()
    if train_y.sum()==0 or train_y.sum()==len(train_y):
        raise ValueError("Development set must contain S5 and normal examples")

    test_f,test_y=load(heldout,"test")
    cal_f,cal_y=load(heldout,"cal")
    if cal_y.attack.astype(bool).any():
        raise ValueError("Held-out calibration must be normal-only")
    test=graph_frame(test_f,test_y)
    cal=graph_frame(cal_f,cal_y)
    state=without_mining(heldout,test_f)
    test=test.merge(state,on=KEYS,validate="one_to_one")
    truth=test.attack.astype(bool).to_numpy()
    without=test.WITHOUT_MINING.astype(bool).to_numpy()
    baseline=metrics(truth,without)
    baseline_s5=episode_recall(test,without)

    reports=[]
    for name,model in model_set(seed).items():
        model.fit(dev[list(FEATURES)].to_numpy(float),train_y)
        cal_score=model.predict_proba(cal[list(FEATURES)].to_numpy(float))[:,1]
        test_score=model.predict_proba(test[list(FEATURES)].to_numpy(float))[:,1]
        for alpha in ALPHAS:
            threshold=conformal_threshold(cal_score,alpha)
            point=test_score>threshold if np.isfinite(threshold) else np.zeros(len(test_score),dtype=bool)
            for hits,width in GATES:
                confirmed=apply_persistence(test,point,hits,width)
                full=without|confirmed
                total=metrics(truth,full)
                s5=episode_recall(test,full)
                tg_s5=episode_recall(test,confirmed)
                reports.append(dict(
                    model=name,alpha=alpha,hits=hits,width=width,
                    threshold=threshold if np.isfinite(threshold) else None,
                    calibration_alarm_rate=float((cal_score>threshold).mean()) if np.isfinite(threshold) else 0.0,
                    metrics=total,
                    graph_confirmed_metrics=metrics(truth,confirmed),
                    s5_episode=s5,
                    graph_s5_episode=tg_s5,
                    additional_tp=int((truth&full&~without).sum()),
                    additional_fp=int((~truth&full&~without).sum()),
                    additional_s5_episodes=int(s5["detected"]-baseline_s5["detected"]),
                    fpr_target_met=bool(total["fpr"]<=.001),
                ))
    result=dict(
        heldout_seed=seed,
        development_rows=len(dev),
        features=list(FEATURES),
        baseline=baseline,
        baseline_s5_episode=baseline_s5,
        reports=reports,
        scope="Development-only cross-seed screen; final candidate requires new prospective seeds.")
    (output/"report.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps(result,allow_nan=False),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--heldout",required=True)
    p.add_argument("--development",nargs="+",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--seed",type=int,required=True)
    a=p.parse_args()
    run(a.heldout,a.development,a.output,a.seed)
