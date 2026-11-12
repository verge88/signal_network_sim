"""Strict logistic temporal-graph development sweep for the 0.1% system FPR budget."""
import argparse, json, math
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from experiments.distributed_5g.temporal_graph_v2 import FEATURES
from experiments.distributed_5g.temporal_graph_v2_replay import load, graph_frame, without_mining, apply_persistence, metrics, episode_recall

KEYS=["run_id","window_id","nf_id"]
ALPHAS=(.0001,.00025,.0005,.00075,.001,.0015,.002)
GATES=((4,8),(5,8),(5,10),(6,10),(6,12),(7,12),(8,12),(8,15),(10,15))


def cut(scores,alpha):
    v=np.asarray(scores,float); rank=math.ceil((len(v)+1)*(1-alpha))
    return float(np.sort(v)[rank-1]) if rank<=len(v) else math.inf


def run(heldout,development,output,seed):
    heldout=Path(heldout); development=[Path(x) for x in development]
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    dev=[]
    for root in development:
        f,y=load(root,"test"); g=graph_frame(f,y)
        dev.append(g[(g.scenario=="S5")|g.attack.eq(0)])
    dev=pd.concat(dev,ignore_index=True)
    ydev=dev.scenario.eq("S5").astype(int).to_numpy()
    model=make_pipeline(StandardScaler(),LogisticRegression(
        C=1.0,class_weight="balanced",max_iter=1000,random_state=seed))
    model.fit(dev[list(FEATURES)].to_numpy(float),ydev)

    tf,ty=load(heldout,"test"); cf,cy=load(heldout,"cal")
    if cy.attack.astype(bool).any(): raise ValueError("cal must be normal")
    test=graph_frame(tf,ty); cal=graph_frame(cf,cy)
    test=test.merge(without_mining(heldout,tf),on=KEYS,validate="one_to_one")
    truth=test.attack.astype(bool).to_numpy(); without=test.WITHOUT_MINING.astype(bool).to_numpy()
    base=metrics(truth,without); base_s5=episode_recall(test,without)
    cal_score=model.predict_proba(cal[list(FEATURES)].to_numpy(float))[:,1]
    score=model.predict_proba(test[list(FEATURES)].to_numpy(float))[:,1]
    reports=[]
    for alpha in ALPHAS:
        threshold=cut(cal_score,alpha)
        point=score>threshold if np.isfinite(threshold) else np.zeros(len(score),bool)
        for hits,width in GATES:
            confirmed=apply_persistence(test,point,hits,width)
            full=without|confirmed; m=metrics(truth,full); s5=episode_recall(test,full)
            reports.append(dict(alpha=alpha,hits=hits,width=width,threshold=threshold if np.isfinite(threshold) else None,
                calibration_alarm_rate=float((cal_score>threshold).mean()) if np.isfinite(threshold) else 0.,
                metrics=m,graph_metrics=metrics(truth,confirmed),s5_episode=s5,
                graph_s5_episode=episode_recall(test,confirmed),
                additional_tp=int((truth&full&~without).sum()),
                additional_fp=int((~truth&full&~without).sum()),
                additional_s5_episodes=int(s5["detected"]-base_s5["detected"]),
                fpr_target_met=bool(m["fpr"]<=.001)))
    result=dict(heldout_seed=seed,baseline=base,baseline_s5_episode=base_s5,reports=reports,
                scope="Development-only strict logistic sweep; selected config requires new prospective seeds.")
    (output/"report.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps(result,allow_nan=False),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--heldout",required=True);p.add_argument("--development",nargs="+",required=True);p.add_argument("--output",required=True);p.add_argument("--seed",type=int,required=True)
    a=p.parse_args();run(a.heldout,a.development,a.output,a.seed)
