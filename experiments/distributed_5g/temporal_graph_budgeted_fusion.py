"""Development-only budgeted fusion: hard rules + stricter accumulator + temporal graph."""
import argparse, json, math
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from experiments.distributed_5g.temporal_graph_v2 import FEATURES
from experiments.distributed_5g.temporal_graph_v2_replay import load, graph_frame, apply_persistence, metrics, episode_recall

KEYS=["run_id","window_id","nf_id"]
ALPHAS=(.0005,.001,.0015,.002,.003,.005)
GATES=((3,5),(4,6),(5,8),(6,10),(7,12))
ACCUM_MULTIPLIERS=(1.05,1.10,1.15,1.20,1.25,1.30)


def conformal_cut(scores,alpha):
    v=np.asarray(scores,float); rank=math.ceil((len(v)+1)*(1-alpha))
    return float(np.sort(v)[rank-1]) if rank<=len(v) else math.inf


def hierarchy_state(root,features):
    root=Path(root)
    zones=json.loads((root/"zone_models.json").read_text())
    zone_for={identity:model for model in zones.values() for identity in model["members"]}
    rows=[]
    with (root/"test"/"distributed_predictions.jsonl").open() as stream:
        for line in stream:
            w=json.loads(line)
            for identity,d in w["decisions"].items():
                rows.append(dict(
                    run_id=str(features.run_id.iloc[0]),window_id=int(w["window"]),nf_id=identity,
                    local_security=bool(d["local_security"]),
                    recovery=float(d["recovery_accumulated"]),
                    zone_threshold=float(zone_for[identity]["threshold"])))
    return pd.DataFrame(rows)


def run(heldout,development,output,seed):
    heldout=Path(heldout); development=[Path(x) for x in development]
    output=Path(output); output.mkdir(parents=True,exist_ok=True)

    dev=[]
    for root in development:
        f,y=load(root,"test"); g=graph_frame(f,y)
        dev.append(g[(g.scenario=="S5")|g.attack.eq(0)])
    dev=pd.concat(dev,ignore_index=True); ydev=dev.scenario.eq("S5").astype(int).to_numpy()
    model=make_pipeline(StandardScaler(),LogisticRegression(
        C=1.0,class_weight="balanced",max_iter=1000,random_state=seed))
    model.fit(dev[list(FEATURES)].to_numpy(float),ydev)

    tf,ty=load(heldout,"test"); cf,cy=load(heldout,"cal")
    if cy.attack.astype(bool).any(): raise ValueError("calibration must be normal-only")
    test=graph_frame(tf,ty); cal=graph_frame(cf,cy)
    test=test.merge(hierarchy_state(heldout,tf),on=KEYS,validate="one_to_one")
    truth=test.attack.astype(bool).to_numpy()
    original=(test.local_security | (test.recovery>test.zone_threshold)).to_numpy()
    original_metrics=metrics(truth,original); original_s5=episode_recall(test,original)

    cal_score=model.predict_proba(cal[list(FEATURES)].to_numpy(float))[:,1]
    test_score=model.predict_proba(test[list(FEATURES)].to_numpy(float))[:,1]
    reports=[]
    for mult in ACCUM_MULTIPLIERS:
      accum=(test.local_security | (test.recovery>mult*test.zone_threshold)).to_numpy()
      for alpha in ALPHAS:
        cut=conformal_cut(cal_score,alpha)
        point=test_score>cut if np.isfinite(cut) else np.zeros(len(test_score),bool)
        for hits,width in GATES:
          graph=apply_persistence(test,point,hits,width)
          full=accum|graph; m=metrics(truth,full); s5=episode_recall(test,full)
          reports.append(dict(
            accumulator_multiplier=mult,alpha=alpha,hits=hits,width=width,
            threshold=cut if np.isfinite(cut) else None,
            calibration_alarm_rate=float((cal_score>cut).mean()) if np.isfinite(cut) else 0.,
            metrics=m,accumulator_metrics=metrics(truth,accum),graph_metrics=metrics(truth,graph),
            s5_episode=s5,graph_s5_episode=episode_recall(test,graph),
            tp_delta_vs_original=m["tp"]-original_metrics["tp"],
            fp_delta_vs_original=m["fp"]-original_metrics["fp"],
            s5_episode_delta_vs_original=s5["detected"]-original_s5["detected"],
            fpr_target_met=bool(m["fpr"]<=.001)))
    result=dict(heldout_seed=seed,original=original_metrics,original_s5=original_s5,reports=reports,
      selection_goal="FPR<=0.001, nonnegative total TP delta, positive S5 episode delta; development only.")
    (output/"report.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps(result,allow_nan=False),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--heldout",required=True);p.add_argument("--development",nargs="+",required=True);p.add_argument("--output",required=True);p.add_argument("--seed",type=int,required=True)
    a=p.parse_args();run(a.heldout,a.development,a.output,a.seed)
