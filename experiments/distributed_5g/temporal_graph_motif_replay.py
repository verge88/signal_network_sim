"""Fast development screen for calibrated localized temporal-graph motif scores."""
import argparse, json, math
from pathlib import Path
import numpy as np
import pandas as pd

from experiments.distributed_5g.temporal_graph_v2 import LocalSynchronyTransformer, FEATURES
from experiments.distributed_5g.temporal_graph_v2_replay import load, graph_frame, without_mining, apply_persistence, metrics, episode_recall

KEYS=["run_id","window_id","nf_id"]
ALPHAS=(.001,.002,.005,.01,.02)
GATES=((2,3),(3,5),(4,6),(5,8))


def threshold(scores,alpha):
    values=np.asarray(scores,float)
    rank=math.ceil((len(values)+1)*(1-alpha))
    return float(np.sort(values)[rank-1]) if rank<=len(values) else math.inf


def motif_scores(frame):
    x=frame[list(FEATURES)]
    positive=lambda name: np.maximum(0.,x[name].to_numpy(float))
    top=np.clip((x["top2_corr_min"].to_numpy(float)+1.)/2.,0.,1.)
    tri=np.clip((x["triangle_corr"].to_numpy(float)+1.)/2.,0.,1.)
    contrast=positive("localized_corr_contrast")
    tcontrast=positive("triangle_contrast")
    count=np.clip(x["high_corr_peer_count"].to_numpy(float)/2.,0.,1.)
    similarity=x["localized_delta_similarity"].to_numpy(float)
    return {
      "triangle_product": top*tri*(1.+contrast+tcontrast),
      "localized_contrast": contrast+tcontrast+.25*count,
      "clique_score": top*tri*count*(1.+tcontrast)+.2*similarity,
    }


def run(heldout,output,seed):
    heldout=Path(heldout); output=Path(output); output.mkdir(parents=True,exist_ok=True)
    test_f,test_y=load(heldout,"test"); cal_f,cal_y=load(heldout,"cal")
    if cal_y.attack.astype(bool).any(): raise ValueError("Calibration must be normal-only")
    test=graph_frame(test_f,test_y); cal=graph_frame(cal_f,cal_y)
    test=test.merge(without_mining(heldout,test_f),on=KEYS,validate="one_to_one")
    truth=test.attack.astype(bool).to_numpy(); without=test.WITHOUT_MINING.astype(bool).to_numpy()
    base=metrics(truth,without); base_s5=episode_recall(test,without)
    test_scores=motif_scores(test); cal_scores=motif_scores(cal)
    reports=[]
    for score_name,score in test_scores.items():
      for alpha in ALPHAS:
        cut=threshold(cal_scores[score_name],alpha)
        point=score>cut if np.isfinite(cut) else np.zeros(len(score),bool)
        for hits,width in GATES:
          confirmed=apply_persistence(test,point,hits,width)
          full=without|confirmed
          total=metrics(truth,full); s5=episode_recall(test,full)
          reports.append(dict(score=score_name,alpha=alpha,hits=hits,width=width,
            threshold=cut if np.isfinite(cut) else None,
            calibration_alarm_rate=float((cal_scores[score_name]>cut).mean()) if np.isfinite(cut) else 0.,
            metrics=total,graph_confirmed_metrics=metrics(truth,confirmed),
            s5_episode=s5,graph_s5_episode=episode_recall(test,confirmed),
            additional_tp=int((truth&full&~without).sum()),
            additional_fp=int((~truth&full&~without).sum()),
            additional_s5_episodes=int(s5["detected"]-base_s5["detected"]),
            fpr_target_met=bool(total["fpr"]<=.001)))
    result=dict(heldout_seed=seed,baseline=base,baseline_s5_episode=base_s5,reports=reports,
      scope="Development-only motif screen; selected formula/gate requires new prospective confirmation.")
    (output/"report.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps(result,allow_nan=False),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--heldout",required=True); p.add_argument("--output",required=True); p.add_argument("--seed",type=int,required=True)
    a=p.parse_args(); run(a.heldout,a.output,a.seed)
