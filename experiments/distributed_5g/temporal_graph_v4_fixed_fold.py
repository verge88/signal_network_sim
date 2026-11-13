"""One-fold fixed V4 validation for parallel GitHub Actions execution."""
import argparse
import json
import math
from pathlib import Path
import numpy as np

import experiments.distributed_5g.temporal_graph_v4_development as screen

SEEDS=screen.SEEDS


def run_one(roots,heldout,output):
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    prepared={int(seed):screen.prepare_seed(root) for seed,root in roots.items()}
    development=[prepared[s]["test"] for s in sorted(prepared) if s!=heldout]
    model,training_meta=screen.fit_v4(development,heldout)
    test=prepared[heldout]["test"]
    truth=test.attack.astype(bool).to_numpy()
    original=(test.local_security|(test.recovery>test.zone_threshold)).to_numpy()
    original_metric=screen.metrics(truth,original)
    original_s5=screen.episode_recall(test,original)
    original_cyclic=screen.episode_recall(test,original,cyclic_only=True)

    test_probability=model.predict_proba(test[list(screen.FEATURES_V4)].to_numpy(float))[:,1]
    calibration_probability={
        seed:model.predict_proba(item["cal"][list(screen.FEATURES_V4)].to_numpy(float))[:,1]
        for seed,item in prepared.items()
    }
    order=2; decay=.75; alpha=.00015; multiplier=1.50
    test_structural=screen.structural_score(test,test_probability,order)
    calibration_structural={
        seed:screen.structural_score(prepared[seed]["cal"],probability,order)
        for seed,probability in calibration_probability.items()
    }
    test_evidence=screen.accumulated_evidence(test,test_structural,decay)
    calibration_evidence={
        seed:screen.accumulated_evidence(prepared[seed]["cal"],score,decay)
        for seed,score in calibration_structural.items()
    }
    thresholds={seed:screen.conformal_threshold(evidence,alpha)
                for seed,evidence in calibration_evidence.items()}
    threshold=max(thresholds.values())
    graph=(test_evidence>threshold if math.isfinite(threshold)
           else np.zeros(len(test),dtype=bool))
    budgeted=(test.local_security|(test.recovery>multiplier*test.zone_threshold)).to_numpy()
    full=budgeted|graph
    total=screen.metrics(truth,full)
    s5=screen.episode_recall(test,full)
    cyclic=screen.episode_recall(test,full,cyclic_only=True)
    result={
        "heldout_seed":heldout,
        "training_meta":training_meta,
        "config":{"cluster_order":order,"decay":decay,"alpha":alpha,
                  "accumulator_multiplier":multiplier},
        "original":original_metric,
        "original_s5":original_s5,
        "original_cyclic_s5":original_cyclic,
        "full":total,
        "budgeted":screen.metrics(truth,budgeted),
        "graph":screen.metrics(truth,graph),
        "s5":s5,
        "cyclic_s5":cyclic,
        "graph_s5":screen.episode_recall(test,graph),
        "graph_cyclic_s5":screen.episode_recall(test,graph,cyclic_only=True),
        "tp_delta":total["tp"]-original_metric["tp"],
        "fp_delta":total["fp"]-original_metric["fp"],
        "s5_delta":s5["detected"]-original_s5["detected"],
        "cyclic_s5_delta":cyclic["detected"]-original_cyclic["detected"],
        "primary_pass":bool(total["fpr"]<=screen.PRIMARY_FPR_MARGIN and
                            total["tp"]>=original_metric["tp"] and
                            s5["detected"]>original_s5["detected"] and
                            cyclic["detected"]==cyclic["episodes"]),
        "threshold":threshold if math.isfinite(threshold) else None,
    }
    (output/"report.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps(result,allow_nan=False),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--heldout",type=int,required=True,choices=SEEDS)
    for seed in SEEDS:
        p.add_argument(f"--seed-{seed}",required=True)
    p.add_argument("--output",required=True)
    a=p.parse_args()
    roots={seed:getattr(a,f"seed_{seed}") for seed in SEEDS}
    run_one(roots,a.heldout,a.output)
