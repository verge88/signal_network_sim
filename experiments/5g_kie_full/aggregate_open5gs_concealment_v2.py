"""Aggregate independently-run real Open5GS v1/v2 KIE concealment metrics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from aggregate_open5gs_boundary_audit import wilson_interval

PHASES=("truthful_udm_restart","hidden_udm_restart",
        "truthful_nrf_burst","hidden_nrf_burst")


def pool(root:Path, output_dir:Path, seeds:set[int]) -> dict:
    pieces=[]
    for seed in sorted(seeds):
        paths=list(root.rglob(f"results_open5gs_v2_{seed}/v2_concealment_events.csv"))
        if len(paths)!=1:
            raise ValueError(f"Expected one complete runner {seed}; found {len(paths)}")
        rows=pd.read_csv(paths[0])
        if rows["seed"].nunique()!=1 or int(rows["seed"].iloc[0])!=seed:
            raise ValueError("Runner seed mismatch")
        pieces.append(rows)
    full=pd.concat(pieces,ignore_index=True)
    output_dir.mkdir(parents=True,exist_ok=True)

    phases=[]
    runners=[]
    for phase in PHASES:
        subset=full[(full["audit_phase"]==phase)&(full["eligible_as_confirmed_exposure"])]
        for seed in sorted(seeds):
            g=subset[subset.seed==seed]
            n=len(g)
            runners.append({
                "phase":phase,"seed":seed,"events":n,
                "v1_alarms":int(g.v1_alarm.sum()),
                "v2_alarms":int(g.v2_alarm.sum()),
                "v2_rate":float(g.v2_alarm.mean()) if n else None,
                "v2_unknown":int(
                    (~g["state"].isin(["acknowledged","unacknowledged"])).sum()
                ),
            })
        n=len(subset);v1=int(subset.v1_alarm.sum());v2=int(subset.v2_alarm.sum())
        low,high=wilson_interval(v2,n)
        phases.append({
            "phase":phase,"events":n,"v1_alarms":v1,"v2_alarms":v2,
            "v1_rate":v1/n if n else None,"v2_rate":v2/n if n else None,
            "v2_wilson95_low_descriptive":low,
            "v2_wilson95_high_descriptive":high,
        })

    summary={
        "seeds":sorted(seeds),
        "source":"new independently randomized real Open5GS runners",
        "total_events":int(len(full)),
        "known_phase_events":int(full.audit_phase.isin(PHASES).sum()),
        "unresolved_events":int(full.audit_phase.eq("unresolved").sum()),
        "outside_protocol_events":int((~full.audit_phase.isin(PHASES+("unresolved",))).sum()),
        "original_boundary_unlabeled":int(full.legacy_unlabeled.sum()),
        "alarms_changed_v1_to_v2":int(full.alarm_changed.sum()),
        "phase_metrics":phases,
        "per_runner":runners,
        "limitations":[
            "The change from +3s to +4s came from a single previous-run false accusation.",
            "Consecutive events within one runner share environmental conditions.",
            "Wilson intervals are descriptive only, not a proof of operational FPR.",
            "The signature does not attest the process or independently prove truthful reporting.",
        ],
    }
    full.to_csv(output_dir/"v2_pooled_events.csv",index=False)
    pd.DataFrame(runners).to_csv(output_dir/"v2_per_runner_metrics.csv",index=False)
    pd.DataFrame(phases).to_csv(output_dir/"v2_pooled_phase_metrics.csv",index=False)
    (output_dir/"v2_pooled_summary.json").write_text(
        json.dumps(summary,indent=2,ensure_ascii=False),encoding="utf-8")
    return summary


def main()->None:
    p=argparse.ArgumentParser()
    p.add_argument("--input-root",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--seeds",type=int,nargs="+",required=True)
    args=p.parse_args()
    print(json.dumps(pool(args.input_root,args.output_dir,set(args.seeds)),indent=2))


if __name__=="__main__":
    main()
