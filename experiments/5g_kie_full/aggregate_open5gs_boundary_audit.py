"""Pool already scored Open5GS boundary-audit results across independent runners.

Do not silently count ambiguous events as negative examples. Report per-run
results alongside pooled descriptive rates, with Wilson intervals *only* as
a limited, event-level descriptive summary (not an independence guarantee).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

PHASES = (
    "truthful_udm_restart",
    "hidden_udm_restart",
    "truthful_nrf_burst",
    "hidden_nrf_burst",
)


def wilson_interval(k: int, n: int, z: float=1.959963984540054) -> tuple[float | None,float | None]:
    if n <= 0:
        return None,None
    p=k/n
    denom=1+z*z/n
    center=(p+z*z/(2*n))/denom
    delta=z*((p*(1-p)/n+z*z/(4*n*n))**.5)/denom
    return max(0.0,center-delta),min(1.0,center+delta)


def aggregate(root: Path, output_dir: Path, expected_seeds: set[int]) -> dict:
    docs=[]
    audit_frames=[]
    covers=[]
    for path in sorted(root.rglob("boundary_audit_summary.json")):
        doc=json.loads(path.read_text(encoding="utf-8"))
        seed=int(doc["seed"])
        if seed not in expected_seeds:
            continue
        if any(int(d["seed"])==seed for d in docs):
            raise ValueError(f"duplicate replay output for seed {seed}")
        docs.append(doc)
        frame=pd.read_csv(path.parent/"boundary_audit_events.csv")
        frame["seed"]=seed
        audit_frames.append(frame)
        covers.append(pd.read_csv(path.parent/"boundary_intervention_coverage.csv"))

    observed_seeds={int(d["seed"]) for d in docs}
    if observed_seeds != expected_seeds:
        raise ValueError(f"missing or extra seeds: expected {expected_seeds}, got {observed_seeds}")

    events=pd.concat(audit_frames,ignore_index=True)
    coverage=pd.concat(covers,ignore_index=True)
    per_runner=[]
    pooled=[]
    for seed in sorted(expected_seeds):
        sub=events[events["seed"]==seed]
        for phase in PHASES:
            g=sub[(sub["audit_phase"]==phase) & sub["eligible_as_confirmed_exposure"]]
            n=int(len(g))
            k=int(g["concealment_alarm"].astype(bool).sum())
            per_runner.append({
                "seed":seed, "phase":phase, "evaluable_events":n,
                "alarms":k,"rate":k/n if n else None,
            })
    for phase in PHASES:
        g=events[
            (events["audit_phase"]==phase)
            & events["eligible_as_confirmed_exposure"]
        ]
        n=int(len(g)); k=int(g["concealment_alarm"].astype(bool).sum())
        lo,hi=wilson_interval(k,n)
        sample=[r["rate"] for r in per_runner
                if r["phase"]==phase and r["rate"] is not None]
        pooled.append({
            "phase":phase,
            "denominator_evaluable_events":n,
            "concealment_alarms":k,
            "observed_rate":k/n if n else None,
            "wilson95_low_descriptive":lo,
            "wilson95_high_descriptive":hi,
            "min_runner_rate":min(sample) if sample else None,
            "max_runner_rate":max(sample) if sample else None,
        })
    diagnostics={
        "seed_count":len(docs),
        "seeds":sorted(expected_seeds),
        "total_nrf_events":int(events["nf"].eq("nrf").sum()),
        "original_unlabeled_nrf":int(
            ((events["nf"]=="nrf") & events["legacy_unlabeled"]).sum()
        ),
        "original_unlabeled_alarms":int(
            events.loc[events["legacy_unlabeled"],"concealment_alarm"]
            .astype(bool).sum()
        ),
        "original_unlabeled_reassigned_to_perturbation":int(
            (events["legacy_unlabeled"] & events["audit_phase"].isin(PHASES)).sum()
        ),
        "remaining_unresolved_events":int(events["audit_phase"].eq("unresolved").sum()),
        "remaining_unresolved_alarms":int(
            events.loc[events["audit_phase"]=="unresolved","concealment_alarm"]
            .astype(bool).sum()
        ),
        "independently_labeled_background_events":int(
            events["audit_status"].eq("independent_background_control").sum()
        ),
        "independently_labeled_background_alarms":int(
            events.loc[
                events["audit_status"]=="independent_background_control",
                "concealment_alarm"
            ].astype(bool).sum()
        ),
        "planned_interventions":int(len(coverage)),
        "planned_without_unique_witness":int(
            coverage["missing_external_witness"].astype(bool).sum()
        ),
        "planned_with_duplicate_witness":int(
            coverage["fragmented_duplicate_witness"].astype(bool).sum()
        ),
        "total_verified_evaluable_events":int(events["eligible_as_confirmed_exposure"].sum()),
    }

    output_dir.mkdir(parents=True,exist_ok=True)
    events.to_csv(output_dir/"pooled_boundary_events.csv",index=False)
    coverage.to_csv(output_dir/"pooled_intervention_coverage.csv",index=False)
    pd.DataFrame(per_runner).to_csv(output_dir/"pooled_per_runner_metrics.csv",index=False)
    pd.DataFrame(pooled).to_csv(output_dir/"pooled_phase_metrics.csv",index=False)
    summary={
        "diagnostics":diagnostics,
        "phase_metrics":pooled,
        "per_runner":per_runner,
        "review_status":"post-hoc exploratory replay of previously frozen detector decisions",
        "critical_limits":[
            "No offline cryptographic re-verification: HMAC secret stayed in the original runner.",
            "Original in-run HMAC validation succeeded; replay did not alter alarm decisions.",
            "Original 2-second bins are correlated; Wilson intervals are descriptive only.",
            "Phase attribution uses packet timestamps and archived experimental intervals only for evaluation.",
            "Outside-phase or unresolved events are never auto-assigned to hidden/truthful groups.",
            "A 0/n truthful event-accusation rate does NOT establish <1% FPR.",
        ]
    }
    (output_dir/"pooled_boundary_summary.json").write_text(
        json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8"
    )
    return summary


def main() -> None:
    p=argparse.ArgumentParser(description="Aggregate event-boundary audit across Open5GS runner artifacts")
    p.add_argument("--input-root",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--seeds",type=int,nargs="+",default=[1729,2707,4051])
    args=p.parse_args()
    print(json.dumps(aggregate(args.input_root,args.output_dir,set(args.seeds)),
                     ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
