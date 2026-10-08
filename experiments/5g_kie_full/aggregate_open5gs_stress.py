"""Evaluation-only aggregation for 2x2 real Open5GS cadence/load stress.

Phase, burst profile, fault-injection truth, and background-rate labels are
NEVER provided to v1 or v2. All those annotations are joined after decisions
are archived. Missing witnessed events and missing sidecar samples stay visible.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from aggregate_open5gs_boundary_audit import wilson_interval

PHASES=("truthful_udm_restart","hidden_udm_restart",
        "truthful_nrf_burst","hidden_nrf_burst")

def _read_truth(path:Path) -> pd.DataFrame:
    with path.open(encoding="utf-8") as handle:
        rows=[json.loads(line) for line in handle if line.strip()]
    if not rows:
        raise ValueError("missing ground-truth KIE readings")
    return pd.DataFrame([
        {"ts":float(r["ts"]),"nf":str(r["nf_id"]).lower(),
         "masking_active":bool(r.get("masking_active",False))}
        for r in rows
    ])

def collect(root:Path, output:Path, expected:dict[int,tuple[float,float]]) -> dict:
    events=[]; interventions=[]; signatures=[]; conditions=[]
    for seed,(period,load) in sorted(expected.items()):
        folders=list(root.rglob(f"5g-open5gs-stress-{seed}"))
        if len(folders)!=1:
            raise ValueError(f"expected one artifact folder for seed {seed}, got {len(folders)}")
        folder=folders[0]
        manifest=json.loads((folder/"stress_manifest.json").read_text())
        if (manifest["seed"]!=seed or
            float(manifest["kie_period_s"])!=period or
            float(manifest["background_rps"])!=load):
            raise ValueError(f"frozen stress factor mismatch for {seed}")
        validation=json.loads((folder/"kie_report_validation.json").read_text())
        if not validation.get("all_signatures_valid",False):
            raise ValueError(f"invalid/missing live HMAC validation for {seed}")
        signatures.append({
            "seed":seed,
            "report_count":validation["report_count"],
            "invalid_signature_count":validation["invalid_signature_count"],
            "live_all_signatures_valid":True
        })
        original=pd.read_csv(folder/"concealment_events.csv")
        v2=pd.read_csv(folder/"v2_concealment_events.csv")
        intervals=pd.read_csv(folder/"phase_intervals.csv")
        burst_profiles=pd.read_csv(folder/"burst_profiles.csv")
        truth=_read_truth(folder/"kie_ground_truth.jsonl")
        expected_phases=intervals[intervals.phase.isin(PHASES)]
        if len(expected_phases)!=int(manifest["cycles"])*4:
            raise ValueError(f"incomplete planned protocol for seed {seed}")
        cols=["ts","nf","external_event"]
        if original.duplicated(cols).any() or v2.duplicated(cols).any():
            raise ValueError("duplicate external event witness identity")
        merged=v2.merge(
            original[cols+["state","concealment_alarm","integrity_alarm"]].rename(
                columns={"state":"v1_state",
                         "concealment_alarm":"v1_alarm_source",
                         "integrity_alarm":"v1_integrity_alarm"}
            ),on=cols,validate="one_to_one",how="left"
        )
        if merged.v1_state.isna().any() or len(merged)!=len(original):
            raise ValueError("missing/extra model input event")
        if not (merged.v1_alarm.astype(bool)==merged.v1_alarm_source.astype(bool)).all():
            raise ValueError("v1 decisions altered during v2 scoring")
        if len(burst_profiles)!=len(expected_phases):
            raise ValueError("missing per-intervention profile trace")
        key={
            (str(r.phase),int(r.cycle)):(str(r.profile),int(r.requested),int(r.successful))
            for r in burst_profiles.itertuples()
        }
        bgpath=folder/"nrf_background_load.csv"
        if load>0:
            if not bgpath.exists(): raise ValueError("requested background load is missing")
            bg=pd.read_csv(bgpath)
            successes=int((bg.curl_exit_code==0).sum())
            attempted=int(len(bg))
        else:
            successes=0;attempted=0
        conditions.append({"seed":seed,"kie_period_s":period,"background_rps":load,
                           "background_attempted":attempted,"background_successful":successes,
                           "planned_interventions":len(expected_phases)})
        for row in merged.to_dict("records"):
            phase=str(row["audit_phase"])
            cycle=int(row["audit_cycle"])
            profile,req,ok=key.get((phase,cycle),("unresolved",0,0))
            t=float(row["ts"])
            sub=truth[(truth.nf==row["nf"]) &
                      (truth.ts>=t-1.0)&(truth.ts<=t+4.0)]
            row.update({
                "seed":seed,
                "kie_period_s":period,
                "background_rps":load,
                "profile":profile,
                "requested_nrf_gets":req if row["nf"]=="nrf" else None,
                "successful_nrf_gets":ok if row["nf"]=="nrf" else None,
                "v1_known":str(row["v1_state"]) in ("acknowledged","unacknowledged"),
                "v2_known":str(row["state"]) in ("acknowledged","unacknowledged"),
                "decision_window_report_count":int(len(sub)),
                "masked_report_exposure":int(sub.masking_active.sum()),
                "masked_report_exposure_known":bool(len(sub)>0),
            })
            events.append(row)
        for p in expected_phases.itertuples():
            matching=merged[(merged.audit_phase==p.phase)&(merged.audit_cycle==p.cycle)]
            profile,req,ok=key[(p.phase,int(p.cycle))]
            interventions.append({
                "seed":seed,"kie_period_s":period,"background_rps":load,
                "phase":p.phase,"cycle":int(p.cycle),"profile":profile,
                "witness_count":len(matching),
                "missing_external_witness":bool(matching.empty),
                "duplicate_external_witness":bool(len(matching)>1),
                "attempted_nrf_gets":req if "nrf" in p.phase else None,
                "successful_nrf_gets":ok if "nrf" in p.phase else None,
            })

    all_events=pd.DataFrame(events)
    all_interventions=pd.DataFrame(interventions)
    per_condition=[]; pooled=[]
    for phase in PHASES:
        sub=all_events[all_events.audit_phase==phase]
        for profile in ("short","standard"):
            if "nrf" not in phase and profile=="standard":continue
            eligible=sub[sub.profile==profile] if "nrf" in phase else sub
            if eligible.empty: continue
            n=len(eligible)
            v1=eligible[eligible.v1_known]
            v2=eligible[eligible.v2_known]
            lo,hi=wilson_interval(int(v2.v2_alarm.astype(bool).sum()),len(v2))
            pooled.append({
                "phase":phase,"profile":profile if "nrf" in phase else "all",
                "external_events":n,
                "v1_evaluable":len(v1),
                "v2_evaluable":len(v2),
                "v1_alarms":int(v1.v1_alarm.astype(bool).sum()),
                "v2_alarms":int(v2.v2_alarm.astype(bool).sum()),
                "v1_unknown":n-len(v1),"v2_unknown":n-len(v2),
                "v2_wilson95_low_descriptive":lo,
                "v2_wilson95_high_descriptive":hi,
                "no_masked_report_in_v2_window":(
                    int((eligible.masked_report_exposure==0).sum())
                    if phase.startswith("hidden") else None
                ),
            })
        for seed in expected:
            cell=sub[sub.seed==seed]
            for profile in ("short","standard"):
                sample=cell[cell.profile==profile] if "nrf" in phase else cell
                if "nrf" not in phase and profile=="standard":continue
                n=len(sample)
                for version,state_col,alarm_col in (
                    ("v1","v1_known","v1_alarm"),
                    ("v2","v2_known","v2_alarm")
                ):
                    evals=sample[sample[state_col]]
                    per_condition.append({
                        "seed":seed,"period_s":expected[seed][0],
                        "background_rps":expected[seed][1],
                        "phase":phase,"profile":profile if "nrf" in phase else "all",
                        "version":version,"witnessed_events":n,
                        "evaluable_events":len(evals),
                        "unknown_events":n-len(evals),
                        "alarms":int(evals[alarm_col].astype(bool).sum()),
                    })
    output.mkdir(parents=True,exist_ok=True)
    all_events.to_csv(output/"stress_event_comparisons.csv",index=False)
    all_interventions.to_csv(output/"stress_intervention_coverage.csv",index=False)
    pd.DataFrame(per_condition).to_csv(output/"stress_per_condition.csv",index=False)
    pd.DataFrame(pooled).to_csv(output/"stress_pooled_metrics.csv",index=False)
    pd.DataFrame(signatures).to_csv(output/"stress_hmac_provenance.csv",index=False)
    summary={
        "design":"2x2 sampling cadence x background traffic, short/standard within-run paired",
        "runs":conditions,
        "total_event_witnesses":len(all_events),
        "total_planned_interventions":len(all_interventions),
        "missing_external_witnesses":int(all_interventions.missing_external_witness.sum()),
        "duplicate_external_witnesses":int(all_interventions.duplicate_external_witness.sum()),
        "ambiguous_unlabeled_events":int(all_events.audit_phase.eq("unresolved").sum()),
        "invalid_hmac_reports":int(sum(x["invalid_signature_count"] for x in signatures)),
        "total_hmac_verified_reports":int(sum(x["report_count"] for x in signatures)),
        "pooled_metrics":pooled,
        "limitations":[
            "All detector verdicts are frozen before evaluation profile/fault labels are attached.",
            "Hidden phases with zero signed masked samples in the event window are not evidence of a concealment-reporting failure.",
            "Small n per cell; Wilson intervals are descriptive and ignore within-run dependence.",
            "Background HTTP2 traffic is a benign load, not an adversarial network compromise.",
            "HMAC authenticates bytes from the sidecar and does not remotely attest NF truthfulness.",
        ]
    }
    (output/"stress_summary.json").write_text(
        json.dumps(summary,indent=2,ensure_ascii=False),encoding="utf-8")
    return summary

def main()->None:
    p=argparse.ArgumentParser()
    p.add_argument("--input-root",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    args=p.parse_args()
    expected={
        13007:(1.0,0.0),
        14009:(1.0,2.0),
        15013:(4.0,0.0),
        16001:(4.0,2.0),
    }
    result=collect(args.input_root,args.output_dir,expected)
    print(json.dumps(result,indent=2,ensure_ascii=False))

if __name__=="__main__":
    main()
