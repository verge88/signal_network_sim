"""Iteration 4: packet-onset event construction, frozen v1/v2 KIE rescoring.

No labels, phase intervals, masking flags or ground-truth enter packet event
discovery or either KIE scorer. Scores use archived signed-report bytes with
the original, separately validated in-run HMAC provenance: not new HMAC checks.
This is retrospective/exploratory event-selection analysis, not a deployable
prevalidated replacement detector.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from aggregate_open5gs_phase_trial import RUNNERS, CATEGORIES
from audit_open5gs_raw_packet_phase import (
    _collected_requests, max_count_in_sliding_window, DEFAULT_THRESHOLD_COUNT,
)
from run_open5gs_concealment import evaluate_one_event, load_signed_reports
from run_open5gs_concealment_v2 import score_event_v2
from run_open5gs_phase_trial import FROZEN_DESIGN, paired_factorial_plan

GAP_S = 4.0
WINDOW_S = 2.0
MIN_GET_IN_WINDOW = DEFAULT_THRESHOLD_COUNT
KNOWN_STATES = ("acknowledged", "unacknowledged")
VARIANTS = ("v1", "v2")


def packet_onset_candidates(requests: pd.DataFrame) -> pd.DataFrame:
    """Construct bursts ONLY from independent raw HTTP/2 GET packet timestamps.

    A candidate has >=20 GETs in some sliding 2s window. Burst groups are
    divided at packet-to-packet gaps strictly >4s. Event timestamp is the
    FIRST observed packet; no experimental intervals or labels are consulted.
    """
    if not {"ts", "request_key"}.issubset(requests):
        raise ValueError("raw HTTP2 request evidence missing timestamp or identity")
    if requests.request_key.duplicated().any() or requests.ts.isna().any():
        raise ValueError("duplicate/missing raw request identity or timestamp")
    ordered = requests[["ts", "request_key"]].sort_values("ts").reset_index(drop=True)
    if ordered.empty:
        return pd.DataFrame(columns=[
            "ts","nf","external_event","nf_instance_id","raw_get_count",
            "peak_sliding_2s","first_get_ts","last_get_ts","packet_group_id"
        ])
    ordered["packet_group_id"] = (
        ordered.ts.diff().fillna(0).gt(GAP_S).cumsum().astype(int)
    )
    rows = []
    for groupid,g in ordered.groupby("packet_group_id"):
        values = g.ts.to_numpy(float)
        peak = max_count_in_sliding_window(values, WINDOW_S)
        if peak < MIN_GET_IN_WINDOW:
            continue
        first, last = float(values[0]),float(values[-1])
        rows.append({
            "ts":first, "nf":"nrf", "external_event":"nrf_query_burst",
            "nf_instance_id":None, "raw_get_count":len(values),
            "peak_sliding_2s":peak, "first_get_ts":first,"last_get_ts":last,
            "packet_group_id":int(groupid),
        })
    return pd.DataFrame(rows)


def score_frozen(candidates: pd.DataFrame, signed_reports: pd.DataFrame) -> pd.DataFrame:
    """Use production-unchanged decision routines; only event source differs."""
    if candidates.empty:
        return pd.DataFrame()
    out = []
    for event in candidates.to_dict("records"):
        # Only packet identity/time enters the detector. In particular, no
        # ground truth, randomization or masking flag passes to this function.
        basic = {k:event[k] for k in ("ts","nf","external_event","nf_instance_id")}
        v1 = evaluate_one_event(basic,signed_reports)
        v2 = score_event_v2(basic,signed_reports)
        if float(v1["ts"]) != float(v2["ts"]) or v1["nf"] != v2["nf"]:
            raise AssertionError("KIE variants scored different packet-onset evidence")
        row = dict(event)
        for name,item in (("v1",v1),("v2",v2)):
            row[name+"_state"]=item["state"]
            row[name+"_alarm"]=bool(item["concealment_alarm"])
            row[name+"_known"]=item["state"] in KNOWN_STATES
            row[name+"_pre_reports"]=int(item["pre_reports"])
            row[name+"_post_reports"]=int(item["post_reports"])
            row[name+"_cpu_response_ticks"]=item.get("cpu_response_ticks")
        out.append(row)
    return pd.DataFrame(out)


def _archive_provenance(directory: Path, seed: int, period: float) -> dict:
    manifest=json.loads((directory/"phase_trial_manifest.json").read_text())
    if (manifest.get("design")!=FROZEN_DESIGN or
        int(manifest["seed"])!=seed or float(manifest["kie_period_s"])!=period or
        manifest["ordered_interventions"]!=paired_factorial_plan(seed)):
        raise ValueError(f"randomization/manifest altered: {seed}")
    verify=json.loads((directory/"kie_report_validation.json").read_text())
    if (verify.get("all_signatures_valid") is not True or
        int(verify.get("invalid_signature_count",-1))!=0 or
        verify.get("all_reports_truth_paired") is not True or
        not (int(verify["report_count"])>0 and
             int(verify["report_count"])==int(verify["paired_collector_samples"]))):
        raise ValueError(f"live original KIE HMAC/collector evidence failed: {seed}")
    reports_file=directory/"kie_reports.jsonl"
    reports=[json.loads(x) for x in reports_file.read_text().splitlines() if x.strip()]
    if len(reports)!=int(verify["report_count"]):
        raise ValueError(f"archived KIE reports differ from signed run count: {seed}")
    return {
        "seed":seed,"period_s":period,
        "original_live_hmac_validated":True,
        "offline_hmac_reverified":False,
        "original_signed_reports":len(reports),
        "archived_report_sha256":hashlib.sha256(reports_file.read_bytes()).hexdigest(),
    }


def _trial_inventory(directory: Path, seed: int, period: float) -> pd.DataFrame:
    intervals=pd.read_csv(directory/"phase_intervals.csv")
    planned=intervals[intervals.phase.isin(CATEGORIES)][
        ["phase","cycle","start_ts","end_ts"]
    ].copy()
    if len(planned)!=12 or planned.duplicated(["cycle","phase"]).any():
        raise ValueError(f"missing/duplicate intervention intervals: {seed}")
    design=pd.DataFrame(paired_factorial_plan(seed))
    if design.duplicated(["cycle","phase"]).any():
        raise AssertionError("duplicate preregistered trial")
    planned=planned.merge(
        design,how="outer",on=["cycle","phase"],validate="one_to_one",indicator=True
    )
    if not planned._merge.eq("both").all():
        raise ValueError(f"missing randomized trial: {seed}")
    prev=pd.read_csv(directory/"boundary_intervention_coverage.csv")
    if len(prev)!=12 or prev.duplicated(["cycle","phase"]).any():
        raise ValueError(f"missing original packet evidence coverage: {seed}")
    planned=planned.merge(
        prev[["cycle","phase","witnessed_events"]],
        on=["cycle","phase"],how="outer",validate="one_to_one",indicator="_prev"
    )
    if not planned._prev.eq("both").all():
        raise ValueError(f"unmatched old witness: {seed}")
    planned=planned.rename(columns={"witnessed_events":"legacy_2s_event_count"})
    plan=pd.read_csv(directory/"randomized_alignment.csv")
    if len(plan)!=12 or plan.duplicated(["cycle","phase"]).any():
        raise ValueError(f"missing onset alignment: {seed}")
    planned=planned.merge(
        plan[["cycle","phase","actual_phase_fraction","phase_alignment_overrun"]],
        how="outer",on=["cycle","phase"],validate="one_to_one",indicator="_align"
    )
    if not planned._align.eq("both").all():
        raise ValueError(f"unmatched alignment: {seed}")
    if planned.phase_alignment_overrun.astype(bool).any():
        raise ValueError(f"realized sampler phase overrun (not silently included): {seed}")
    planned["seed"]=seed
    planned["period_s"]=period
    return planned.drop(columns=["_merge","_prev","_align"])


def _mask_exposure(directory: Path, event_ts: float) -> dict:
    # Only called AFTER completely frozen KIE decisions.
    truth=pd.read_json(directory/"kie_ground_truth.jsonl",lines=True)
    nrf=truth[truth.nf_id.astype(str).str.lower().eq("nrf")]
    relevant=nrf[(nrf.ts>=event_ts-1) & (nrf.ts<=event_ts+4)]
    return {
        "collector_nrf_samples":int(len(relevant)),
        "masked_nrf_samples":int(relevant.masking_active.astype(bool).sum()),
        "exposed_to_mask":bool(relevant.masking_active.astype(bool).any()),
    }


def evaluate_runner(directory: Path, seed: int, period: float) -> tuple[pd.DataFrame,pd.DataFrame,dict]:
    provenance=_archive_provenance(directory,seed,period)
    # Blind event inference: packet timestamps/identity and frozen report data
    # only. This must run BEFORE any interval or experimental label is read.
    requests=_collected_requests(directory/"sbi_http2_events.tsv",
                                directory/"open5gs_endpoints.json")
    candidates=packet_onset_candidates(requests)
    if candidates.empty:
        raise ValueError(f"no packet-onset independent witnesses: {seed}")
    signed=load_signed_reports(directory/"kie_reports.jsonl",secret=None)
    # As above, offline valid_signature=True is strictly provenance-based,
    # NOT a new cryptographic verification of archived historical keys.
    scored=score_frozen(candidates,signed)

    trials=_trial_inventory(directory,seed,period)
    selected={}
    unassigned=[]
    for idx,record in scored.iterrows():
        inside=trials[
            (trials.start_ts<=float(record["ts"])) &
            (trials.end_ts>float(record["ts"]))
        ]
        if len(inside)!=1:
            unassigned.append(float(record["ts"]))
            continue
        row=inside.iloc[0]
        key=(int(row["cycle"]),str(row["phase"]))
        if key in selected:
            raise ValueError(f"fragmented/duplicate packet-onset burst in {seed} {key}")
        selected[key]=record.to_dict()
    if unassigned:
        raise ValueError(f"independent candidate outside trial intervals: {seed}: {unassigned}")

    # Independent packet witness completeness measured at the population
    # level: missing stays missing; no oracle-created synthetic event.
    exposures=[]
    old_events=pd.read_csv(directory/"v2_concealment_events.csv")
    for trial in trials.to_dict("records"):
        key=(int(trial["cycle"]),str(trial["phase"]))
        evidence=selected.get(key)
        row={
            "seed":seed,"period_s":period,"cycle":key[0],"phase":key[1],
            "profile":trial["profile"],
            "assigned_phase_fraction":float(trial["assigned_phase_fraction"]),
            "actual_phase_fraction":float(trial["actual_phase_fraction"]),
            "legacy_2s_event_count":int(trial["legacy_2s_event_count"]),
            "packet_onset_witness":evidence is not None,
            "newly_confirmed":evidence is not None and int(trial["legacy_2s_event_count"])==0,
        }
        if evidence is not None:
            for col,val in evidence.items():
                if col in ("nf","external_event","nf_instance_id","packet_group_id"):
                    continue
                row[col]=val
            # Ground truth and previous decisions are evaluation-only.
            row.update(_mask_exposure(directory,float(evidence["ts"])))
            prev=old_events[
                old_events.audit_phase.eq(key[1]) &
                old_events.audit_cycle.eq(key[0]) &
                old_events.nf.eq("nrf")
            ]
            if int(trial["legacy_2s_event_count"])==1 and len(prev)!=1:
                raise ValueError(f"legacy event identity mismatch: {seed}, {key}")
            if int(trial["legacy_2s_event_count"])==0 and len(prev)!=0:
                raise ValueError(f"legacy missing witness inconsistency: {seed}, {key}")
            if len(prev)==1:
                row["legacy_v1_alarm"]=bool(prev.iloc[0]["v1_alarm"])
                row["legacy_v2_alarm"]=bool(prev.iloc[0]["v2_alarm"])
                row["legacy_event_ts"]=float(prev.iloc[0]["ts"])
                row["legacy_v1_state"]=str(
                    pd.read_csv(directory/"concealment_events.csv").loc[
                        lambda f:(f.ts.eq(float(prev.iloc[0]["ts"]))) &
                            f.nf.eq("nrf"),"state"
                    ].iloc[0]
                )
                row["legacy_v2_state"]=str(prev.iloc[0]["state"])
            else:
                row["legacy_v1_alarm"]=None
                row["legacy_v2_alarm"]=None
                row["legacy_event_ts"]=None
                row["legacy_v1_state"]=None
                row["legacy_v2_state"]=None
        exposures.append(row)
    df=pd.DataFrame(exposures)
    if len(df)!=12 or df.duplicated(["cycle","phase"]).any():
        raise AssertionError("trial-level coverage lost")
    scored.insert(0,"seed",seed)
    scored.insert(1,"period_s",period)
    provenance["independent_raw_gets"]=len(requests)
    provenance["eligible_packet_candidates"]=len(scored)
    provenance["original_fixed_2s_witnessed"]=int(df.legacy_2s_event_count.eq(1).sum())
    provenance["packet_witnessed"]=int(df.packet_onset_witness.sum())
    return df,scored,provenance


def run(input_root: Path, output_dir: Path) -> dict:
    trials,blind_events,provenance=[],[],[]
    for seed,period in sorted(RUNNERS.items()):
        candidates=list(input_root.rglob(f"5g-open5gs-phase-{seed}"))
        if len(candidates)!=1:
            raise ValueError(f"expected exactly one independent artifact: {seed}")
        observed,unlabeled,prov=evaluate_runner(candidates[0],seed,period)
        trials.append(observed);blind_events.append(unlabeled);provenance.append(prov)
    all_trials=pd.concat(trials,ignore_index=True)
    blind=pd.concat(blind_events,ignore_index=True)
    if len(all_trials)!=144 or len(provenance)!=12:
        raise ValueError("not all frozen environments accounted for")
    output_dir.mkdir(parents=True,exist_ok=True)
    all_trials.to_csv(output_dir/"packet_onset_kie_trials.csv",index=False)
    blind.to_csv(output_dir/"blind_packet_onset_events.csv",index=False)
    pd.DataFrame(provenance).to_csv(output_dir/"report_signature_provenance.csv",index=False)

    def summaries(data:pd.DataFrame,cohort:str)->list[dict]:
        summary=[]
        for (period,profile,phase),g in data.groupby(["period_s","profile","phase"]):
            for det in VARIANTS:
                confirmed=g[g.packet_onset_witness].copy()
                known=confirmed[confirmed[det+"_known"].eq(True)]
                summary.append({
                    "cohort":cohort,"period_s":period,"profile":profile,"phase":phase,
                    "detector":det,"planned":len(g),"independent_runners":g.seed.nunique(),
                    "packet_witnessed":len(confirmed),"legacy_2s_witnessed":
                        int(g.legacy_2s_event_count.eq(1).sum()),
                    "evaluable":len(known),"unknown":len(confirmed)-len(known),
                    "alarms":int(known[det+"_alarm"].sum()),
                    "rate_on_evaluable":float(known[det+"_alarm"].mean()) if len(known) else None,
                    "collector_mask_exposed":int(confirmed.exposed_to_mask.sum()),
                })
        return summary
    cells=summaries(all_trials,"all12")+summaries(
        all_trials[all_trials.seed<30000],"original6"
    )+summaries(all_trials[all_trials.seed>=30000],"replicate6")
    cells_df=pd.DataFrame(cells)
    cells_df.to_csv(output_dir/"packet_onset_descriptive_cells.csv",index=False)

    missing=all_trials[all_trials.legacy_2s_event_count.eq(0)]
    newly=missing[missing.packet_onset_witness]
    gap=all_trials[all_trials.packet_onset_witness]
    summary={
        "classification":"retrospective packet-onset external-event reconstruction, frozen v1/v2 response rules",
        "planned_interventions":len(all_trials),
        "independent_runners":len(provenance),
        "raw_packet_events_unmatched":int((~all_trials.packet_onset_witness).sum()),
        "original_fixed_window_nonwitnesses":len(missing),
        "recovered_packet_onset_witnesses":len(newly),
        "frozen_v1_v2_cpu_rule_modified":False,
        "external_event_timestamp_changed":True,
        "historical_offline_hmac_reverified":False,
        "archived_live_hmac_provenance_checked":True,
        "new_witnesses_exposed_to_mask":int(newly.exposed_to_mask.sum()),
        "new_witnesses_v1_evaluable":int(newly.v1_known.eq(True).sum()),
        "new_witnesses_v2_evaluable":int(newly.v2_known.eq(True).sum()),
        "new_witnesses_v1_alarms":int(newly[newly.v1_known.eq(True)].v1_alarm.sum()),
        "new_witnesses_v2_alarms":int(newly[newly.v2_known.eq(True)].v2_alarm.sum()),
        "known_packet_evidence_events":len(gap),
        "interpretation":[
            "Packet onset derived blind from >20 passively witnessed HTTP/2 GET in any sliding 2s window.",
            "Original pre/post KIE v1/v2 CPU-tick logic reused unchanged, but event timestamp and witness definition changed.",
            "Mask/truth and scheduled trial intervals enter only after blind packet-onset scoring.",
            "Offline historical signature verification is impossible without ephemeral keys; original contemporaneous validity must be distinguished from re-verification.",
            "A restored packet witness may still be unknown or wrongly classified by KIE; do not count unknown as correct negative.",
            "Within-run events are correlated; no population p-value or operational false-positive claim.",
        ],
    }
    (output_dir/"packet_onset_kie_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2))
    print(cells_df.to_csv(index=False))
    print("RECOVERED_15_DETAIL")
    print(newly[[
        "seed","period_s","cycle","phase","profile","ts",
        "v1_state","v2_state","v1_alarm","v2_alarm",
        "exposed_to_mask","masked_nrf_samples"
    ]].to_csv(index=False))
    return summary


def main()->None:
    p=argparse.ArgumentParser()
    p.add_argument("--input-root",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    args=p.parse_args()
    run(args.input_root,args.output_dir)


if __name__=="__main__":
    main()
