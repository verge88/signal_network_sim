"""Independent dual capture and stdlib-only NRF HTTP/2 witness validation.

This analyst does not call the prior PCAP feature decoder, audit parser or
packet_onset_candidates. Dual tcpdump/tshark captures have separate processes
but share one loopback interface and tshark dissector. This is *not* a
physically independent external laboratory. KIE v1/v2 scorers remain frozen.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from reevaluate_open5gs_packet_onset_kie import _archive_provenance, _load_checked_prefix
from run_open5gs_concealment import evaluate_one_event
from run_open5gs_concealment_v2 import score_event_v2
from run_open5gs_phase_trial import paired_factorial_plan

RUNNERS={41011:1.0,41013:1.0,42011:2.0,42013:2.0,44011:4.0,44013:4.0}
EVENT_THRESHOLD=20
WINDOW_SECONDS=2.0
MIN_GROUP_GAP_SECONDS=4.0
SCENARIOS={"truthful_nrf_burst","hidden_nrf_burst"}
KNOWN={"acknowledged","unacknowledged"}


def decode_independent_tsv(tsv:Path, endpoints:Path)->list[dict]:
    """Use ONLY Python's CSV/json libraries, not old pandas tshark decoder."""
    host=json.loads(endpoints.read_text(encoding="utf-8"))
    nrf_ips={
        str(x["address"]) for x in host.get("endpoints",[])
        if str(x.get("nf","")).lower()=="nrf"
    }
    if not nrf_ips:
        raise ValueError("NRF endpoint mapping absent")
    found={}
    with tsv.open(newline="",encoding="utf-8") as stream:
        reader=csv.DictReader(stream,delimiter="\t")
        for row in reader:
            if row.get("ip.dst","") not in nrf_ips:
                continue
            names=str(row.get("http2.header.name","")).split(",")
            values=str(row.get("http2.header.value","")).split(",")
            headers={k.strip().lower():(values[i].strip() if i<len(values) else "")
                     for i,k in enumerate(names)}
            if headers.get(":method","").upper()!="GET":
                continue
            if "/nnrf-" not in headers.get(":path",""):
                continue
            tcp=str(row.get("tcp.stream","")).strip()
            h2=str(row.get("http2.streamid","")).split(",")[0].strip()
            if not tcp or not h2:
                raise ValueError("missing independent HTTP2 request identity")
            key=f"{tcp}|{h2}"
            ts=float(row["frame.time_epoch"])
            if key not in found or found[key]["ts"]>ts:
                found[key]={"ts":ts,"key":key}
    return sorted(found.values(),key=lambda x:x["ts"])


def sliding_peak(values:list[float],seconds:float=WINDOW_SECONDS)->int:
    if seconds<=0:
        raise ValueError("nonpositive sliding witness window")
    timestamps=sorted(values)
    left=0
    best=0
    for i,t in enumerate(timestamps):
        while t-timestamps[left]>=seconds:
            left+=1
        best=max(best,i-left+1)
    return best


def blind_bursts(packets:list[dict])->list[dict]:
    """Build events from raw packets alone, prior to opening trial labels."""
    groups=[]
    group=[]
    for packet in packets:
        if group and packet["ts"]-group[-1]["ts"]>MIN_GROUP_GAP_SECONDS:
            groups.append(group);group=[]
        group.append(packet)
    if group:groups.append(group)
    events=[]
    for g in groups:
        peak=sliding_peak([x["ts"] for x in g])
        if peak<EVENT_THRESHOLD:continue
        events.append({
            "ts":g[0]["ts"],"raw_get_count":len(g),
            "peak_2s_gets":peak,"last_ts":g[-1]["ts"],
        })
    return events


def frozen_kie(event:dict,reports:pd.DataFrame)->dict:
    """Call original untouched detector functions; never pass any label."""
    payload={
        "ts":event["ts"],"nf":"nrf",
        "external_event":"nrf_query_burst","nf_instance_id":None,
    }
    v1=evaluate_one_event(payload,reports)
    v2=score_event_v2(payload,reports)
    result=dict(event)
    for name,scored in (("v1",v1),("v2",v2)):
        result[name+"_state"]=str(scored["state"])
        result[name+"_known"]=str(scored["state"]) in KNOWN
        result[name+"_alarm"]=bool(scored["concealment_alarm"])
        result[name+"_pre_reports"]=int(scored["pre_reports"])
        result[name+"_post_reports"]=int(scored["post_reports"])
    return result


def validate_runner(directory:Path,seed:int,period:float)->tuple[pd.DataFrame,pd.DataFrame,dict]:
    # No truth, phase or control input is parsed before the independent
    # network events have been discovered AND frozen KIE scores archived.
    validation=_archive_provenance(directory,seed,period)
    a=decode_independent_tsv(directory/"sbi_http2_events.tsv",
                             directory/"open5gs_endpoints.json")
    b=decode_independent_tsv(directory/"observer_tshark_http2_events.tsv",
                             directory/"open5gs_endpoints.json")
    report_data=_load_checked_prefix(directory,validation["original_signed_reports"])
    candidates_a=[frozen_kie(event,report_data) for event in blind_bursts(a)]
    candidates_b=[frozen_kie(event,report_data) for event in blind_bursts(b)]
    validation.update({
        "tcpdump_nrf_gets":len(a),"tshark_nrf_gets":len(b),
        "tcpdump_candidate_bursts":len(candidates_a),
        "tshark_candidate_bursts":len(candidates_b),
    })

    plan=pd.DataFrame(paired_factorial_plan(seed))
    intervals=pd.read_csv(directory/"phase_intervals.csv")
    episodes=intervals[intervals.phase.isin(SCENARIOS)].copy()
    if len(episodes)!=12 or episodes.duplicated(["cycle","phase"]).any():
        raise ValueError(f"invalid independent trial intervals: {seed}")
    episodes=episodes.merge(
        plan,on=["cycle","phase"],how="outer",validate="one_to_one",indicator=True
    )
    if not episodes._merge.eq("both").all():
        raise ValueError(f"independent manifest pairing failed: {seed}")

    # Independent witness labelling is EVALUATION-ONLY; a missing event is
    # not converted into a KIE false negative or scored by oracle ts.
    labels=[]
    def locate(found, start, end):
        return [r for r in found if start<=r["ts"]<end]
    truth=pd.read_json(directory/"kie_ground_truth.jsonl",lines=True)
    nrf_truth=truth[truth.nf_id.astype(str).str.lower().eq("nrf")]
    for trial in episodes.to_dict("records"):
        start,end=float(trial["start_ts"]),float(trial["end_ts"])
        pa=locate(candidates_a,start,end)
        pb=locate(candidates_b,start,end)
        if len(pa)>1 or len(pb)>1:
            raise ValueError(f"multiple independent witness events in one trial: {seed}")
        row={
            "seed":seed,"period_s":period,"cycle":int(trial["cycle"]),
            "phase":trial["phase"],"profile":trial["profile"],
            "assigned_fraction":float(trial["assigned_phase_fraction"]),
            "tcpdump_witness":len(pa)==1,
            "tshark_witness":len(pb)==1,
            "observer_agrees":len(pa)==len(pb),
            "tcpdump_raw_get_count":sum(start<=r["ts"]<end for r in a),
            "tshark_raw_get_count":sum(start<=r["ts"]<end for r in b),
        }
        if pa and pb:
            row["onset_difference_ms"]=1000.*(pa[0]["ts"]-pb[0]["ts"])
            row["observer_agrees"]=(abs(row["onset_difference_ms"])<=100.0)
            for name in ("v1","v2"):
                row[name+"_observer_decision_agrees"]=(
                    pa[0][name+"_state"]==pb[0][name+"_state"] and
                    pa[0][name+"_alarm"]==pb[0][name+"_alarm"]
                )
        else:
            row["onset_difference_ms"]=None
            for name in ("v1","v2"):
                row[name+"_observer_decision_agrees"]=False
        if pa:
            for col,value in pa[0].items():
                row[col]=value
            sampled=nrf_truth[
                (nrf_truth.ts>=pa[0]["ts"]-1) &
                (nrf_truth.ts<=pa[0]["ts"]+4)
            ]
            row["masked_sample_exposure"]=bool(sampled.masking_active.astype(bool).any())
            row["masked_sample_count"]=int(sampled.masking_active.astype(bool).sum())
        else:
            row.update({"masked_sample_exposure":None,"masked_sample_count":None})
        labels.append(row)

    result=pd.DataFrame(labels)
    if result.duplicated(["phase","cycle"]).any() or len(result)!=12:
        raise ValueError("trial-level analysis incomplete")
    if result.phase.str.startswith("truthful_").any() and bool(
        result.loc[result.phase.str.startswith("truthful_"),"masked_sample_exposure"].fillna(False).any()
    ):
        raise ValueError("truthful validation trial unexpectedly contains masked samples")

    # Negative controls were run after interventions with masks disabled.
    controls=pd.read_csv(directory/"independent_negative_controls.csv")
    if len(controls)!=4 or sorted(controls.control_type.value_counts().to_dict().items()) != [
        ("benign_low_rate",2),("idle",2)
    ]:
        raise ValueError("four separate independent benign/idle controls required")
    negatives=[]
    for c in controls.to_dict("records"):
        start,end=float(c["start_ts"]),float(c["end_ts"])
        pa=[x["ts"] for x in a if start<=x["ts"]<end]
        pb=[x["ts"] for x in b if start<=x["ts"]<end]
        incident_a=locate(candidates_a,start,end)
        incident_b=locate(candidates_b,start,end)
        negatives.append({
            "seed":seed,"period_s":period,"block":int(c["block"]),
            "control":c["control_type"],"attempted":int(c["attempted"]),
            "successful":int(c["successful"]),
            "tcpdump_gets":len(pa),"tshark_gets":len(pb),
            "tcpdump_peak_2s":sliding_peak(pa),"tshark_peak_2s":sliding_peak(pb),
            "tcpdump_false_bursts":len(incident_a),
            "tshark_false_bursts":len(incident_b),
        })
    negatives=pd.DataFrame(negatives)
    if not negatives.loc[negatives.control.eq("benign_low_rate"),"attempted"].eq(10).all():
        raise ValueError("independent negative control stimulus protocol changed")
    validation["observer_trial_matches"]=int(result.observer_agrees.sum())
    validation["observer_event_score_matches"]=int(
        (result.v1_observer_decision_agrees & result.v2_observer_decision_agrees).sum()
    )
    validation["independent_negative_controls"]=len(negatives)
    return result,negatives,validation


def aggregate(root:Path,output:Path)->dict:
    trials=[];controls=[];provenance=[]
    for seed,period in sorted(RUNNERS.items()):
        candidates=list(root.rglob(f"5g-independent-phase-{seed}"))
        if len(candidates)!=1:
            raise ValueError(f"exactly one independent validation artifact required for {seed}")
        trial,negative,prov=validate_runner(candidates[0],seed,period)
        trials.append(trial);controls.append(negative);provenance.append(prov)
    all_trials=pd.concat(trials,ignore_index=True)
    all_controls=pd.concat(controls,ignore_index=True)
    if len(all_trials)!=72 or len(all_controls)!=24:
        raise ValueError("missing validation interventions or controls")
    output.mkdir(parents=True,exist_ok=True)
    all_trials.to_csv(output/"independent_validation_trials.csv",index=False)
    all_controls.to_csv(output/"independent_negative_controls.csv",index=False)
    pd.DataFrame(provenance).to_csv(output/"independent_runner_provenance.csv",index=False)
    metrics=[]
    for (period,profile,phase),g in all_trials.groupby(["period_s","profile","phase"]):
        for version in ("v1","v2"):
            w=g[g.tcpdump_witness & g.tshark_witness & g.observer_agrees]
            known=w[w[version+"_known"].eq(True)]
            metrics.append({
                "period_s":period,"profile":profile,"phase":phase,
                "detector":version,"planned":len(g),"independent_runners":g.seed.nunique(),
                "two_observers_agree":len(w),"known":len(known),
                "unknown":len(w)-len(known),
                "alarms":int(known[version+"_alarm"].sum()),
                "rate":float(known[version+"_alarm"].mean()) if len(known) else None,
                "masked_exposure":int(w.masked_sample_exposure.sum()),
            })
    pd.DataFrame(metrics).to_csv(output/"independent_validation_by_condition.csv",index=False)
    summary={
        "classification":"new prospective six-runner cross-OS, dual-capture lab replication; not external-laboratory validation",
        "runner_os":"ubuntu-24.04",
        "independent_runners":6,"planned_interventions":72,
        "negative_control_windows":24,
        "tcpdump_witnessed":int(all_trials.tcpdump_witness.sum()),
        "second_tshark_witnessed":int(all_trials.tshark_witness.sum()),
        "independent_capture_agreement":int(all_trials.observer_agrees.sum()),
        "v1_v2_score_agreement_between_captures":int(
            (all_trials.v1_observer_decision_agrees &
             all_trials.v2_observer_decision_agrees).sum()
        ),
        "benign_control_false_witnesses_tcpdump":int(all_controls.tcpdump_false_bursts.sum()),
        "benign_control_false_witnesses_tshark":int(all_controls.tshark_false_bursts.sum()),
        "all_orig_inrun_live_hmac_verified":all(p["original_live_hmac_validated"] for p in provenance),
        "after_validation_report_rows_excluded":sum(
            int(p["postverification_unvalidated_rows_excluded"]) for p in provenance
        ),
        "independent_parser":"separate Python standard-library CSV parser, separate tshark live pcapng capture",
        "not_independent_dimensions":[
            "Both observers tap the same Linux loopback interface in one VM.",
            "Both capture files are decoded by the tshark HTTP2 dissector.",
            "Open5GS package family and KIE v1/v2 code are shared with prior runs.",
            "GitHub-hosted infrastructure and one research group's orchestration are shared.",
            "Only two independent VMs per cadence; no powered population-level significance.",
            "Controlled local signed telemetry masking is not a compromised production NF.",
        ],
    }
    (output/"independent_validation_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2))
    print(pd.DataFrame(metrics).to_csv(index=False))
    print("INDIVIDUAL_NEGATIVE_CONTROLS")
    print(all_controls.to_csv(index=False))
    return summary


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input-root",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    args=p.parse_args()
    aggregate(args.input_root,args.output_dir)


if __name__=="__main__":
    main()
