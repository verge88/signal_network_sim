"""Event-conditioned Open5GS concealment detector.

Inputs used for decisions:
 * independent NRF daemon journal: UDM registration event with NF instance UUID
 * passive HTTP/2 NRF request rate: externally witnessed query burst
 * signed local UDM/NRF KIE process identity and resource deltas

Labels, ground truth, phase_hint and fault_injection are NEVER decision inputs.
The detector only issues concealment accusations when independent event evidence
is present and a verified matching KIE report fails to acknowledge it.

Finite-lag evaluation: UDM t-4..t+6 s; NRF t-8..t+3 s.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re

import numpy as np
import pandas as pd

from open5gs_kie_sidecar import verify_report


UDM_ASSOC = re.compile(
    r"\[UDM\]\s+NFInstance associated \[([a-fA-F0-9-]{16,})\]"
)
TIMESTAMP = re.compile(
    r"(\d\d/\d\d)\s+(\d\d:\d\d:\d\d\.\d+)"
)


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_signed_reports(path: Path, secret: bytes | None) -> pd.DataFrame:
    """No experimental labels/flags or collector ground truth are read."""
    rows = []
    for report in read_jsonl(path):
        payload = report.get("reported", {})
        verified = verify_report(report, secret) if secret is not None else True
        rows.append({
            "ts":float(report["ts"]),
            "nf":str(report["nf_id"]).lower(),
            "valid_signature":bool(verified),
            "pid":int(payload.get("pid",0) or 0),
            "generation":int(payload.get("restart_generation",0) or 0),
            "invocation":str(payload.get("invocation_id","")),
            "cpu_delta":float(payload.get("cpu_ticks_delta",0) or 0),
        })
    if not rows:
        raise ValueError("no signed KIE reports")
    df = pd.DataFrame(rows).sort_values(["nf","ts"]).reset_index(drop=True)
    # Signature checking is skipped only for explicit offline/replay mode.
    return df


def parse_nrf_journal(
    path: Path,
    reference_epoch: float,
) -> pd.DataFrame:
    """Parse NRF-side UDM NF-instance events, not UDM's own sidecar."""
    seen: set[tuple[str,float]] = set()
    rows = []
    yr = datetime.fromtimestamp(reference_epoch,timezone.utc).year
    for line in path.read_text(encoding="utf-8",errors="replace").splitlines():
        if not line.strip():
            continue
        if line.lstrip().startswith("{"):
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            # systemd JSON export represents binary/non-UTF8 journal fields as
            # lists of byte values. Open5GS emits ANSI escape codes in MESSAGE,
            # so a plain str(list) loses all searchable event text.
            raw_msg = entry.get("MESSAGE", "")
            if isinstance(raw_msg, list) and all(
                isinstance(v, int) and 0 <= v <= 255 for v in raw_msg
            ):
                msg = bytes(raw_msg).decode("utf-8", errors="replace")
            elif isinstance(raw_msg, str):
                msg = raw_msg
            else:
                continue
            if entry.get("_SYSTEMD_UNIT") not in (None, "open5gs-nrfd.service"):
                continue
            timestamp_us = entry.get("__REALTIME_TIMESTAMP")
            if timestamp_us is None:
                continue
            ts = float(timestamp_us)/1_000_000.0
        else:
            # Reproduceability for older experiment archives recorded as text:
            # require the emitter to be NRF, not UDM/SMF.
            if "open5gs-nrfd[" not in line:
                continue
            msg = line
            match_ts = TIMESTAMP.search(line)
            if match_ts is None:
                continue
            try:
                t = datetime.strptime(
                    f"{yr}/{match_ts.group(1)} {match_ts.group(2)}",
                    "%Y/%m/%d %H:%M:%S.%f",
                ).replace(tzinfo=timezone.utc)
            except ValueError:
                continue
            ts=t.timestamp()
        match = UDM_ASSOC.search(msg)
        if match is None:
            continue
        uuid=match.group(1).lower()
        key=(uuid,round(ts,3))
        if key not in seen:
            seen.add(key)
            rows.append({"ts":ts,"nf":"udm","external_event":"nrf_udm_association","nf_instance_id":uuid})
    if not rows:
        return pd.DataFrame(columns=["ts","nf","external_event","nf_instance_id"])
    return pd.DataFrame(rows).sort_values("ts").reset_index(drop=True)


def nrf_query_events(per_nf_features: pd.DataFrame, min_rps: float=10.0) -> pd.DataFrame:
    """Burst events are selected ONLY from the independently captured SBI trace.

    A two-second bucket is marked when ≥10 HTTP/2 GET/s target NRF. Nearby
    marked buckets become one event (no scenario labels used for selection).
    """
    nrf = per_nf_features.loc[
        (per_nf_features["server_nf"] == "nrf")
        & (pd.to_numeric(per_nf_features["nrf_query_rate"],errors="coerce")>=min_rps)
    ].sort_values("window_start")
    rows=[]
    last_t=-1e30
    for _,r in nrf.iterrows():
        t=float(r["window_start"])+1.0
        if t-last_t > 4.0:
            rows.append({
                "ts":t,"nf":"nrf","external_event":"nrf_query_burst",
                "nf_instance_id":None,
            })
        last_t=t
    return pd.DataFrame(rows,columns=["ts","nf","external_event","nf_instance_id"])


def _is_valid_coverage(pre: pd.DataFrame, post: pd.DataFrame) -> str:
    if pre.empty or post.empty:
        return "insufficient_reports"
    if not bool(pd.concat([pre,post])["valid_signature"].all()):
        return "invalid_signature"
    return "ok"


def evaluate_one_event(event: dict, kie: pd.DataFrame) -> dict:
    t=float(event["ts"])
    nf=str(event["nf"])
    reports=kie[kie["nf"]==nf]
    output=dict(event)
    output.update({
        "state":"unknown",
        "concealment_alarm":False,
        "integrity_alarm":False,
        "report_response":None,
        "pre_reports":0,
        "post_reports":0,
        "report_lag_s":6.0 if nf=="udm" else 3.0,
    })

    if nf=="udm":
        pre=reports[(reports["ts"]>=t-4)&(reports["ts"]<t)]
        post=reports[(reports["ts"]>=t)&(reports["ts"]<=t+6)]
        status=_is_valid_coverage(pre,post)
        output["pre_reports"]=len(pre)
        output["post_reports"]=len(post)
        if status != "ok":
            output["state"]=status
            output["integrity_alarm"]=status=="invalid_signature"
            return output

        combined=pd.concat([pre,post])
        responded = (
            combined["pid"].nunique()>1 or
            combined["generation"].nunique()>1 or
            combined["invocation"].nunique()>1
        )
        output["report_response"]=bool(responded)
        output["state"]="acknowledged" if responded else "unacknowledged"
        output["concealment_alarm"]=not responded
        return output

    # External evidence is NRF traffic, *not* the orchestrator's phase flag.
    # Independently sampled NRF CPU ticks must rise after the burst versus a
    # preceding reference period. Detect at most 3s after the first hot bucket,
    # before report-masking is lifted in the runner.
    pre=reports[(reports["ts"]>=t-8)&(reports["ts"]<t-2)]
    post=reports[(reports["ts"]>=t-1)&(reports["ts"]<=t+3)]
    status=_is_valid_coverage(pre,post)
    output["pre_reports"]=len(pre)
    output["post_reports"]=len(post)
    if status!="ok":
        output["state"]=status
        output["integrity_alarm"]=status=="invalid_signature"
        return output

    rise=float(post["cpu_delta"].max()-pre["cpu_delta"].median())
    output["report_response"]=bool(rise>=1.0)
    output["cpu_response_ticks"]=rise
    output["state"]="acknowledged" if rise>=1.0 else "unacknowledged"
    output["concealment_alarm"]=rise<1.0
    return output


def _assign_phase(t: float, intervals: pd.DataFrame) -> tuple[str,int]:
    hit=intervals[(intervals["start_ts"]<=t)&(t<intervals["end_ts"])]
    if hit.empty:
        return "unlabeled",-1
    row=hit.iloc[0]
    return str(row["phase"]),int(row["cycle"])


def confusion(events: pd.DataFrame) -> list[dict]:
    out=[]
    for (nf,phase), group in events.groupby(["nf","phase"],dropna=False):
        known=group[group["state"].isin(("acknowledged","unacknowledged"))]
        hidden=phase.startswith("hidden_")
        truthful=phase.startswith("truthful_")
        if hidden or truthful:
            metric="hidden_recall" if hidden else "truthful_false_accusation_rate"
            rate=float(known["concealment_alarm"].mean()) if len(known) else None
        else:
            metric="unsolicited_event_alarm_rate"
            rate=float(known["concealment_alarm"].mean()) if len(known) else None
        out.append({
            "nf":nf,"phase":phase,"event_count":int(len(group)),
            "verified_evaluable_events":int(len(known)),
            "concealment_alarms":int(group["concealment_alarm"].sum()),
            "integrity_alarms":int(group["integrity_alarm"].sum()),
            "unknown_events":int(len(group)-len(known)),
            "metric":metric,"rate":rate,
        })
    return out


def run(
    global_features:Path,
    per_nf_features:Path,
    nrf_journal:Path,
    kie_reports:Path,
    intervals_path:Path,
    output_dir:Path,
    secret:bytes|None,
) -> dict:
    global_df=pd.read_csv(global_features)
    per_nf_df=pd.read_csv(per_nf_features)
    intervals=pd.read_csv(intervals_path)
    origin=float(intervals["start_ts"].min())
    end=float(intervals["end_ts"].max())
    kie=load_signed_reports(kie_reports,secret)
    log_events=parse_nrf_journal(nrf_journal,origin)
    query_events=nrf_query_events(per_nf_df)
    observed=pd.concat([log_events,query_events],ignore_index=True)
    observed=observed[(observed["ts"]>=origin)&(observed["ts"]<end)].sort_values("ts")
    if observed.empty:
        raise ValueError("no independent external NRF events within experiment window")

    evaluated=pd.DataFrame([evaluate_one_event(row,kie) for row in observed.to_dict("records")])
    evaluated[["phase","cycle"]]=pd.DataFrame(
        [_assign_phase(float(t),intervals) for t in evaluated["ts"]],
        index=evaluated.index,
    )
    # Keep phases and cycle IDs for EVALUATION only; the detector above never
    # receives any phase, mask indication, sidecar ground truth or label.
    stats=confusion(evaluated)
    output_dir.mkdir(parents=True,exist_ok=True)
    evaluated.to_csv(output_dir/"concealment_events.csv",index=False)
    pd.DataFrame(stats).to_csv(output_dir/"concealment_phase_metrics.csv",index=False)
    per_cycle=[]
    for (nf,phase,cycle),g in evaluated.groupby(["nf","phase","cycle"]):
        if phase.startswith(("truthful_","hidden_")):
            per_cycle.append({
                "nf":nf,"phase":phase,"cycle":int(cycle),
                "external_events":len(g),
                "concealment_detected":bool(g["concealment_alarm"].any()),
                "unverified_events":int((~g["state"].isin(["acknowledged","unacknowledged"])).sum()),
            })
    pd.DataFrame(per_cycle).to_csv(output_dir/"concealment_cycle_metrics.csv",index=False)
    summary={
        "source_counts":{"nrf_registry_journal_udm":len(log_events),
                         "passive_pcap_nrf_burst":len(query_events)},
        "inside_window_events":len(observed),
        "signature_check":"HMAC_verified" if secret is not None else "SKIPPED_OFFLINE_REPLAY",
        "thresholds_predeclared":{
            "nrf_query_burst_min_rps":10.0,
            "nrf_report_cpu_response_ticks":1.0,
            "udm_pre_event_s":4.0,"udm_post_event_s":6.0,
            "nrf_reference_s":8.0,"nrf_post_event_s":3.0,
        },
        "metrics":stats,
        "limitations":[
            "Event-level metrics differ from earlier 2-second window recall.",
            "NRF server journal is independent of UDM sidecar, but shares a host and is not adversarially hardened.",
            "Signature validity authenticates bytes to a shared key; it does not attest sidecar truthfulness.",
            "CPU-tick burst response is a heuristic and can fail with other workload/OS scheduling.",
            "Baseline with zero external events cannot establish a per-window 1% FPR.",
            "Labels/phase and ground truth are used only after decisions for evaluation.",
        ]
    }
    (output_dir/"concealment_summary.json").write_text(
        json.dumps(summary,indent=2,ensure_ascii=False),encoding="utf-8"
    )
    return summary


def main() -> None:
    p=argparse.ArgumentParser(description="External-event-conditioned Open5GS concealment detector")
    p.add_argument("--global-features",type=Path,required=True)
    p.add_argument("--per-nf-features",type=Path,required=True)
    p.add_argument("--nrf-journal",type=Path,required=True)
    p.add_argument("--reports",type=Path,required=True)
    p.add_argument("--intervals",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--offline-no-verify",action="store_true",
                   help="Only for analysis of old artifacts whose one-time HMAC secret was not exported")
    a=p.parse_args()
    secret=os.environ.get("OPEN5GS_KIE_SECRET","")
    if not a.offline_no_verify and not secret:
        raise SystemExit("LIVE HMAC verification required; set OPEN5GS_KIE_SECRET")
    result=run(a.global_features,a.per_nf_features,a.nrf_journal,
               a.reports,a.intervals,a.output_dir,
               secret.encode() if not a.offline_no_verify else None)
    print(json.dumps(result,indent=2,ensure_ascii=False))


if __name__=="__main__":
    main()
