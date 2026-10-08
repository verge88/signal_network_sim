from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from open5gs_kie_sidecar import sign_report
from run_open5gs_concealment import (
    evaluate_one_event,
    load_signed_reports,
    nrf_query_events,
    parse_nrf_journal,
    _assign_phase,
)


def _signed(ts,nf,pid,gen,cpu,secret,phase_hint="baseline",fault="none"):
    payload={
        "schema":"kie-open5gs-v1",
        "ts":float(ts),
        "sequence":int(ts),
        "nonce":f"nonce{int(ts)}",
        "nf_id":nf,
        "unit":f"open5gs-{nf}d.service",
        "phase_hint":phase_hint,
        "fault_injection":fault,
        "reported":{
            "pid":pid,"restart_generation":gen,
            "invocation_id":f"inv{pid}",
            "service_uptime_s":50.0,
            "cpu_ticks_delta":cpu,
        },
    }
    return {**payload,"signature":sign_report(payload,secret)}


def test_nrf_journal_parses_nf_instance_identity(tmp_path:Path):
    path=tmp_path/"nrf.jsonl"
    path.write_text(json.dumps({
        "__REALTIME_TIMESTAMP":"1760000000000000",
        "MESSAGE":"10/09 09:00:00.000: [sbi] INFO: [UDM] NFInstance associated [abcde123-0000-4444-8888-123456789abc]"
    })+"\n",encoding="utf-8")
    observed=parse_nrf_journal(path,1760000000.)
    assert len(observed)==1
    assert observed.iloc[0]["nf"]=="udm"
    assert observed.iloc[0]["nf_instance_id"]=="abcde123-0000-4444-8888-123456789abc"


def test_hidden_udm_accused_truthful_restart_not_accused(tmp_path:Path):
    secret=b"test-key"
    records=[]
    for t,pid,gen in [(96,5,0),(98,5,0),(100,9,1),(102,9,1),(104,9,1)]:
        records.append(_signed(t,"udm",pid,gen,0,secret))
    for t,pid,gen in [(196,9,1),(198,9,1),(200,9,1),(202,9,1),(204,9,1)]:
        records.append(_signed(t,"udm",pid,gen,0,secret))
    path=tmp_path/"reports.jsonl"
    path.write_text("".join(json.dumps(r)+"\n" for r in records),encoding="utf-8")
    reports=load_signed_reports(path,secret)
    truthful=evaluate_one_event({
        "ts":99.,"nf":"udm","external_event":"nrf_udm_association","nf_instance_id":"uuid1"
    },reports)
    hidden=evaluate_one_event({
        "ts":199.,"nf":"udm","external_event":"nrf_udm_association","nf_instance_id":"uuid2"
    },reports)
    assert truthful["state"]=="acknowledged"
    assert not truthful["concealment_alarm"]
    assert hidden["state"]=="unacknowledged"
    assert hidden["concealment_alarm"]


def test_nrf_burst_response_distinguishes_hidden(tmp_path:Path):
    secret=b"local"
    records=[]
    for offset,cpu in [(-6,0),(-4,0),(-1,0),(1,3),(3,1)]:
        records.append(_signed(100+offset,"nrf",1,0,cpu,secret))
    for offset,cpu in [(-6,0),(-4,0),(-1,0),(1,0),(3,0)]:
        records.append(_signed(200+offset,"nrf",1,0,cpu,secret,
                               fault="baseline_report_mask"))
    path=tmp_path/"kie.jsonl"
    path.write_text("".join(json.dumps(r)+"\n" for r in records))
    reports=load_signed_reports(path,secret)
    for t,expected in [(100.,False),(200.,True)]:
        got=evaluate_one_event({
            "ts":t,"nf":"nrf","external_event":"nrf_query_burst","nf_instance_id":None
        },reports)
        assert got["state"] in {"acknowledged","unacknowledged"}
        assert got["concealment_alarm"]==expected


def test_phase_and_fault_markers_not_used_as_detector_inputs(tmp_path:Path):
    secret=b"test"
    records=[
        _signed(t,"udm",10,1,0,secret,phase_hint="hidden_udm_restart",
                fault="baseline_report_mask")
        for t in (96,98,100,102,104)
    ]
    path=tmp_path/"kie.jsonl"
    path.write_text("".join(json.dumps(r)+"\n" for r in records))
    before=evaluate_one_event(
        {"ts":99.,"nf":"udm","external_event":"nrf_udm_association","nf_instance_id":"x"},
        load_signed_reports(path,secret)
    )
    for r in records:
        r["phase_hint"]="truthful_udm_restart"
        r["fault_injection"]="none"
        r["signature"]=sign_report({k:v for k,v in r.items() if k!="signature"},secret)
    path.write_text("".join(json.dumps(r)+"\n" for r in records))
    after=evaluate_one_event(
        {"ts":99.,"nf":"udm","external_event":"nrf_udm_association","nf_instance_id":"x"},
        load_signed_reports(path,secret)
    )
    assert before["concealment_alarm"]==after["concealment_alarm"]


def test_bad_signature_is_integrity_not_concealment(tmp_path:Path):
    secret=b"test"
    rows=[_signed(t,"udm",1,0,0,secret) for t in (96,98,100,102)]
    rows[2]["reported"]["pid"]=900
    path=tmp_path/"invalid.jsonl"
    path.write_text("".join(json.dumps(r)+"\n" for r in rows))
    r=evaluate_one_event({
        "ts":99.,"nf":"udm","external_event":"nrf_udm_association","nf_instance_id":"x"
    },load_signed_reports(path,secret))
    assert r["state"]=="invalid_signature"
    assert r["integrity_alarm"]
    assert not r["concealment_alarm"]


def test_nrf_query_extraction_never_needs_labels():
    df=pd.DataFrame([
        {"server_nf":"nrf","window_start":0.,"nrf_query_rate":0.},
        {"server_nf":"nrf","window_start":2.,"nrf_query_rate":12.},
        {"server_nf":"nrf","window_start":4.,"nrf_query_rate":14.},
        {"server_nf":"nrf","window_start":40.,"nrf_query_rate":15.},
        {"server_nf":"udm","window_start":42.,"nrf_query_rate":60.},
    ])
    events=nrf_query_events(df)
    assert len(events)==2
    assert events["ts"].tolist()==[3.,41.]
