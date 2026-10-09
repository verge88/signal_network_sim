from __future__ import annotations

import json
import sys
from pathlib import Path
import pandas as pd
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from validate_open5gs_dual_observer import (
    decode_independent_tsv,sliding_peak,blind_bursts,frozen_kie,aggregate,RUNNERS
)
from run_open5gs_validation_controls import run as run_controls


def test_independent_decoder_uses_h2_pseudopath_and_stream_identity(tmp_path):
    ep=tmp_path/"endpoints.json"
    ep.write_text(json.dumps({"endpoints":[{"nf":"nrf","address":"127.0.0.10"}]}))
    tsv=tmp_path/"raw.tsv"
    tsv.write_text(
        "frame.time_epoch\tip.src\tip.dst\ttcp.stream\thttp2.streamid\thttp2.header.name\thttp2.header.value\n"
        "100.01\t127.0.0.1\t127.0.0.10\t1\t3\t:method,:path\tGET,/nnrf-nfm/v1/nf-instances\n"
        "100.02\t127.0.0.1\t127.0.0.10\t1\t3\t:method,:path\tGET,/nnrf-nfm/v1/nf-instances\n"
        "100.03\t127.0.0.1\t127.0.0.10\t1\t5\t:method,:path\tGET,/nnrf-nfm/v1/nf-instances\n"
        "100.04\t127.0.0.1\t127.0.0.10\t2\t1\t:method,:path\tPOST,/nnrf-nfm/v1/nf-instances\n"
        "100.05\t127.0.0.1\t127.0.0.11\t3\t1\t:method,:path\tGET,/nnrf-nfm/v1/nf-instances\n"
    )
    events=decode_independent_tsv(tsv,ep)
    assert len(events)==2
    assert [x["ts"] for x in events]==[100.01,100.03]


def test_sliding_witness_not_origin_locked():
    packets=[
        {"ts":101.55+i*0.04,"key":f"{i}|1"}
        for i in range(30)
    ]
    assert sliding_peak([x["ts"] for x in packets],2.0)==30
    found=blind_bursts(packets)
    assert len(found)==1
    assert found[0]["ts"]==pytest.approx(101.55)


def test_negative_control_not_detected_as_burst():
    packets=[
        {"ts":100.+i*.8,"key":f"{i}|1"}
        for i in range(10)
    ]
    assert sliding_peak([x["ts"] for x in packets])<20
    assert blind_bursts(packets)==[]


def test_separated_events_and_no_oracle():
    records=[]
    for base in (100.,130.):
        for i in range(30):
            records.append({"ts":base+i*.03,"key":f"{base}-{i}","mask":True,"phase":"hidden_nrf_burst"})
    events=blind_bursts(records)
    assert len(events)==2
    assert "mask" not in events[0] and "phase" not in events[0]


def test_frozen_functions_not_retuned():
    r=pd.DataFrame([
        {"ts":float(t),"nf":"nrf","valid_signature":True,
         "cpu_delta":0.0,"pid":1,"generation":1,"invocation":"x"}
        for t in range(90,110)
    ])
    scored=frozen_kie({"ts":100.0,"raw_get_count":30},r)
    assert scored["v1_alarm"] is True and scored["v2_alarm"] is True
    assert scored["v1_state"]=="unacknowledged"


def test_invalid_control_configuration_fails_closed(tmp_path):
    with pytest.raises(ValueError,match="predeclared"):
        run_controls(tmp_path,idle_seconds=1.)
    with pytest.raises(ValueError,match="predeclared"):
        run_controls(tmp_path,benign_requests=20)


def test_six_new_independent_seeds():
    assert len(RUNNERS)==6
    assert sorted(RUNNERS.values())==[1.,1.,2.,2.,4.,4.]
    assert min(RUNNERS)>40000


def test_absent_data_cannot_be_reported_as_validation(tmp_path):
    with pytest.raises(ValueError,match="exactly one independent"):
        aggregate(tmp_path,tmp_path/"out")
