"""Scientific contract for iteration 4: exact packet-onset event evidence."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from reevaluate_open5gs_packet_onset_kie import (
    packet_onset_candidates,score_frozen,run,MIN_GET_IN_WINDOW,GAP_S,
)
from run_open5gs_concealment import evaluate_one_event
from run_open5gs_concealment_v2 import score_event_v2


def requests_for(*starts:float,n:int=30):
    rows=[]
    for i,start in enumerate(starts):
        for j in range(n):
            rows.append({"ts":start+j*0.03,"request_key":f"{i}|{j}"})
    return pd.DataFrame(rows)


def reports_for(t:float,rise_after:bool=True):
    times=np.arange(t-10,t+6,.5)
    return pd.DataFrame([{
        "ts":float(x),"nf":"nrf","valid_signature":True,
        "cpu_delta":float(2 if rise_after and x>=t else 0),
        "pid":1,"generation":1,"invocation":"i",
    } for x in times])


def test_original_packet_witness_gap_is_blind_to_all_oracle_columns():
    packet=requests_for(100.,135.,170.)
    packet["phase"]="hidden_nrf_burst"
    packet["masking_active"]=True
    packet["experimental_cycle"]=99
    result=packet_onset_candidates(packet)
    assert result.ts.tolist()==pytest.approx([100.,135.,170.])
    assert all(result.peak_sliding_2s.eq(30))
    assert "masking_active" not in result
    assert "experimental_cycle" not in result


def test_fixed_boundary_alias_is_recovered_using_packet_first_timestamp():
    start=101.6
    inp=requests_for(start)
    events=packet_onset_candidates(inp)
    assert len(events)==1
    assert events.iloc[0].ts==start
    assert events.iloc[0].raw_get_count==30
    assert MIN_GET_IN_WINDOW==20
    assert GAP_S==4.


def test_below_threshold_not_promoted_by_oracle_label():
    inp=requests_for(100.,n=19)
    inp["phase"]="hidden_nrf_burst"
    assert packet_onset_candidates(inp).empty


def test_duplicate_http2_request_id_rejected_fail_closed():
    inp=requests_for(100.,n=30)
    inp.loc[1,"request_key"]=inp.loc[0,"request_key"]
    with pytest.raises(ValueError,match="duplicate"):
        packet_onset_candidates(inp)


def test_frozen_cpu_tick_v1_v2_rules_are_same_function_calls():
    candidates=packet_onset_candidates(requests_for(100.))
    reports=reports_for(100.)
    scored=score_frozen(candidates,reports)
    raw={"ts":100.,"nf":"nrf",
         "external_event":"nrf_query_burst","nf_instance_id":None}
    v1=evaluate_one_event(raw,reports)
    v2=score_event_v2(raw,reports)
    assert bool(scored.iloc[0].v1_alarm)==bool(v1["concealment_alarm"])
    assert bool(scored.iloc[0].v2_alarm)==bool(v2["concealment_alarm"])
    assert scored.iloc[0].v1_pre_reports==v1["pre_reports"]
    assert scored.iloc[0].v2_post_reports==v2["post_reports"]
    assert not scored.iloc[0].v1_alarm
    assert not scored.iloc[0].v2_alarm


def test_without_cpu_response_both_frozen_detectors_accuse():
    s=score_frozen(packet_onset_candidates(requests_for(100.)),
                   reports_for(100.,rise_after=False))
    assert bool(s.iloc[0].v1_alarm)
    assert bool(s.iloc[0].v2_alarm)
    assert s.iloc[0].v1_state=="unacknowledged"


def test_unknown_report_coverage_not_false_accusation():
    cases=packet_onset_candidates(requests_for(100.))
    reports=reports_for(100.)
    reports=reports[reports.ts>=99.]
    scored=score_frozen(cases,reports)
    assert scored.iloc[0].v1_state=="insufficient_reports"
    assert not scored.iloc[0].v1_known
    assert not scored.iloc[0].v1_alarm
    assert scored.iloc[0].v2_state=="insufficient_reports"


def test_offline_archive_absence_fails_closed(tmp_path):
    with pytest.raises(ValueError,match="exactly one independent artifact"):
        run(tmp_path,tmp_path/"output")
