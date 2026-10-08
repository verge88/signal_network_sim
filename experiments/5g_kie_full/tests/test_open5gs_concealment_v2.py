from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from run_open5gs_concealment import evaluate_one_event
from run_open5gs_concealment_v2 import score_event_v2


def sample_kie(cpu_by_time: dict[float,float], valid: bool=True) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "nf":"nrf","ts":t,"cpu_delta":cpu,"valid_signature":valid,
            "pid":999,"generation":0,"invocation":"instance",
        }
        for t,cpu in sorted(cpu_by_time.items())
    ])


def event(t:float) -> dict:
    return {
        "ts":t,"nf":"nrf","external_event":"nrf_query_burst",
        "nf_instance_id":None
    }


def test_quantized_cpu_report_just_after_legacy_cutoff_is_acknowledged():
    # Reproduces actual failure shape in prospective runner 6113:
    # t+3.177s contains a real +2 CPU tick sample, missed by the old +3s.
    kie=sample_kie({
        -6.833:0,-4.831:0,-2.829:0,
        -0.827:0,1.175:0,3.177:2,5.179:0,
    })
    old=evaluate_one_event(event(0.),kie)
    new=score_event_v2(event(0.),kie)
    assert old["concealment_alarm"]
    assert not new["concealment_alarm"]
    assert new["state"]=="acknowledged"
    assert abs(new["first_positive_kie_offset_s"]-3.177)<1e-6


def test_hidden_report_remains_unacknowledged_with_extended_window():
    kie=sample_kie({
        -6.7:0,-4.7:0,-2.7:0,
        -0.7:0,1.3:0,3.3:0,5.3:9,
    })
    new=score_event_v2(event(0.),kie)
    assert new["concealment_alarm"]
    assert new["state"]=="unacknowledged"
    assert new["post_reports"]==3


def test_valid_signature_required_before_concealment_assessment():
    kie=sample_kie({
        -6.:0,-4.:0,-2.5:0,-.5:0,1.5:0,3.5:2
    },valid=False)
    new=score_event_v2(event(0.),kie)
    assert new["state"]=="invalid_signature"
    assert new["integrity_alarm"]
    assert not new["concealment_alarm"]


def test_no_post_report_is_unknown_not_concealment():
    kie=sample_kie({-6.:0,-4.:0,-2.5:0})
    new=score_event_v2(event(0.),kie)
    assert new["state"]=="insufficient_reports"
    assert not new["concealment_alarm"]


def test_udm_rule_unchanged():
    kie=pd.DataFrame([
        {"nf":"udm","ts":t,"pid":pid,"generation":gen,
         "invocation":f"id{pid}","valid_signature":True,"cpu_delta":0.}
        for t,pid,gen in [(97,1,0),(99,1,0),(101,3,1),(103,3,1)]
    ])
    one={
        "ts":100.0,"nf":"udm","external_event":"nrf_udm_association",
        "nf_instance_id":"uuid"
    }
    assert evaluate_one_event(one,kie)["concealment_alarm"] == score_event_v2(one,kie)["concealment_alarm"]


def test_phase_or_fault_markers_do_not_enter_detector_inputs():
    kie=sample_kie({
        -6.5:0,-4.5:0,-2.5:0,
        -.5:0,1.5:0,3.5:1
    })
    before=score_event_v2(event(0.),kie)
    altered=kie.assign(phase_hint="hidden_nrf_burst",fault_injection="baseline_report_mask")
    after=score_event_v2(event(0.),altered)
    assert before["concealment_alarm"]==after["concealment_alarm"]
