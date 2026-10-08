from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from audit_open5gs_event_boundaries import assign_using_packet_evidence


def _intervals():
    return pd.DataFrame([
        {"start_ts":80.,"end_ts":95.,"phase":"stable_recovery","cycle":1},
        {"start_ts":100.,"end_ts":106.,"phase":"hidden_nrf_burst","cycle":2},
        {"start_ts":106.,"end_ts":120.,"phase":"washout","cycle":2},
        {"start_ts":140.,"end_ts":147.,"phase":"truthful_nrf_burst","cycle":3},
    ])


def _event(t, old_phase="unlabeled", nf="nrf"):
    return pd.Series({"ts":t,"nf":nf,"phase":old_phase,"cycle":-1})


def test_window_center_precedes_phase_but_raw_requests_are_inside():
    # Original point-timestamp classification was "unlabeled".
    packets=np.array([100.05,100.1,100.12,100.14,100.16,100.19,100.24])
    result=assign_using_packet_evidence(_event(99.4),_intervals(),packets)
    assert result["audit_phase"]=="hidden_nrf_burst"
    assert result["audit_status"]=="packet_verified_boundary_relabel"
    assert result["raw_requests"]==len(packets)
    assert result["phase_fraction"]==1.0
    assert result["audit_cycle"]==2
    assert result["timestamp_delta_to_phase_start_s"]<0


def test_truthful_boundary_never_assigned_hidden():
    packets=np.array([140.05,140.1,140.2,140.3,140.35,140.5])
    result=assign_using_packet_evidence(_event(139.7),_intervals(),packets)
    assert result["audit_phase"]=="truthful_nrf_burst"
    assert result["audit_cycle"]==3


def test_split_packet_evidence_remains_unresolved():
    intervals=pd.DataFrame([
        {"start_ts":100.,"end_ts":101.,"phase":"truthful_nrf_burst","cycle":1},
        {"start_ts":101.,"end_ts":102.,"phase":"hidden_nrf_burst","cycle":1},
    ])
    packets=np.array([100.5]*5+[101.5]*5)
    result=assign_using_packet_evidence(_event(101.),intervals,packets)
    assert result["audit_phase"]=="unresolved"
    assert result["audit_status"]=="ambiguous_multi_phase_window"


def test_benign_window_is_not_relabelled_as_truthful_intervention():
    packets=np.array([90.0,90.1,90.2,90.3,90.4,90.5])
    result=assign_using_packet_evidence(_event(90.0),_intervals(),packets)
    assert result["audit_phase"]=="stable_recovery"
    assert result["audit_status"]=="independent_background_control"


def test_leadin_without_phase_interval_is_not_forced_into_nearest():
    packets=np.array([98.6,98.7,98.8,98.9,99.0,99.1])
    result=assign_using_packet_evidence(_event(99),_intervals(),packets)
    assert result["audit_phase"]=="unresolved"
    assert result["audit_status"]=="unassigned_outside_intervals"


def test_too_few_raw_packets_cannot_reclassify():
    packets=np.array([100.15,100.2,100.3])
    result=assign_using_packet_evidence(_event(99.4),_intervals(),packets)
    assert result["audit_phase"]=="unresolved"
    assert result["audit_status"]=="insufficient_raw_requests"


def test_non_nrf_journal_events_keep_exact_timestamps():
    event=pd.Series({"ts":101.,"nf":"udm","phase":"truthful_udm_restart","cycle":2})
    out=assign_using_packet_evidence(event,_intervals(),np.array([]))
    assert out["audit_status"]=="journal_event_exact_timestamp"
    assert out["audit_phase"]=="truthful_udm_restart"
