from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from run_open5gs_semantic_kie import (
    _upper_p,baseline_split,semantic_signals,
)


def _synthetic_joined(n: int=430) -> pd.DataFrame:
    rng=np.random.default_rng(714)
    rows=[]
    for i in range(n):
        is_attack=i>=400
        hidden=is_attack and i<415
        external=1.0+(.1*rng.normal())+(4.5 if is_attack else 0)
        report=1.0+(.1*rng.normal())+(0.0 if hidden else (4.4 if is_attack else 0))
        rows.append({
            "window_id":i,"window_start":i*2.0,
            "phase":"baseline" if i<400 else (
                "hidden_nrf_burst" if hidden else "truthful_nrf_burst"),
            "cycle":0 if i<400 else 1,
            "nrf_register_rate":0.0,"nrf_delete_rate":0.0,
            "ext_nrf_nrf_register_rate":0.0,
            "ext_nrf_nrf_delete_rate":0.0,
            "kie_nrf_external_activity_raw":external,
            "kie_nrf_report_activity_raw":report,
            "kie_nrf_signature_valid":True,
            "kie_nrf_reported_generation_delta":0.0,
            "kie_udm_external_activity_raw":.7+.02*rng.normal(),
            "kie_udm_report_activity_raw":.7+.02*rng.normal(),
            "kie_udm_signature_valid":True,
            "kie_udm_reported_generation_delta":0.0,
            "fault_injection":"baseline_report_mask" if hidden else "none",
        })
    return pd.DataFrame(rows)


def test_directional_gap_distinguishes_hidden_from_truthful():
    frame=_synthetic_joined()
    split=baseline_split(frame)
    scored=semantic_signals(frame,split["train"])
    hidden=scored.loc[400:414,"nrf_directional_gap"].mean()
    truthful=scored.loc[420:429,"nrf_directional_gap"].mean()
    assert hidden>truthful+3.0


def test_fault_marker_and_phase_labels_never_affect_scores():
    frame=_synthetic_joined()
    base=semantic_signals(frame,baseline_split(frame)["train"])
    changed=frame.copy()
    changed.loc[changed["phase"]!="baseline","fault_injection"]="none"
    changed.loc[changed["phase"]!="baseline","cycle"]=777
    newer=semantic_signals(changed,baseline_split(changed)["train"])
    assert np.allclose(base["semantic_raw"],newer["semantic_raw"])
    assert np.array_equal(base["signature_invalid"],newer["signature_invalid"])


def test_signature_tamper_is_separate_integrity_alarm():
    frame=_synthetic_joined()
    frame.loc[401,"kie_nrf_signature_valid"]=False
    scored=semantic_signals(frame,baseline_split(frame)["train"])
    assert bool(scored.loc[401,"signature_invalid"])
    assert not bool(scored.loc[400,"signature_invalid"])


def test_empirical_conformal_resolution_and_tie_handling():
    ref=np.arange(119,dtype=float)
    p=_upper_p(np.array([0.,118.,119.,1000.]),ref)
    assert p[0]==1.0
    assert p[2]==1/120
    assert p[3]==1/120
    assert p[1]==2/120


def test_rolling_temporal_signals_are_causal():
    frame=_synthetic_joined()
    train=baseline_split(frame)["train"]
    first=semantic_signals(frame,train)
    modified=frame.copy()
    modified.loc[426:429,"kie_nrf_external_activity_raw"]=1000
    second=semantic_signals(modified,train)
    assert np.allclose(
        first.loc[:425,"semantic_raw"],
        second.loc[:425,"semantic_raw"],
    )
