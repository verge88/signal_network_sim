from __future__ import annotations
import json
from pathlib import Path
import sys

import pandas as pd
import pytest

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

from run_open5gs_stress import profile_plan, PROFILES
from aggregate_open5gs_stress import collect


@pytest.mark.parametrize("seed",[13007,14009,15013,16001])
def test_balanced_frozen_short_standard_profiles(seed):
    first=profile_plan(seed,6)
    assert first==profile_plan(seed,6)
    assert sorted(first)==list(range(1,7))
    assert list(first.values()).count("short")==3
    assert list(first.values()).count("standard")==3
    assert PROFILES["short"]["post_hold_s"]<PROFILES["standard"]["post_hold_s"]

def test_odd_cycles_rejected():
    with pytest.raises(ValueError):profile_plan(1,5)

def _fake_runner(root:Path,seed:int,period:float,load:float) -> None:
    dir=root/f"5g-open5gs-stress-{seed}"
    dir.mkdir(parents=True)
    (dir/"stress_manifest.json").write_text(json.dumps({
        "seed":seed,"kie_period_s":period,"background_rps":load,"cycles":1
    }),encoding="utf-8")
    (dir/"kie_report_validation.json").write_text(json.dumps({
        "all_signatures_valid":True,"invalid_signature_count":0,
        "report_count":10
    }),encoding="utf-8")
    vals=[
        (100.,"nrf","nrf_query_burst","truthful_nrf_burst",False,False),
        (140.,"nrf","nrf_query_burst","hidden_nrf_burst",True,True),
    ]
    pd.DataFrame([
        {"ts":ts,"nf":nf,"external_event":etype,
         "state":"acknowledged" if not alarm else "unacknowledged",
         "concealment_alarm":alarm,"integrity_alarm":False}
        for ts,nf,etype,phase,alarm,_ in vals
    ]).to_csv(dir/"concealment_events.csv",index=False)
    pd.DataFrame([
        {"ts":ts,"nf":nf,"external_event":etype,
         "state":"acknowledged" if not v2 else "unacknowledged",
         "v1_alarm":alarm,"v2_alarm":v2,
         "audit_phase":phase,"audit_cycle":1,"legacy_unlabeled":False}
        for ts,nf,etype,phase,alarm,v2 in vals
    ]).to_csv(dir/"v2_concealment_events.csv",index=False)
    pd.DataFrame([
        {"start_ts":99.,"end_ts":107.,"phase":"truthful_nrf_burst","cycle":1},
        {"start_ts":139.,"end_ts":144.,"phase":"hidden_nrf_burst","cycle":1},
    ]).to_csv(dir/"phase_intervals.csv",index=False)
    pd.DataFrame([
        {"phase":"truthful_nrf_burst","cycle":1,"profile":"short",
         "requested":30,"successful":30},
        {"phase":"hidden_nrf_burst","cycle":1,"profile":"short",
         "requested":30,"successful":30},
    ]).to_csv(dir/"burst_profiles.csv",index=False)
    with (dir/"kie_ground_truth.jsonl").open("w") as f:
        for t,mask in [(100.2,False),(140.3,True)]:
            f.write(json.dumps({"ts":t,"nf_id":"nrf","masking_active":mask})+"\n")
    if load:
        pd.DataFrame([{"ts":99.5,"curl_exit_code":0,"duration_s":0.1}]
        ).to_csv(dir/"nrf_background_load.csv",index=False)

def test_stress_aggregator_keeps_distinct_models_and_exposure(tmp_path:Path):
    source=tmp_path/"src"
    _fake_runner(source,13007,1.0,0.0)
    _fake_runner(source,14009,4.0,2.0)
    # Aggregate accepts its trial-specific seed/condition dictionary;
    # expected 4-phase completion in real CI is independently recorded.
    # Synthetic fixture provides only two profiles to exercise the join,
    # and must fail closed until all four planned scenario phases exist.
    with pytest.raises(ValueError,match="incomplete planned protocol"):
        collect(source,tmp_path/"out",{13007:(1.0,0.0),14009:(4.0,2.0)})

def test_wrong_condition_never_silently_aggregated(tmp_path:Path):
    source=tmp_path/"source"
    _fake_runner(source,13007,1.0,0.0)
    with pytest.raises(ValueError,match="frozen stress factor mismatch"):
        collect(source,tmp_path/"out",{13007:(4.0,0.0)})
