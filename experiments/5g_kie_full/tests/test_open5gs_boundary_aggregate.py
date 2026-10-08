from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd
import pytest

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from aggregate_open5gs_boundary_audit import aggregate, wilson_interval


def test_wilson_uncertainty_nonzero_for_zero_false_accusations():
    lo,hi=wilson_interval(0,18)
    assert lo==0
    assert 0.1 < hi < 0.2
    assert wilson_interval(0,0)==(None,None)


def test_aggregate_excludes_unknowns_and_keeps_background_alarms(tmp_path:Path):
    root=tmp_path/"inputs"
    for seed in (17,19):
        folder=root/str(seed)
        folder.mkdir(parents=True)
        (folder/"boundary_audit_summary.json").write_text(
            json.dumps({"seed":seed}),encoding="utf-8"
        )
        pd.DataFrame([
            {"seed":seed,"nf":"nrf","legacy_unlabeled":True,
             "audit_phase":"hidden_nrf_burst",
             "audit_status":"packet_verified_boundary_relabel",
             "eligible_as_confirmed_exposure":True,
             "concealment_alarm":True},
            {"seed":seed,"nf":"nrf","legacy_unlabeled":True,
             "audit_phase":"unresolved",
             "audit_status":"ambiguous_boundary",
             "eligible_as_confirmed_exposure":False,
             "concealment_alarm":True},
            {"seed":seed,"nf":"udm","legacy_unlabeled":False,
             "audit_phase":"truthful_udm_restart",
             "audit_status":"journal_event_exact_timestamp",
             "eligible_as_confirmed_exposure":True,
             "concealment_alarm":False},
        ]).to_csv(folder/"boundary_audit_events.csv",index=False)
        pd.DataFrame([
            {"seed":seed,"phase":"hidden_nrf_burst","cycle":1,
             "missing_external_witness":False,"fragmented_duplicate_witness":False},
        ]).to_csv(folder/"boundary_intervention_coverage.csv",index=False)
    result=aggregate(root,tmp_path/"out",{17,19})
    diag=result["diagnostics"]
    assert diag["original_unlabeled_nrf"]==4
    assert diag["original_unlabeled_reassigned_to_perturbation"]==2
    assert diag["remaining_unresolved_alarms"]==2
    hidden=next(m for m in result["phase_metrics"] if m["phase"]=="hidden_nrf_burst")
    assert hidden["denominator_evaluable_events"]==2
    assert hidden["concealment_alarms"]==2


def test_missing_runner_is_error_not_silent_coverage_loss(tmp_path:Path):
    with pytest.raises(ValueError,match="missing or extra seeds"):
        aggregate(tmp_path,tmp_path/"out",{1729,2707,4051})
