from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from aggregate_open5gs_concealment_v2 import pool


def test_aggregate_distinguishes_v1_and_v2_false_accusations(tmp_path:Path):
    root=tmp_path/"inputs"
    for seed in (11,13):
        folder=root/f"5g-open5gs-concealment-v2-{seed}"
        folder.mkdir(parents=True)
        pd.DataFrame([
            {"seed":seed,"audit_phase":"truthful_nrf_burst",
             "eligible_as_confirmed_exposure":True,"state":"acknowledged",
             "v1_alarm":True,"v2_alarm":False,"alarm_changed":True,
             "legacy_unlabeled":False},
            {"seed":seed,"audit_phase":"hidden_nrf_burst",
             "eligible_as_confirmed_exposure":True,"state":"unacknowledged",
             "v1_alarm":True,"v2_alarm":True,"alarm_changed":False,
             "legacy_unlabeled":False},
        ]).to_csv(folder/"v2_concealment_events.csv",index=False)
    result=pool(root,tmp_path/"out",{11,13})
    truthful=next(x for x in result["phase_metrics"] if x["phase"]=="truthful_nrf_burst")
    hidden=next(x for x in result["phase_metrics"] if x["phase"]=="hidden_nrf_burst")
    assert truthful["v1_alarms"]==2
    assert truthful["v2_alarms"]==0
    assert hidden["v2_alarms"]==2
    assert result["alarms_changed_v1_to_v2"]==2
    assert result["unresolved_events"]==0


def test_missing_seed_causes_aggregation_failure(tmp_path:Path):
    with pytest.raises(ValueError,match="Expected one complete runner"):
        pool(tmp_path,tmp_path/"out",{6113,7247,8309})
