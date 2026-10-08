from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from run_open5gs_randomized_scenarios import (
    SCENARIOS, atomic_control, endpoint, randomized_plan,
)


@pytest.mark.parametrize("seed",[1729,2707,4051])
def test_plan_balanced_per_cycle_and_reproducible(seed):
    one=randomized_plan(seed,6)
    two=randomized_plan(seed,6)
    assert one==two
    assert len(one)==24
    for cycle in range(1,7):
        items=[r["phase"] for r in one if r["cycle"]==cycle]
        assert Counter(items)==Counter(SCENARIOS)
        assert [r["position"] for r in one if r["cycle"]==cycle]==[0,1,2,3]


def test_different_seeds_have_different_orders():
    a=[x["phase"] for x in randomized_plan(1729,6)]
    b=[x["phase"] for x in randomized_plan(2707,6)]
    assert a != b


def test_control_transition_is_atomic_and_truthful_mode_empty(tmp_path:Path):
    path=tmp_path/"kie_control.json"
    atomic_control(path,"hidden_udm_restart","udm")
    assert json.loads(path.read_text())=={
        "phase":"hidden_udm_restart","mask_nfs":["udm"]
    }
    assert not path.with_suffix(".json.tmp").exists()
    atomic_control(path,"truthful_udm_restart",None)
    assert json.loads(path.read_text())["mask_nfs"]==[]


def test_no_reachable_nrf_fails_closed(tmp_path:Path):
    path=tmp_path/"endpoints.json"
    path.write_text(json.dumps({"endpoints":[
        {"nf":"nrf","reachable":False,"address":"127.0.0.10","port":7777}
    ]}))
    with pytest.raises(ValueError,match="no reachable NRF"):
        endpoint(path)


def test_invalid_cycles_rejected():
    with pytest.raises(ValueError):
        randomized_plan(1,0)
