"""Contracts for causal V11: no lookahead, no identity dependence, safe state."""
import pytest
from experiments.distributed_5g.v11_staircase import (
    CausalStaircaseDetector, EPISODE_LENGTH, PERIOD, WAVE)

ZONES = {"a":"x", "b":"x", "c":"x", "d":"x"}

def trace():
    rows = []
    for w in range(48):
        row = {"a":6+w%3, "b":7+w%2, "c":5+w%4, "d":8+w%2}
        if 5 <= w < 5 + EPISODE_LENGTH:
            offset = WAVE[((w-5)//PERIOD) % len(WAVE)]
            row["a"], row["b"] = 2+offset, 3+offset
        rows.append(row)
    return rows

def run(rows,zones=ZONES):
    detector=CausalStaircaseDetector(zones)
    return [detector.step(w,row) for w,row in enumerate(rows)]

def test_prefix_is_causal_and_wave_hits_at_third_window():
    rows=trace()
    out=run(rows)
    assert out[5]["a"]==out[6]["a"]==0
    assert all(out[w]["a"]==out[w]["b"]==1 for w in range(7,35))
    assert all(out[w]["a"]==out[w]["b"]==0 for w in range(35,48))
    modified=rows[:17]+[{key:11 for key in rows[0]} for _ in rows[17:]]
    assert run(modified)[:17]==out[:17]

def test_nf_renaming_within_zone_is_symmetric():
    rows=trace()
    renaming={"a":"z8", "b":"z1", "c":"z4", "d":"z2"}
    other=[{renaming[nf]:value for nf,value in row.items()} for row in rows]
    expected=[{renaming[nf]:alarm for nf,alarm in row.items()} for row in run(rows)]
    assert run(other,{v:"x" for v in renaming.values()})==expected

def test_false_staircase_is_cancelled_on_next_mismatch():
    rows=[{"a":7,"b":8,"c":7,"d":8} for _ in range(12)]
    for w in (1,2,3):
        rows[w]["a"],rows[w]["b"]=2,3
    rows[4]["a"],rows[4]["b"]=3,4
    out=run(rows)
    assert out[3]["a"]==out[4]["a"]==1
    assert out[5]["a"]==0
    assert sum(row["a"] for row in out)==2

def test_missing_nf_and_fractional_count_fail_closed():
    d=CausalStaircaseDetector({"a":0,"b":0})
    with pytest.raises(ValueError,match="coverage"):
        d.step(0,{"a":2})
    with pytest.raises(ValueError,match="Non-integral"):
        d.step(0,{"a":2.5,"b":3})

def test_gap_flushes_pending_state():
    d=CausalStaircaseDetector({"a":0,"b":0})
    for w in range(3):
        d.step(w,{"a":2,"b":3})
    assert all(v==0 for v in d.step(9,{"a":3,"b":4}).values())

def test_cross_zone_pairs_are_never_grouped():
    d=CausalStaircaseDetector({"a":"x","b":"y"})
    for w in range(10):
        assert all(v==0 for v in d.step(w,{"a":2,"b":3}).values())
