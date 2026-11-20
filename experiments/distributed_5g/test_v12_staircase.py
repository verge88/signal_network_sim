"""V12 scientific contracts: no unconfirmed alarms, no future leakage, no label inputs."""
import pytest

from experiments.distributed_5g.v11_staircase import CausalStaircaseDetector, WAVE, PERIOD
from experiments.distributed_5g.v12_staircase import ConfirmedStaircaseDetector

ZONES = {"a":"x","b":"x","c":"x","d":"x"}

def trace():
    rows=[]
    for w in range(50):
        row={"a":6+w%3,"b":7+w%2,"c":5+w%4,"d":8+w%2}
        if 5 <= w < 35:
            offset=WAVE[((w-5)//PERIOD)%len(WAVE)]
            row["a"],row["b"]=2+offset,3+offset
        rows.append(row)
    return rows

def replay(rows, klass=ConfirmedStaircaseDetector, zones=ZONES):
    d=klass(zones)
    return [d.step(w, row) for w,row in enumerate(rows)]

def test_confirmation_only_recall_exactly_27_of_30():
    alarms=replay(trace())
    assert all(alarms[w]["a"]==alarms[w]["b"]==0 for w in (5,6,7))
    assert all(alarms[w]["a"]==alarms[w]["b"]==1 for w in range(8,35))
    assert sum(v["a"] for v in alarms[5:35]) == 27
    assert all(v["a"]==v["b"]==0 for v in alarms[35:])

def test_v12_is_subset_of_v11_across_arbitrary_prefix():
    rows=trace()
    # Add normal constant windows and a short failed staircase to exercise pending.
    rows += [{"a":2,"b":3,"c":7,"d":8} for _ in range(4)]
    old=replay(rows,klass=CausalStaircaseDetector)
    new=replay(rows)
    assert all(all(v <= old[i][nf] for nf,v in row.items()) for i,row in enumerate(new))

def test_no_provisional_alerts_when_confirmation_fails():
    rows=[{"a":7,"b":8,"c":7,"d":8} for _ in range(15)]
    for w in (1,2,3):
        rows[w]["a"],rows[w]["b"]=2,3
    rows[4]["a"],rows[4]["b"]=2,3  # no +1 at expected next window
    out=replay(rows)
    assert not any(row["a"] or row["b"] for row in out)

def test_no_lookahead_and_nf_permutation_equivariance():
    original=trace()
    future=original[:17]+[{nf:11 for nf in ZONES} for _ in original[17:]]
    assert replay(original)[:17] == replay(future)[:17]
    renamed={"a":"z8","b":"z1","c":"z4","d":"z2"}
    renamed_rows=[{renamed[nf]:v for nf,v in row.items()} for row in original]
    expected=[{renamed[nf]:v for nf,v in row.items()} for row in replay(original)]
    assert replay(renamed_rows,zones={v:"x" for v in renamed.values()}) == expected

def test_no_cross_zone_detection_and_no_unvalidated_input():
    d=ConfirmedStaircaseDetector({"a":"x","b":"y"})
    for w in range(8):
        assert d.step(w,{"a":2,"b":3})=={"a":0,"b":0}
    with pytest.raises(ValueError,match="coverage"):
        d.step(8,{"a":2})
    with pytest.raises(ValueError,match="Non-integral"):
        d.step(8,{"a":2.1,"b":3})

def test_discontinuity_clears_pending():
    d=ConfirmedStaircaseDetector({"a":0,"b":0})
    for w in range(3):
        assert d.step(w,{"a":2,"b":3})=={"a":0,"b":0}
    assert d.step(10,{"a":3,"b":4})=={"a":0,"b":0}
