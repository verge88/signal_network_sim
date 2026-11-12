import copy
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from experiments.distributed_5g.temporal_graph import TemporalGraphDetector
from experiments.distributed_5g.temporal_graph_v2 import LocalSynchronyTransformer, SupervisedTemporalGraphDetector, FEATURES as V2_FEATURES
from experiments.robustness_5g.design import attack_schedule, s5_members, s5_count


def frame(windows=16, run_id="normal"):
    rows=[]
    for w in range(windows):
        for slot in range(32):
            count=2 + ((w + slot * 3) % 5)
            rows.append(dict(
                run_id=run_id, window_id=w, nf_id=f"slot-{slot}", timestamp=float(w),
                request_count=float(count),
                interarrival_mean=1.0/max(1,count),
                latency_ms=1.0 + (slot % 4) * .1,
                error_rate=0.0,
            ))
    return pd.DataFrame(rows)


def test_temporal_graph_batch_matches_online_window_scoring():
    train=frame(24, "train")
    detector=TemporalGraphDetector(seed=123, max_samples=128).fit(train)

    batch=detector.clean_copy()
    batch_score=batch.decision_function(frame(18, "test"))

    online=detector.clean_copy()
    parts=[]
    test=frame(18, "test")
    for _, part in test.groupby("window_id", sort=False):
        parts.append(online.decision_function(part))
    online_score=np.concatenate(parts)
    assert np.allclose(batch_score, online_score, rtol=0, atol=1e-12)


def test_s5_is_multi_node_non_overlapping_preregistered_lab_scenario():
    schedule=attack_schedule(3600, 89901)
    s5=[entry for entry in schedule if entry[2] == "S5"]
    assert len(s5) == 4
    assert all(end-start == 30 for start,end,*_ in s5)
    for entry in s5:
        members=s5_members(entry)
        assert len(members) == 3
        assert len({m//8 for m in members}) == 1
        for w in range(entry[0], entry[1]):
            counts=[s5_count(entry,w,"reference",m) for m in members]
            assert all(2 <= value <= 5 for value in counts)
    for i,left in enumerate(schedule):
        for right in schedule[i+1:]:
            assert left[1] <= right[0] or right[1] <= left[0]


def _graph_pattern(localized):
    rows=[]
    for w in range(20):
        global_wave=(w//3)%4
        for slot in range(32):
            if localized and slot in (1,2,3):
                count=3+((w//2)%3)
            elif localized:
                count=2+((w*7+slot*5)%5)
            else:
                # Legitimate whole-zone/global synchrony: every NF follows the same cycle.
                count=3+global_wave
            rows.append(dict(run_id="pattern",window_id=w,nf_id=f"slot-{slot}",
                             timestamp=float(w),request_count=float(count)))
    return pd.DataFrame(rows)


def test_invariant_graph_distinguishes_localized_from_global_synchrony():
    transformer=LocalSynchronyTransformer(history=8)
    global_features=transformer.transform(_graph_pattern(False))
    local_features=transformer.transform(_graph_pattern(True))
    names={name:i for i,name in enumerate(V2_FEATURES)}
    # Compare the final 10 windows of the three coordinated nodes.
    idx=[]
    for w in range(10,20):
        for slot in (1,2,3):
            idx.append(w*32+slot)
    global_contrast=float(np.mean(global_features[idx,names["localized_corr_contrast"]]))
    local_contrast=float(np.mean(local_features[idx,names["localized_corr_contrast"]]))
    global_triangle=float(np.mean(global_features[idx,names["triangle_contrast"]]))
    local_triangle=float(np.mean(local_features[idx,names["triangle_contrast"]]))
    assert local_contrast > global_contrast + .20
    assert local_triangle > global_triangle + .20


def test_v2_transform_batch_matches_online_windows():
    test=frame(18,"v2-test")
    batch=LocalSynchronyTransformer(history=8).transform(test)
    online=LocalSynchronyTransformer(history=8)
    parts=[]
    for _,part in test.groupby("window_id",sort=False):
        parts.append(online.transform_online(part))
    assert np.allclose(batch,np.vstack(parts),rtol=0,atol=1e-12)


def test_v2_supervised_detector_roundtrips_with_clean_state():
    dev=frame(24,"dev")
    labels=pd.DataFrame({
        "attack":np.zeros(len(dev),dtype=int),
        "scenario":["N1"]*len(dev),
    })
    mask=dev.window_id.between(10,19) & dev.nf_id.isin(["slot-1","slot-2","slot-3"])
    labels.loc[mask,"attack"]=1
    labels.loc[mask,"scenario"]="S5"
    detector=SupervisedTemporalGraphDetector(seed=123,history=8)
    detector.fit_development([(dev,labels)])

    test=frame(16,"heldout")
    batch=detector.clean_copy().decision_function(test)
    online=detector.clean_copy()
    parts=[]
    for _,part in test.groupby("window_id",sort=False):
        parts.append(online.decision_function(part))
    assert np.allclose(batch,np.concatenate(parts),rtol=0,atol=1e-12)

    with tempfile.TemporaryDirectory() as directory:
        path=Path(directory)/"model.joblib"
        clean=detector.clean_copy()
        joblib.dump(clean,path)
        loaded=joblib.load(path)
        loaded.reset()
        assert np.allclose(batch,loaded.decision_function(test),rtol=0,atol=1e-12)
