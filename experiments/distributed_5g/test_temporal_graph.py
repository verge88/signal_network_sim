import copy
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from experiments.distributed_5g.temporal_graph import TemporalGraphDetector
from experiments.distributed_5g.temporal_graph_v2 import LocalSynchronyTransformer, SupervisedTemporalGraphDetector, FEATURES as V2_FEATURES
from experiments.distributed_5g.temporal_graph_v3 import TemporalGraphV3Detector
from experiments.distributed_5g.temporal_graph_v4 import CommonModeResidualTransformer, TemporalGraphV4Detector, FEATURES_V4
from experiments.distributed_5g.temporal_graph_v5 import CyclicTripleRescue
from experiments.distributed_5g.temporal_graph_v6 import PermutationInvariantGroupRescue
from experiments.robustness_5g.design import attack_schedule, s5_members, s5_count
from experiments.distributed_5g.v6_design import v6_attack_schedule, v6_members, v6_count


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


def test_v3_structural_evidence_matches_batch_and_online():
    dev=frame(30,"v3-dev")
    labels=pd.DataFrame({"attack":np.zeros(len(dev),dtype=int),
                         "scenario":["N1"]*len(dev)})
    mask=dev.window_id.between(10,24) & dev.nf_id.isin(["slot-1","slot-2","slot-3"])
    # Give the laboratory S5 examples a coordinated temporal shape.
    for window in range(10,25):
        wave=(window//2)%3
        for slot in (1,2,3):
            row=(dev.window_id.eq(window)&dev.nf_id.eq(f"slot-{slot}"))
            dev.loc[row,"request_count"]=3+wave+(slot==2)
    labels.loc[mask,"attack"]=1
    labels.loc[mask,"scenario"]="S5"

    detector=TemporalGraphV3Detector(seed=321,history=8,cluster_order=2,decay=.75)
    detector.fit_development([(dev,labels)])
    test=frame(20,"v3-heldout")

    batch=detector.clean_copy().decision_function(test)
    online=detector.clean_copy()
    pieces=[]
    for _,part in test.groupby("window_id",sort=False):
        pieces.append(online.decision_function(part))
    assert np.allclose(batch,np.concatenate(pieces),rtol=0,atol=1e-12)


def test_v3_detector_roundtrips_with_clean_temporal_state():
    dev=frame(30,"v3-dev-serialization")
    labels=pd.DataFrame({"attack":np.zeros(len(dev),dtype=int),
                         "scenario":["N1"]*len(dev)})
    mask=dev.window_id.between(8,22)&dev.nf_id.isin(["slot-9","slot-10","slot-11"])
    for window in range(8,23):
        wave=(window//2)%3
        for slot in (9,10,11):
            row=(dev.window_id.eq(window)&dev.nf_id.eq(f"slot-{slot}"))
            dev.loc[row,"request_count"]=3+wave+(slot==10)
    labels.loc[mask,"attack"]=1
    labels.loc[mask,"scenario"]="S5"
    detector=TemporalGraphV3Detector(seed=456,cluster_order=2,decay=.75)
    detector.fit_development([(dev,labels)])
    expected=detector.clean_copy().decision_function(frame(18,"v3-roundtrip"))

    with tempfile.TemporaryDirectory() as directory:
        path=Path(directory)/"v3.joblib"
        joblib.dump(detector.clean_copy(),path)
        loaded=joblib.load(path)
        loaded.reset()
        actual=loaded.decision_function(frame(18,"v3-roundtrip"))
    assert np.allclose(expected,actual,rtol=0,atol=1e-12)


def test_v4_common_mode_rejection_preserves_local_residual_wave():
    global_frame=_graph_pattern(False)
    local_frame=_graph_pattern(True)
    global_features=CommonModeResidualTransformer(history=8).transform(global_frame)
    local_features=CommonModeResidualTransformer(history=8).transform(local_frame)
    names={name:i for i,name in enumerate(FEATURES_V4)}
    idx=[w*32+slot for w in range(10,20) for slot in (1,2,3)]
    global_contrast=float(np.mean(global_features[idx,names["localized_residual_corr_contrast"]]))
    local_contrast=float(np.mean(local_features[idx,names["localized_residual_corr_contrast"]]))
    assert local_contrast > global_contrast + .15


def test_v4_transform_batch_matches_online_windows():
    test=frame(18,"v4-test")
    batch=CommonModeResidualTransformer(history=8).transform(test)
    online=CommonModeResidualTransformer(history=8)
    parts=[]
    for _,part in test.groupby("window_id",sort=False):
        parts.append(online.transform_online(part))
    assert np.allclose(batch,np.vstack(parts),rtol=0,atol=1e-12)


def test_v4_detector_roundtrips_with_clean_state():
    dev=frame(30,"v4-dev")
    labels=pd.DataFrame({
        "attack":np.zeros(len(dev),dtype=int),
        "scenario":["N1"]*len(dev),
    })
    mask=dev.window_id.between(18,27) & dev.nf_id.isin(["slot-1","slot-2","slot-3"])
    labels.loc[mask,"attack"]=1
    labels.loc[mask,"scenario"]="S5"
    detector=TemporalGraphV4Detector(seed=123,history=8,cluster_order=2,decay=.75)
    detector.fit_development([(dev,labels)])
    test=frame(16,"v4-heldout")
    batch=detector.clean_copy().decision_function(test)
    online=detector.clean_copy()
    parts=[]
    for _,part in test.groupby("window_id",sort=False):
        parts.append(online.decision_function(part))
    assert np.allclose(batch,np.concatenate(parts),rtol=0,atol=1e-12)
    with tempfile.TemporaryDirectory() as directory:
        path=Path(directory)/"v4.joblib"
        joblib.dump(detector.clean_copy(),path)
        loaded=joblib.load(path);loaded.reset()
        assert np.allclose(batch,loaded.decision_function(test),rtol=0,atol=1e-12)


def _cyclic_rescue_pattern(attack):
    rows=[]
    for w in range(30):
        base=2+(w//8)
        wave=(0,1,1,2,1,0)[(w//3)%6]
        for slot in range(32):
            count=base
            if attack and slot in (17,18,19):
                count=base+wave+(slot==18)
            rows.append(dict(run_id="cyclic-rescue",window_id=w,nf_id=f"slot-{slot}",
                             timestamp=float(w),request_count=float(count),
                             interarrival_mean=1.0/max(1,count),latency_ms=1.0,error_rate=0.0))
    return pd.DataFrame(rows)


def test_cyclic_rescue_detects_local_triple_not_global_common_mode():
    detector=CyclicTripleRescue(iqr_limit=1.0,decay=.8)
    normal=detector.clean_copy().decision_function(_cyclic_rescue_pattern(False))
    attack=detector.clean_copy().decision_function(_cyclic_rescue_pattern(True))
    assert float(normal.max()) <= 2.0
    attack_frame=_cyclic_rescue_pattern(True)
    target=attack_frame.nf_id.isin(["slot-17","slot-18","slot-19"]).to_numpy()
    assert float(attack[target].max()) > 2.0


def test_cyclic_rescue_batch_matches_online_and_serializes():
    test=_cyclic_rescue_pattern(True)
    detector=CyclicTripleRescue(iqr_limit=1.0,decay=.8)
    batch=detector.clean_copy().decision_function(test)
    online=detector.clean_copy()
    pieces=[]
    for _,part in test.groupby("window_id",sort=False):
        pieces.append(online.decision_function(part))
    assert np.allclose(batch,np.concatenate(pieces),rtol=0,atol=1e-12)
    with tempfile.TemporaryDirectory() as directory:
        path=Path(directory)/"rescue.joblib"
        joblib.dump(detector.clean_copy(),path)
        loaded=joblib.load(path);loaded.reset()
        assert np.allclose(batch,loaded.decision_function(test),rtol=0,atol=1e-12)


def _single_window():
    return pd.DataFrame([
        dict(run_id="v6-window",window_id=0,nf_id=f"slot-{slot}",
             timestamp=0.0,request_count=2.0)
        for slot in range(32)
    ])


def test_v6_group_scores_are_permutation_invariant_within_zone():
    part=_single_window()
    probability=np.full(32,.05,dtype=float)
    probability[[0,3,6,7]]=[.92,.90,.88,.86]
    detector=PermutationInvariantGroupRescue(
        model=object(),history=8,min_group=2,max_group=5,decay=.75)
    original=detector._window_scores(part,probability)

    mapping={0:5,1:1,2:7,3:0,4:4,5:2,6:6,7:3}
    renamed=part.copy()
    renamed["nf_id"]=renamed.nf_id.map(
        lambda x: f"slot-{(int(x.split('-')[1])//8)*8 + mapping[int(x.split('-')[1])%8]}")
    permuted=detector._window_scores(renamed,probability)

    for zone in range(4):
        left=sorted(original[f"slot-{zone*8+i}"] for i in range(8))
        right=sorted(permuted[f"slot-{zone*8+i}"] for i in range(8))
        assert np.allclose(left,right,rtol=0,atol=1e-12)


def test_v6_group_score_supports_k2_and_k5_without_adjacency():
    part=_single_window()
    detector=PermutationInvariantGroupRescue(
        model=object(),history=8,min_group=2,max_group=5,decay=.75)

    p2=np.full(32,.05,dtype=float)
    p2[[0,7]]=[.95,.92]
    score2=detector._window_scores(part,p2)
    assert score2["slot-0"]>0 and score2["slot-7"]>0
    assert all(score2[f"slot-{i}"]==0 for i in range(1,7))

    p5=np.full(32,.04,dtype=float)
    members=(0,2,3,5,7)
    p5[list(members)]=[.95,.94,.93,.92,.91]
    score5=detector._window_scores(part,p5)
    assert all(score5[f"slot-{i}"]>0 for i in members)
    assert all(score5[f"slot-{i}"]==0 for i in set(range(8))-set(members))


def test_v6_schedule_balances_group_sizes_and_random_positions():
    schedule=v6_attack_schedule(600,90501)
    assert len(schedule)==8
    sizes=[len(v6_members(entry)) for entry in schedule]
    assert sorted(sizes)==[2,2,3,3,4,4,5,5]
    for entry in schedule:
        members=v6_members(entry)
        assert len({m//8 for m in members})==1
        for window in range(entry[0],entry[1]):
            counts=[v6_count(entry,window,m) for m in members]
            assert all(2<=value<=5 for value in counts)
    for i,left in enumerate(schedule):
        for right in schedule[i+1:]:
            assert left[1]<=right[0] or right[1]<=left[0]
