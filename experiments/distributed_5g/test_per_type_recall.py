import numpy as np
import pytest

from experiments.distributed_5g.cross_scenario_detector import MEASUREMENTS, SLOTS
from experiments.distributed_5g.per_type_detector import (
    CONTROL_NF, OnlineScorer, PerTypeModel, PrefixTracker, ProfileModel, missing_live_receipt, shape_panel,
)
from experiments.distributed_5g.per_type_replay import SCENARIOS, alarm_grid


def test_process_health_and_receipt_are_both_required():
    row = dict(window_id=11, completed=2., observer_alive=True, delivered=None)
    assert missing_live_receipt(row, 11, 2.)
    assert not missing_live_receipt(dict(row, observer_alive=False), 11, 2.)
    assert not missing_live_receipt(dict(row, delivered={"report": {}}), 11, 2.)
    assert not missing_live_receipt(None, 11, 2.)
    with pytest.raises(ValueError):
        missing_live_receipt(row, 11, 1.)
    with pytest.raises(KeyError):
        missing_live_receipt({k:v for k,v in row.items() if k!="delivered"}, 11, 2.)


class ScorerFixture:
    def predict_proba(self, x):
        p = 1 / (1 + np.exp(-x[:, 3]))
        return np.column_stack([1-p, p])


def test_stream_reorder_grants_run_reset_and_batch_parity():
    rng = np.random.default_rng(107)
    data = rng.integers(0, 11, (132, 32, 3)).astype(float)
    data[:,:,2] /= 10
    grant = rng.random((132,32)) < .15
    labels = np.full((132,32), "N1")
    labels[4:10,2] = "G"
    labels[25:32,1:4] = "S5"
    profiles = ProfileModel.fit([(data[:,:,0],grant,labels), (data[:,:,0],grant,labels)])
    assert len(profiles.paths) == 4
    assert len(profiles.cohorts) == 1
    model = PerTypeModel(ScorerFixture(), profiles, .95, .3, (), (), cohort_cut=.25, support_weight=.65)
    loc = np.zeros_like(grant)
    loss = np.zeros_like(grant)
    loss[::9,SLOTS.index(CONTROL_NF)] = True
    graph, profile, cohort = model.score_panel(data, grant)
    expected = model.decide(loc,loss,grant,graph,profile,cohort)
    perm = np.arange(32).reshape(4,8)[:,::-1].ravel()
    np.testing.assert_allclose(shape_panel(data[:,perm]),shape_panel(data)[:,perm],atol=1e-12)
    pg, pp, pc = model.score_panel(data[:,perm], grant[:,perm])
    np.testing.assert_allclose(pg, graph[:,perm], atol=1e-12)
    np.testing.assert_array_equal(pp, profile[:,perm])
    np.testing.assert_array_equal(pc, cohort[:,perm])
    scorer = OnlineScorer(model)
    for w, values in enumerate(data):
        rows = [dict(run_id="one",window_id=w,nf_id=nf,**dict(zip(MEASUREMENTS, val)))
                for nf,val in zip(SLOTS,values)]
        decisions = {nf:dict(local_security=False,allowed=a) for nf,a in zip(SLOTS,grant[w])}
        receipt = dict(window_id=w,completed=float(w),observer_alive=True,delivered=None if w%9==0 else {})
        got = scorer.score(rows[::-1],decisions,receipt,float(w))
        assert list(got.values()) == expected[w].tolist()
    with pytest.raises(ValueError, match="contiguous"):
        scorer.score(rows,decisions,receipt,float(w))
    for row in rows:
        row.update(run_id="next",window_id=0)
    assert scorer.score(rows,decisions,None,0.) == OnlineScorer(model).score(rows,decisions,None,0.)


def test_group_requires_distinct_peers_in_one_zone_at_the_same_stage():
    tracker = PrefixTracker([(1, 2), (1, 3)], [(0, 1, 1)])
    row = np.full(32, 9)
    row[[0, 2, 5, 8, 9, 16]] = 1
    tracker.advance(row, np.zeros(32, bool))
    nf, _, _, inclusion = next(tracker.simultaneous_members())
    assert set(nf) == {0, 2, 5}
    assert np.all(inclusion == 1)
    row[[0, 2, 5, 8, 9, 16]] = [3, 2, 3, 3, 2, 3]
    tracker.advance(row, np.zeros(32, bool))
    nf, stage, _, inclusion = next(tracker.simultaneous_members())
    assert set(nf[stage == 1]) == {0, 2, 5}
    assert np.all(inclusion[stage == 1] == 1)


def test_threshold_grid_equals_direct_strict_alarms_at_ties():
    labels = np.repeat(["N1", *SCENARIOS], 3)
    rng = np.random.default_rng(99)
    scores = [rng.choice([-.1, 0., .4, .7, 1.], len(labels)) for _ in range(3)]
    cuts = [np.array([0., .4, 1.]), np.array([.4, .7, 1.]), np.array([0., 1.])]
    fixed = rng.random(len(labels)) < .2
    for name, support, measured in alarm_grid(cuts, scores, fixed, labels):
        assert support == 3
        for ix in np.ndindex(measured.shape):
            alarm = fixed.copy()
            for d in range(3):
                alarm |= scores[d] > cuts[d][ix[d]]
            assert measured[ix] == np.count_nonzero(alarm & (labels == name))


def test_group_inclusion_quantifies_ambiguous_members():
    tracker = PrefixTracker([(1, 2), (1, 3)], [(0, 1, 1)])
    counts = np.full(32, 7)
    counts[:4] = 1
    tracker.advance(counts, np.zeros(32, bool))
    indices, _, _, inclusion = next(tracker.simultaneous_members())
    assert indices.tolist() == [0, 1, 2, 3]
    np.testing.assert_allclose(inclusion, .75)
    counts[:4] = [2, 3, 3, 3]
    tracker.advance(counts, np.zeros(32, bool))
    indices, _, _, inclusion = next(tracker.simultaneous_members())
    assert indices.tolist() == [0, 1, 2, 3]
    np.testing.assert_allclose(inclusion, [1., 2/3, 2/3, 2/3])
