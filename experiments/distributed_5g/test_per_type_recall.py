import numpy as np
import pytest

from experiments.distributed_5g.cross_scenario_detector import MEASUREMENTS, SLOTS
from experiments.distributed_5g.per_type_detector import (
    CONTROL_NF, OnlineScorer, PerTypeModel, ProfileModel, missing_live_receipt, shape_panel,
)


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
    labels[25:32,3] = "S5"
    profiles = ProfileModel.fit([(data[:,:,0],grant,labels), (data[:,:,0],grant,labels)])
    assert len(profiles.paths) == 2
    model = PerTypeModel(ScorerFixture(), profiles, .95, .3, (), ())
    loc = np.zeros_like(grant)
    loss = np.zeros_like(grant)
    loss[::9,SLOTS.index(CONTROL_NF)] = True
    graph, profile = model.score_panel(data, grant)
    expected = model.decide(loc,loss,grant,graph,profile)
    perm = np.arange(32).reshape(4,8)[:,::-1].ravel()
    np.testing.assert_allclose(shape_panel(data[:,perm]),shape_panel(data)[:,perm],atol=1e-12)
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
