import numpy as np
import pandas as pd
import pytest

from experiments.distributed_5g.cross_scenario_detector import (
    MEASUREMENTS, SLOTS, CrossScenarioDetector, WindowScorer, feature_block, feature_panel,
    operating_threshold,
)
from experiments.distributed_5g.cross_scenario_replay import KEYS, aligned


class TestProbability:
    __test__ = False

    def predict_proba(self, x):
        score = 1 / (1 + np.exp(-x[:, 3]))
        return np.column_stack((1 - score, score))


def test_chunk_window_parity_causality_and_zone_permutation():
    rng = np.random.default_rng(53)
    values = rng.integers(0, 15, size=(143, 32, 3)).astype(float)
    values[:, :, 2] /= 14
    expected = feature_panel(values)
    np.testing.assert_allclose(feature_panel(values[:19]), expected[:19], atol=1e-12)
    perm = np.arange(32).reshape(4, 8)[:, ::-1].ravel()
    np.testing.assert_allclose(feature_panel(values[:, perm]), expected[:, perm], atol=1e-12)
    previous = None
    model = CrossScenarioDetector(TestProbability(), .8, (), ())
    scorer = WindowScorer(model)
    local = rng.random((143, 32)) < .1
    allowed = rng.random((143, 32)) < .2
    batch = model.decide(model.score_features(expected), local, allowed)
    for w, observation in enumerate(values):
        block, previous = feature_block(values[w:w+1], previous)
        np.testing.assert_allclose(block[0], expected[w], atol=1e-12)
        rows = [dict(run_id="first", window_id=w, nf_id=nf, **dict(zip(MEASUREMENTS, row)))
                for nf, row in zip(SLOTS, observation)]
        decisions = {nf: dict(local_security=l, allowed=a) for nf, l, a in zip(SLOTS, local[w], allowed[w])}
        result = scorer.score(rows[::-1], decisions)
        assert [result[nf] for nf in SLOTS] == batch[w].tolist()
    with pytest.raises(ValueError, match="contiguous"):
        scorer.score(rows, decisions)
    for row in rows:
        row.update(run_id="second", window_id=0)
    assert scorer.score(rows, decisions) == WindowScorer(model).score(rows, decisions)


def test_strict_threshold_ties_and_local_alarm_budget():
    truth = np.array([False] * 10 + [True])
    local = np.zeros(11, bool)
    local[0] = True
    allowed = np.zeros(11, bool)
    allowed[1] = True
    scores = np.array([0., 1., .9, .8, .8, .5, .4, .3, .2, .1, 1.])
    cut = operating_threshold(scores, truth, local, allowed, budget=.2)
    assert cut == .8
    decision = CrossScenarioDetector(None, cut, (), ()).decide(scores, local, allowed)
    assert np.flatnonzero(decision).tolist() == [0, 2, 10]
    local[1:3] = True
    with pytest.raises(ValueError, match="alone"):
        operating_threshold(scores, truth, local, allowed, budget=.2)


def test_join_reorders_and_rejects_mismatched_or_duplicate_keys():
    keys = pd.MultiIndex.from_product([["one"], [0], SLOTS], names=KEYS)
    frame = keys.to_frame(index=False).assign(attack=np.arange(32))
    assert aligned(frame.iloc[::-1], keys).attack.tolist() == list(range(32))
    frame.loc[0, "nf_id"] = "other"
    with pytest.raises(ValueError, match="identities"):
        aligned(frame, keys)
    frame.loc[0, "nf_id"] = SLOTS[1]
    with pytest.raises(ValueError):
        aligned(frame, keys)
