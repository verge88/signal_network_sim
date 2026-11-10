import json
from experiments.freshness_5g.audit import read_predictions


def test_epoch_seconds_keep_original_precision_and_numeric_type(tmp_path):
    started=1794293706.1234567
    evaluation=1794293706.1238003
    finished=1794293706.1239004
    p=tmp_path/'predictions.jsonl'
    p.write_text(json.dumps(dict(window_id=0,nf_id='slot-4',evaluation_time=evaluation,decision_time=finished)))
    r=next(read_predictions(p).itertuples())
    assert r.evaluation_time==evaluation and r.decision_time==finished
    assert started<=r.evaluation_time<=r.decision_time
    assert r.decision_time-r.evaluation_time==finished-evaluation
