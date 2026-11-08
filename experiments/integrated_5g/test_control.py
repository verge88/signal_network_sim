import copy
import time
from experiments.integrated_5g.control import combine, Control
from experiments.live_5g.collect import sign_report


def test_three_states_and_forgery():
    key = b'lab-test'
    now = time.time()
    r = dict(config_hash='bad', instance='nf', nonce='challenge', timestamp=now)
    e = dict(report=r, signature=sign_report(r, key))
    assert combine(False, e, 'challenge', 'good', 'nf', key, now) == dict(H='normal', O='alarm', K='alarm')
    forged = copy.deepcopy(e)
    forged['report']['config_hash'] = 'good'
    assert combine(False, forged, 'challenge', 'good', 'nf', key, now) == dict(H='normal', O='normal', K='alarm')
    assert combine(False, None, 'challenge', 'good', 'nf', key, now) == dict(H='normal', O='unknown', K='unknown')
    assert combine(False, e, 'new-challenge', 'good', 'nf', key, now) == dict(H='normal', O='unknown', K='unknown')
    assert combine(False, e, 'challenge', 'good', 'nf', key, now+1) == dict(H='normal', O='unknown', K='unknown')
    assert combine(True, None, 'challenge', 'good', 'nf', key, now) == dict(H='alarm', O='alarm', K='alarm')


def test_actual_process_and_outage(tmp_path):
    c = Control(tmp_path, 'instance')
    try:
        c.prepare(1, 'N1')
        row = c.observe(1, 'N1')
        assert row['original']['report']['pid'] != row['ground_truth']['pid']
        assert row['ground_truth']['answer'] == 'ALLOW'
        c.prepare(2, 'F')
        row = c.observe(2, 'F')
        assert row['ground_truth']['answer'] == 'DENY'
        assert row['original']['report']['config_hash'] != row['expected']
        assert row['delivered']['report']['config_hash'] == row['expected']
        c.prepare(40, 'N1')
        assert c.observe(40, 'N1')['delivered'] is None
        c.prepare(44, 'N1')
        assert c.observe(44, 'N1')['original'] is not None
    finally:
        c.close()
