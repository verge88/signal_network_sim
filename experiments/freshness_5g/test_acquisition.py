import copy
import io
import json
import pytest
from experiments.freshness_5g.acquisition import Acquisition
from experiments.integrated_5g.control import combine, MAX_AGE
from experiments.resilient_5g.test_detector import control, engine, row
from experiments.live_5g.collect import sign_report


class Source:
    def __init__(self, delay=0.):
        self.now = 100.
        self.delay = delay
        self.out = io.StringIO()
        self.calls = 0
        self.previous = None

    def observe(self, window, scenario):
        self.calls += 1
        c = control(str(window), self.calls, self.now)
        c.update(window_id=window, expected='expected', started=self.now)
        if scenario == 'C':
            c['delivered']['report']['config_hash'] = 'changed'
            c['delivered']['signature'] = sign_report(c['delivered']['report'], b'lab')
        c['original'] = copy.deepcopy(c['delivered'])
        if scenario == 'R':
            c['delivered'] = copy.deepcopy(self.previous)
        if scenario == 'D':
            c['delivered'] = None
        self.previous = c['original']
        self.now += self.delay
        c['completed'] = self.now
        return c


def state(source, c):
    return combine(False,c['delivered'],c['nonce'],c['expected'],c['instance'],b'lab',source.now)['K']


@pytest.mark.parametrize('mode,expected', [('early','unknown'),('late','alarm')])
def test_long_other_nf_work_does_not_age_just_in_time_report(mode, expected):
    source = Source(); acquisition = Acquisition(source,mode)
    acquisition.at_slot(1,'C')
    source.now += .4  # Actual ordering represented by a controlled test clock.
    c = acquisition.at_decision(1)
    assert state(source,c) == expected
    assert source.calls == 1
    assert len(source.out.getvalue().splitlines()) == 1
    assert MAX_AGE == .25
    assert json.loads(source.out.getvalue()) == c


def test_late_delivery_is_still_unknown_and_timestamp_is_not_rewritten():
    source=Source(delay=.35); acquisition=Acquisition(source,'late')
    acquisition.at_slot(1,'C'); source.now+=.4
    c=acquisition.at_decision(1)
    assert c['original']['report']['timestamp'] == 100.4
    assert state(source,c) == 'unknown'


def test_late_acquisition_preserves_replay_and_suppression_distinction():
    source=Source(); acquisition=Acquisition(source,'late'); detector=engine()
    for window,scenario,advance in [(1,'N1',1),(2,'R',1),(3,'D',7)]:
        acquisition.at_slot(window,scenario); source.now+=advance
        c=acquisition.at_decision(window)
        result=detector.step(row(timestamp=source.now),state(source,c),c,None,source.now)
        if scenario=='N1': assert not result['outputs']['KALL']['security']
        if scenario=='R': assert result['signals']['replay'] and result['outputs']['KALL']['security']
        if scenario=='D': assert result['outputs']['KALL']['availability'] and not result['outputs']['KALL']['security']
    assert source.calls == 3


def test_duplicate_and_mismatched_window_cannot_reuse_report():
    source=Source(); acquisition=Acquisition(source,'late')
    acquisition.at_slot(1,'N1')
    with pytest.raises(RuntimeError): acquisition.at_slot(2,'N1')
    with pytest.raises(RuntimeError): acquisition.at_decision(2)
    acquisition.at_decision(1)
    with pytest.raises(RuntimeError): acquisition.at_decision(1)
    assert source.calls==1
