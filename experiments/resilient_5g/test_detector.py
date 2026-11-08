import copy
import time
from experiments.resilient_5g.detector import Engine, Trend, permit_valid
from experiments.resilient_5g.control import Control
from experiments.live_5g.collect import sign_report


def signed(r,key=b'lab'):
    return dict(report=r,signature=sign_report(r,key))


def row(nf='slot-4',timestamp=1,count=4):
    return dict(run_id='test',nf_id=nf,timestamp=timestamp,request_count=count)


def control(nonce,sequence,now,session='session'):
    r=dict(instance='nf',nonce=nonce,sequence=sequence,session=session,
           timestamp=now,config_hash='expected')
    return dict(nonce=nonce,instance='nf',delivered=signed(r),
                session_hello=signed(dict(session=session,instance='nf',challenge='hello')))


def engine():
    return Engine(dict(temporal_thresholds={'T':12,'P':1},availability_seconds=6),b'lab',b'lab')


def test_replay_late_and_restart_are_distinct():
    e=engine()
    a=control('a',1,1)
    assert not e.step(row(), 'normal',a,None,1)['signals']['replay']
    repeat=control('b',2,2)
    repeat['delivered']=copy.deepcopy(a['delivered'])
    assert e.step(row(), 'unknown',repeat,None,2)['outputs']['KR']['security']
    late=control('c',3,3)
    # b was requested but its original report was not delivered previously.
    late['delivered']=control('b',2,2)['delivered']
    assert not e.step(row(), 'unknown',late,None,3)['signals']['replay']
    assert not e.step(row(), 'normal',control('d',1,4,'new-session'),None,4)['signals']['replay']


def test_unavailability_is_not_security():
    e=engine()
    e.step(row(), 'normal',control('a',1,1),None,1)
    c=control('b',2,8)
    c['delivered']=None
    r=e.step(row(), 'unknown',c,None,8)
    assert r['outputs']['KA']['availability']
    assert not r['outputs']['KA']['security']
    assert r['outputs']['KALL']['state']=='availability'


def test_authorized_ramp_and_forged_grant():
    grant=signed({'run_id':'test','nf_id':'slot-3','from':0,'until':20,'max_requests':24})
    e=engine()
    for i,count in enumerate(range(4,24,2)):
        r=e.step(row('slot-3',i,count),'normal',None,grant,i)
        assert not r['outputs']['KP']['security']
    forged=copy.deepcopy(grant)
    forged['report']['max_requests']=100
    assert not permit_valid(forged,b'lab','test','slot-3',1,4)
    e=engine()
    alerts=[e.step(row('slot-3',i,count),'normal',None,None,i)['outputs']['KP']['security']
            for i,count in enumerate(range(4,24,2))]
    assert any(alerts)


def test_actual_session_and_fixed_replay(tmp_path):
    c=Control(tmp_path,'instance')
    try:
        c.prepare(1,'N1')
        a=c.observe(1,'N1')
        c.prepare(2,'R')
        b=c.observe(2,'R')
        c.prepare(3,'R')
        d=c.observe(3,'R')
        assert b['delivered']==d['delivered']==a['original']
        assert b['original']['report']['sequence']>a['original']['report']['sequence']
        assert b['ground_truth']['answer']=='DENY'
    finally:
        c.close()
