"""Causal protocol, availability and temporal extensions. No labels as inputs."""
import collections
import hashlib
import json
import math
from pathlib import Path
import secrets
import time
import pandas as pd
from experiments.live_5g.collect import sign_report
from experiments.live_5g.evaluate import conformal_threshold

METHODS = ['K','KR','KA','KT','KP','KALL']


def authenticated(envelope,key):
    return bool(envelope and secrets.compare_digest(envelope['signature'],sign_report(envelope['report'],key)))


def fresh(c,now,key):
    e=c['delivered']
    return bool(authenticated(e,key) and e['report']['nonce']==c['nonce']
        and e['report']['instance']==c['instance'] and 0<=now-e['report']['timestamp']<=.25)


def permit_valid(envelope,key,run_id,nf_id,now,count):
    if not authenticated(envelope,key):
        return False
    r=envelope['report']
    return (r['run_id']==run_id and r['nf_id']==nf_id and r['from']<=now<=r['until']
            and count<=r['max_requests'])


class Trend:
    def __init__(self):
        self.previous=None
        self.run=0
        self.minimum=math.inf

    def step(self,value,authorized=False):
        if authorized:
            self.previous=None
            self.run=0
            self.minimum=math.inf
            return 0.
        difference=0 if self.previous is None else value-self.previous
        if difference>0:
            self.run+=1
            self.minimum=min(self.minimum,difference)
        else:
            self.run=0
            self.minimum=math.inf
        self.previous=value
        return float(self.minimum*max(0,self.run-3)) if self.run else 0.


def temporal_scores(directory):
    f=pd.read_csv(directory/'features.csv').sort_values(['window_id','nf_id'])
    grants={r['window_id']:r['envelope'] for r in map(json.loads,(directory/'load_grants.jsonl').read_text().splitlines())}
    key=bytes.fromhex((directory/'lab_operator_key.hex').read_text())
    trackers=collections.defaultdict(lambda:(Trend(),Trend()))
    scores={'T':[],'P':[]}
    for r in f.itertuples():
        plain,policy=trackers[r.nf_id]
        allowed=permit_valid(grants[r.window_id],key,r.run_id,r.nf_id,r.timestamp,r.request_count)
        scores['T'].append(plain.step(r.request_count))
        scores['P'].append(policy.step(r.request_count,allowed))
    return scores


def fit(root):
    frozen=root/'frozen'
    target=frozen/'extensions.json'
    if target.exists():
        raise FileExistsError(target)
    gaps=[]
    for phase in ['train','cal']:
        assert not pd.read_csv(root/phase/'labels.csv').attack.any()
        key=bytes.fromhex((root/phase/'lab_control_key.hex').read_text())
        last=None
        for c in map(json.loads,(root/phase/'control_events.jsonl').read_text().splitlines()):
            if last is None or fresh(c,c['completed'],key):
                last=c['completed']
            else:
                gaps.append(c['completed']-last)
    scores=temporal_scores(root/'cal')
    model=dict(created_at=time.time(),alpha=.001,
        temporal_thresholds={m:conformal_threshold(v,.001) for m,v in scores.items()},
        availability_seconds=math.ceil(max(gaps,default=0))+1,
        maximum_normal_gap_seconds=max(gaps,default=0),
        source_hashes={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
          for phase in ['train','cal'] for p in (root/phase).iterdir() if p.is_file()},
        note='Temporal feature selected after prior failures; thresholds use only new normal calibration. P uses trusted load authorization.')
    target.write_text(json.dumps(model,indent=2,allow_nan=False))
    print(json.dumps({k:v for k,v in model.items() if k!='source_hashes'}))


class Engine:
    def __init__(self,model,control_key,operator_key):
        self.model=model
        self.control_key=control_key
        self.operator_key=operator_key
        self.trends=collections.defaultdict(lambda:(Trend(),Trend()))
        self.issued=set()
        self.seen=set()
        self.high={}
        self.last_fresh=None

    def step(self,row,k_state,c,grant,now):
        allowed=permit_valid(grant,self.operator_key,row['run_id'],row['nf_id'],row['timestamp'],row['request_count'])
        plain,policy=self.trends[row['nf_id']]
        t=plain.step(row['request_count'])
        p=policy.step(row['request_count'],allowed)
        replay=False
        available=True
        gap=0.
        if c is not None:
            self.issued.add(c['nonce'])
            e=c['delivered']
            available=fresh(c,now,self.control_key)
            hello=c['session_hello']
            if not authenticated(hello,self.control_key):
                raise RuntimeError('Untrusted restart identity')
            available=available and e['report']['session']==hello['report']['session']
            if authenticated(e,self.control_key) and e['report']['instance']==c['instance']:
                r=e['report']
                token=(r['session'],r['sequence'])
                if token in self.seen:
                    replay=True
                    available=False
                elif r['nonce']==c['nonce'] and r['session']==hello['report']['session']:
                    replay=r['sequence']<=self.high.get(r['session'],0)
                    self.high[r['session']]=max(r['sequence'],self.high.get(r['session'],0))
                    self.seen.add(token)
                elif r['nonce'] in self.issued:
                    self.seen.add(token)
                else:
                    replay=True  # Authenticated unsolicited response, protocol incident.
                    available=False
                # A late, previously unanswered challenge remains unknown, not replay.
            if self.last_fresh is None:
                self.last_fresh=now
            if available and not replay:
                self.last_fresh=now
            gap=now-self.last_fresh
        availability=c is not None and gap>self.model['availability_seconds']
        trend=t>self.model['temporal_thresholds']['T']
        policy_trend=p>self.model['temporal_thresholds']['P']
        outputs={}
        for method in METHODS:
            security=(k_state=='alarm' or (method in ['KR','KALL'] and replay)
                or (method=='KT' and trend) or (method in ['KP','KALL'] and policy_trend))
            incident=bool(method in ['KA','KALL'] and availability)
            outputs[method]=dict(security=bool(security),availability=incident,
                state='security' if security else 'availability' if incident else k_state)
        return dict(outputs=outputs,signals=dict(replay=bool(replay),availability=bool(availability),
            gap_seconds=gap,temporal_score=t,policy_temporal_score=p,permit_valid=bool(allowed)))


if __name__=='__main__':
    import sys
    fit(Path(sys.argv[1]))
