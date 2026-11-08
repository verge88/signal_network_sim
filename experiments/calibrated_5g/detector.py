"""Prospective refinement: separate tails and persistent policy trend."""
import collections
import hashlib
import json
import time
from pathlib import Path
import pandas as pd
from experiments.live_5g.evaluate import conformal_threshold, TAIL_FEATURES
from experiments.resilient_5g.detector import Trend, permit_valid
from experiments.integrated_5g.control import combine

class Temporal:
    def __init__(self):
        self.reset=Trend(); self.carry=Trend(); self.previous=0.
    def step(self,count,allowed):
        score=self.reset.step(count,allowed)
        persistent=min(self.previous,score)
        self.previous=score
        carried=self.carry.step(count)
        return persistent,0. if allowed else carried

def fit(root):
    target=root/'frozen/refinement.json'
    if target.exists():
        raise FileExistsError(target)
    train=pd.read_csv(root/'train/features.csv'); cal=pd.read_csv(root/'cal/features.csv')
    for phase in ['train','cal']:
        assert not pd.read_csv(root/phase/'labels.csv').attack.any()
    alpha=.001/4
    grants={r['window_id']:r['envelope'] for r in map(json.loads,(root/'cal/load_grants.jsonl').read_text().splitlines())}
    key=bytes.fromhex((root/'cal/lab_operator_key.hex').read_text())
    trackers=collections.defaultdict(Temporal); scores=[[],[]]
    for r in cal.sort_values(['window_id','nf_id']).itertuples():
        allowed=permit_valid(grants[r.window_id],key,r.run_id,r.nf_id,r.timestamp,r.request_count)
        for output,value in zip(scores,trackers[r.nf_id].step(r.request_count,allowed)):
            output.append(value)
    model=dict(created_at=time.time(),alpha_total=.001,alpha_component=alpha,
        feature_thresholds={f:conformal_threshold(cal[f],alpha) for f in TAIL_FEATURES},
        persistence_threshold=conformal_threshold(scores[0],alpha),
        carry_threshold=conformal_threshold(scores[1],alpha),
        source_hashes={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
            for phase in ['train','cal'] for p in (root/phase).iterdir() if p.is_file()},
        note='Four statistical tests per variant; dependent windows do not establish an operational FPR guarantee. Carry is an ablation, not combined with persistence.')
    target.write_text(json.dumps(model,indent=2,allow_nan=False))

class Engine:
    def __init__(self,model,control_key,operator_key):
        self.model=model; self.control_key=control_key; self.operator_key=operator_key
        self.trackers=collections.defaultdict(Temporal)
    def step(self,row,old,c,grant,now):
        allowed=permit_valid(grant,self.operator_key,row['run_id'],row['nf_id'],row['timestamp'],row['request_count'])
        persistent,carry=self.trackers[row['nf_id']].step(row['request_count'],allowed)
        tail=any(row[f]>v for f,v in self.model['feature_thresholds'].items())
        semantic=(row['profile_policy_violation']>0 or row['heartbeat_age']>30 or
            row['forbidden_operation']>0 or row['route_mismatch']>0 or
            row['report_mismatch']>0 or row['auth_failure']>0)
        state='alarm' if tail or semantic else 'normal'
        if c is not None:
            state=combine(tail or semantic,c['delivered'],c['nonce'],c['expected'],c['instance'],self.control_key,now)['K']
        replay=old['signals']['replay']; availability=old['signals']['availability']
        outputs={}
        for method,extra in [('SPLIT',False),('PERSIST',persistent>self.model['persistence_threshold']),('CARRY',carry>self.model['carry_threshold'])]:
            security=state=='alarm' or replay or extra
            outputs[method]=dict(security=bool(security),availability=availability,
                state='security' if security else 'availability' if availability else state)
        return dict(outputs=outputs,signals=dict(separate_tail=bool(tail),semantic=bool(semantic),
            persistence=persistent,carry=carry,permit_valid=bool(allowed)))

if __name__=='__main__':
    import sys
    fit(Path(sys.argv[1]))
