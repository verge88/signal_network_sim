"""Independent replay of extension mathematics and incident categories."""
import collections
import hashlib
import hmac
import json
import math
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from experiments.integrated_5g.audit import audit as audit_baseline


def read(p):
    return json.loads(p.read_text())


def lines(p):
    return list(map(json.loads,p.read_text().splitlines()))


def mac(e,key):
    return bool(e and hmac.compare_digest(e['signature'],hmac.new(key,
       json.dumps(e['report'],sort_keys=True,separators=(',',':')).encode(),hashlib.sha256).hexdigest()))


def replay_temporal(frame,grants,key):
    history=collections.defaultdict(lambda:[None,[],None,[]])
    results={}
    for r in frame.sort_values(['window_id','nf_id']).itertuples():
        e=grants[r.window_id]
        allowed=False
        if mac(e,key):
            g=e['report']
            allowed=(g['run_id']==r.run_id and g['nf_id']==r.nf_id and g['from']<=r.timestamp<=g['until'] and r.request_count<=g['max_requests'])
        state=history[r.nf_id]
        values=[]
        for offset,reset in [(0,False),(2,allowed)]:
            previous,differences=state[offset:offset+2]
            if reset:
                state[offset:offset+2]=[None,[]]
                values.append(0.)
                continue
            delta=0 if previous is None else r.request_count-previous
            differences=(differences+[delta]) if delta>0 else []
            value=min(differences)*max(0,len(differences)-3) if differences else 0.
            state[offset:offset+2]=[r.request_count,differences]
            values.append(float(value))
        results[(r.run_id,r.window_id,r.nf_id)]=(values,allowed)
    return results


def run(root):
    baseaudit=audit_baseline(root)
    checks=dict(baseaudit['checks'])
    model=read(root/'frozen/extensions.json')
    testmanifest=read(root/'test/manifest.json')
    checks['extensions_frozen']=model['created_at']<testmanifest['started_at'] and model==testmanifest['extensions']
    checks['extension_hash']=hashlib.sha256((root/'frozen/extensions.json').read_bytes()).hexdigest()==testmanifest['extensions_sha256']
    checks['normal_sources_hash']=all(hashlib.sha256((root/p).read_bytes()).hexdigest()==v for p,v in model['source_hashes'].items())
    gaps=[]
    for phase in ['train','cal']:
        key=bytes.fromhex((root/phase/'lab_control_key.hex').read_text())
        last=None
        for c in lines(root/phase/'control_events.jsonl'):
            e=c['delivered']
            good=bool(mac(e,key) and e['report']['nonce']==c['nonce'] and e['report']['instance']==c['instance'] and 0<=c['completed']-e['report']['timestamp']<=.25)
            if last is None or good:
                last=c['completed']
            else:
                gaps.append(c['completed']-last)
    checks['availability_fit']=model['availability_seconds']==math.ceil(max(gaps,default=0))+1
    cal=pd.read_csv(root/'cal/features.csv')
    grants={r['window_id']:r['envelope'] for r in lines(root/'cal/load_grants.jsonl')}
    operator=bytes.fromhex((root/'cal/lab_operator_key.hex').read_text())
    temp=replay_temporal(cal,grants,operator)
    rank=math.ceil((len(cal)+1)*(1-model['alpha']))
    thresholds={m:sorted(v[0][i] for v in temp.values())[rank-1] for i,m in enumerate(['T','P'])}
    checks['temporal_calibration']=thresholds==model['temporal_thresholds']
    f=pd.read_csv(root/'test/features.csv')
    labels=pd.read_csv(root/'test/labels.csv').fillna({'episode_id':''})
    keys=['run_id','window_id','nf_id']
    grantmap={r['window_id']:r['envelope'] for r in lines(root/'test/load_grants.jsonl')}
    operator=bytes.fromhex((root/'test/lab_operator_key.hex').read_text())
    temp=replay_temporal(f,grantmap,operator)
    control={r['window_id']:r for r in lines(root/'test/control_events.jsonl')}
    key=bytes.fromhex((root/'test/lab_control_key.hex').read_text())
    old={tuple(r[k] for k in keys):r for r in lines(root/'test/integrated_predictions.jsonl')}
    decisions=lines(root/'test/resilient_predictions.jsonl')
    used=set(); requested=set(); high={}; last=None; verified=True
    for r in decisions:
        identity=tuple(r[k] for k in keys)
        scores,allowed=temp[identity]
        kstate=old[identity]['states']['K']
        replay=False; gap=0.; outage=False
        if r['nf_id']=='slot-4':
            c=control[r['window_id']]
            requested.add(c['nonce'])
            e=c['delivered']; hello=c['session_hello']; now=r['evaluation_time']
            assert mac(hello,key)
            available=bool(mac(e,key) and e['report']['nonce']==c['nonce'] and e['report']['instance']==c['instance']
                  and 0<=now-e['report']['timestamp']<=.25 and e['report']['session']==hello['report']['session'])
            if mac(e,key) and e['report']['instance']==c['instance']:
                p=e['report']; token=(p['session'],p['sequence'])
                if token in used:
                    replay=True; available=False
                elif p['nonce']==c['nonce'] and p['session']==hello['report']['session']:
                    replay=p['sequence']<=high.get(p['session'],0)
                    high[p['session']]=max(p['sequence'],high.get(p['session'],0))
                    used.add(token)
                elif p['nonce'] in requested:
                    used.add(token)
                else:
                    replay=True; available=False
            if last is None or (available and not replay):
                last=now
            gap=now-last
            outage=gap>model['availability_seconds']
        signals=dict(replay=bool(replay),availability=bool(outage),gap_seconds=gap,
          temporal_score=scores[0],policy_temporal_score=scores[1],permit_valid=bool(allowed))
        verified &= signals==r['signals']
        for method,value in r['outputs'].items():
            security=(kstate=='alarm' or (method in ['KR','KALL'] and replay)
                or (method=='KT' and scores[0]>thresholds['T']) or (method in ['KP','KALL'] and scores[1]>thresholds['P']))
            incident=method in ['KA','KALL'] and outage
            expected=dict(security=bool(security),availability=bool(incident),state='security' if security else 'availability' if incident else kstate)
            verified &= value==expected
    checks['extensions_online_replayed']=bool(verified)
    frame=pd.DataFrame(decisions).merge(labels,on=keys,validate='one_to_one')
    checks['complete_new_predictions']=len(frame)==len(f)
    methods=list(decisions[0]['outputs'])
    metrics={}; episodes=[]
    for scope,group in [('all',frame),('slot-3',frame[frame.nf_id=='slot-3']),('slot-4',frame[frame.nf_id=='slot-4'])]:
        metrics[scope]={}
        y=group.attack.astype(bool)
        for m in methods:
            security=group.outputs.map(lambda x:x[m]['security'])
            availability=group.outputs.map(lambda x:x[m]['availability'])
            notification=security|availability
            metrics[scope][m]=dict(security_tp=int((security&y).sum()),security_fp=int((security&~y).sum()),security_fn=int((~security&y).sum()),
                attack_windows=int(y.sum()),normal_windows=int((~y).sum()),
                availability_attack_windows=int((availability&y).sum()),availability_normal_windows=int((availability&~y).sum()),
                operational_tp=int((notification&y).sum()),operational_normal_windows=int((notification&~y).sum()))
    for episode,g in frame[frame.attack==1].groupby('episode_id'):
        for m in methods:
            security=g[g.outputs.map(lambda x:x[m]['security'])]
            incident=g[g.outputs.map(lambda x:x[m]['availability'])]
            notified=g[g.outputs.map(lambda x:x[m]['security'] or x[m]['availability'])]
            episodes.append(dict(episode=episode,scenario=g.scenario.iloc[0],method=m,
              security=not security.empty,availability=not incident.empty,operational=not notified.empty,
              security_delay=float(security.decision_time.min()-g.onset.iloc[0]) if not security.empty else None,
              operational_delay=float(notified.decision_time.min()-g.onset.iloc[0]) if not notified.empty else None))
    result=dict(checks={k:bool(v) for k,v in checks.items()},thresholds=model,metrics=metrics,episodes=episodes,
                overhead=baseaudit['overhead'])
    (root/'resilient_summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({'extension_audit_pass':all(checks.values()),'metrics':metrics['all'],
          'thresholds':thresholds,'availability_seconds':model['availability_seconds']}))
    assert all(checks.values()), checks


if __name__=='__main__':
    run(Path(sys.argv[1]))
