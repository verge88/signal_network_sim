"""Independent arithmetic replay; the measured refinement engine is not imported."""
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from experiments.resilient_5g.audit import run as audit_previous, replay_temporal, lines, read, mac

def temporal(frame,grants,key):
    replay=replay_temporal(frame,grants,key)
    previous={}; result={}
    for r in frame.sort_values(['window_id','nf_id']).itertuples():
        identity=(r.run_id,r.window_id,r.nf_id)
        (plain,reset),allowed=replay[identity]
        result[identity]=(min(previous.get(r.nf_id,0.),reset),0. if allowed else plain,allowed)
        previous[r.nf_id]=reset
    return result

def run(root):
    audit_previous(root)
    oldsummary=read(root/'resilient_summary.json'); checks=dict(oldsummary['checks'])
    model=read(root/'frozen/refinement.json'); manifest=read(root/'test/manifest.json')
    checks['refinement_frozen']=model['created_at']<manifest['started_at'] and model==manifest['refinement']
    checks['refinement_hash']=hashlib.sha256((root/'frozen/refinement.json').read_bytes()).hexdigest()==manifest['refinement_sha256']
    checks['refinement_normal_sources']=all(hashlib.sha256((root/p).read_bytes()).hexdigest()==v for p,v in model['source_hashes'].items())
    cal=pd.read_csv(root/'cal/features.csv')
    def scores(phase,frame):
        grants={r['window_id']:r['envelope'] for r in lines(root/phase/'load_grants.jsonl')}
        key=bytes.fromhex((root/phase/'lab_operator_key.hex').read_text())
        return temporal(frame,grants,key)
    rank=math.ceil((len(cal)+1)*(1-model['alpha_component']))
    assert rank<=len(cal)
    thresholds={f:float(sorted(cal[f])[rank-1]) for f in ['request_count','error_rate','update_count']}
    cal_scores=scores('cal',cal)
    checks['separate_calibration']=thresholds==model['feature_thresholds']
    checks['persistence_calibration']=sorted(x[0] for x in cal_scores.values())[rank-1]==model['persistence_threshold']
    checks['carry_calibration']=sorted(x[1] for x in cal_scores.values())[rank-1]==model['carry_threshold']
    f=pd.read_csv(root/'test/features.csv'); labels=pd.read_csv(root/'test/labels.csv').fillna({'episode_id':''})
    keys=['run_id','window_id','nf_id']; rows={tuple(r[k] for k in keys):r for r in f.to_dict('records')}
    previous={tuple(r[k] for k in keys):r for r in lines(root/'test/resilient_predictions.jsonl')}
    decisions=lines(root/'test/refined_predictions.jsonl'); calculated=scores('test',f)
    control={c['window_id']:c for c in lines(root/'test/control_events.jsonl')}
    key=bytes.fromhex((root/'test/lab_control_key.hex').read_text())
    verified=True
    for d in decisions:
        identity=tuple(d[k] for k in keys); r=rows[identity]; prior=previous[identity]
        persistent,carry,allowed=calculated[identity]
        tail=any(r[k]>v for k,v in thresholds.items())
        semantic=bool(r['profile_policy_violation']>0 or r['heartbeat_age']>30 or r['forbidden_operation']>0 or r['route_mismatch']>0 or r['report_mismatch']>0 or r['auth_failure']>0)
        security=tail or semantic; state='alarm' if security else 'normal'
        if r['nf_id']=='slot-4':
            c=control[r['window_id']]; e=c['delivered']
            if e is None:
                state='alarm' if security else 'unknown'
            else:
                p=e['report']; fresh=(p['nonce']==c['nonce'] and p['instance']==c['instance'] and 0<=d['evaluation_time']-p['timestamp']<=.25)
                security=security or not mac(e,key) or (fresh and p['config_hash']!=c['expected'])
                state='alarm' if security else 'normal' if fresh else 'unknown'
        signals=dict(separate_tail=bool(tail),semantic=semantic,persistence=persistent,carry=carry,permit_valid=bool(allowed))
        verified &= signals==d['signals']
        for method,extra in [('SPLIT',False),('PERSIST',persistent>model['persistence_threshold']),('CARRY',carry>model['carry_threshold'])]:
            alarm=state=='alarm' or prior['signals']['replay'] or extra
            availability=prior['signals']['availability']
            expected=dict(security=bool(alarm),availability=availability,state='security' if alarm else 'availability' if availability else state)
            verified &= d['outputs'][method]==expected
    checks['refinement_online_replayed']=bool(verified)
    frame=pd.DataFrame(decisions).merge(labels,on=keys,validate='one_to_one')
    checks['refinement_complete']=len(frame)==len(f) and len(decisions)==len(f)
    for i,r in frame.iterrows():
        prior=previous[tuple(r[k] for k in keys)]
        r.outputs['PREVIOUS']=prior['outputs']['KALL']
    frame['previous_decision_time']=[previous[tuple(r[k] for k in keys)]['decision_time'] for r in frame.to_dict('records')]
    methods=['PREVIOUS','SPLIT','PERSIST','CARRY']; metrics={}; episodes=[]
    for scope,g in [('all',frame),('slot-3',frame[frame.nf_id=='slot-3']),('slot-4',frame[frame.nf_id=='slot-4'])]:
        metrics[scope]={}
        for method in methods:
            y=g.attack.astype(bool); security=g.outputs.map(lambda x:x[method]['security']); avail=g.outputs.map(lambda x:x[method]['availability'])
            unknown=g.outputs.map(lambda x:x[method]['state']=='unknown')
            metrics[scope][method]=dict(attack_windows=int(y.sum()),normal_windows=int((~y).sum()),
                security_tp=int((security&y).sum()),security_fp=int((security&~y).sum()),security_fn=int((~security&y).sum()),
                availability_attack_windows=int((avail&y).sum()),availability_normal_windows=int((avail&~y).sum()),
                operational_tp=int(((security|avail)&y).sum()),operational_normal_windows=int(((security|avail)&~y).sum()),
                unknown_attack=int((unknown&y).sum()),unknown_normal=int((unknown&~y).sum()))
    for episode,g in frame[frame.attack==1].groupby('episode_id'):
        specification=next(x for x in manifest['schedule'] if x[0]==int(g.window_id.min()))
        for method in methods:
            clock_column='previous_decision_time' if method=='PREVIOUS' else 'decision_time'
            sec=g[g.outputs.map(lambda x:x[method]['security'])]
            avail=g[g.outputs.map(lambda x:x[method]['availability'])]
            notified=g[g.outputs.map(lambda x:x[method]['security'] or x[method]['availability'])]
            episodes.append(dict(episode=episode,scenario=g.scenario.iloc[0],method=method,
                ramp_step=specification[4] if g.scenario.iloc[0]=='G' else None,
                permitted_windows=int(g.signals.map(lambda x:x['permit_valid']).sum()),
                security=not sec.empty,availability=not avail.empty,operational=not notified.empty,
                security_delay=float(sec[clock_column].min()-g.onset.iloc[0]) if not sec.empty else None,
                operational_delay=float(notified[clock_column].min()-g.onset.iloc[0]) if not notified.empty else None))
    result=dict(checks={k:bool(v) for k,v in checks.items()},thresholds=model,metrics=metrics,episodes=episodes,overhead=oldsummary['overhead'])
    (root/'calibrated_summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({'audit_pass':all(checks.values()),'metrics':metrics['all']}))
    assert all(checks.values()),checks

if __name__=='__main__':
    run(Path(sys.argv[1]))
