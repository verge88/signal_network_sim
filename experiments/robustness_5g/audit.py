"""Replay unchanged detectors, then audit the new test design and strata."""
import hashlib,json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from experiments.calibrated_5g_v2.audit import run as replay

def run(root):
    replay(root)
    result=json.loads((root/'calibrated_summary.json').read_text())
    origin=json.loads((root/'frozen_origin.json').read_text())
    pins=json.loads(Path(__file__).with_name('source_pins.json').read_text())[str(origin['source_seed'])]
    checks=result['checks']
    checks['previous_confirmed_bundle']=origin==pins and all(hashlib.sha256((root/'frozen'/n).read_bytes()).hexdigest()==v for n,v in pins['frozen_files'].items())
    manifest=json.loads((root/'test/manifest.json').read_text())
    features=pd.read_csv(root/'test/features.csv',float_precision='round_trip')
    labels=pd.read_csv(root/'test/labels.csv')
    keys=['run_id','window_id','nf_id']
    frame=features.merge(labels,on=keys,validate='one_to_one').merge(pd.read_json(root/'test/refined_predictions.jsonl',lines=True),on=keys,validate='one_to_one')
    old=pd.read_json(root/'test/resilient_predictions.jsonl',lines=True)
    lookup={tuple(r[k] for k in keys):r for r in old.to_dict('records')}
    for _,r in frame.iterrows(): r.outputs['PREVIOUS']=lookup[tuple(r[k] for k in keys)]['outputs']['KALL']
    checks['test_design_size']=manifest['windows']==3600 and manifest['actors'] in [8,16,32]
    checks['regime_labels']=all(r.load_regime==['reference','higher_load','cyclic','bursty_recovery'][r.window_id//900] for r in frame.itertuples())
    growth_ok=True
    for entry in manifest['schedule']:
        start,end,name,slot,step,variant=entry
        g=frame[(frame.window_id>=start)&(frame.window_id<end)&(frame.nf_id==f'slot-{slot}')]
        growth_ok &= len(g)==end-start and g.attack.eq(1).all() and g.variant.eq(variant).all()
        if name=='G':
            for r in g.itertuples():
                t=r.window_id-start
                expected=4+step*t if variant=='fast' else 4+t//6 if variant in ['slow','inside_permit'] else (4+t//6 if t%6<3 else 2)
                growth_ok &= r.request_count==expected+1
    checks['actual_growth_and_schedule']=bool(growth_ok)
    metrics={}
    for regime,g in frame.groupby('load_regime'):
        metrics[regime]={}
        for method in ['PREVIOUS','SPLIT','PERSIST','CARRY']:
            y=g.attack.astype(bool); alarm=g.outputs.map(lambda x:x[method]['security']); avail=g.outputs.map(lambda x:x[method]['availability'])
            metrics[regime][method]=dict(attack_windows=int(y.sum()),normal_windows=int((~y).sum()),
                security_tp=int((alarm&y).sum()),security_fp=int((alarm&~y).sum()),security_fn=int((~alarm&y).sum()),
                availability_normal=int((avail&~y).sum()),unknown_normal=int((g.outputs.map(lambda x:x[method]['state']=='unknown')&~y).sum()))
    for e in result['episodes']:
        g=frame[frame.episode_id==e['episode']]
        e['variant']=g.variant.iloc[0]; e['load_regime']=g.load_regime.iloc[0]
        e['windows']=len(g)
    intervals=[]
    for _,g in features.groupby('nf_id'):
        intervals.extend(g.sort_values('window_id').timestamp.diff().dropna().tolist())
    result.update(source=origin,actors=manifest['actors'],test_seed=manifest['seed'],regime_metrics=metrics,
        observation_interval_p50_s=float(np.median(intervals)),observation_interval_p95_s=float(np.percentile(intervals,95)))
    result['normal_security_segments']={}
    for method in ['PREVIOUS','SPLIT','PERSIST','CARRY']:
        segments=0
        for _,g in frame.groupby('nf_id'):
            g=g.sort_values('window_id'); active=g.outputs.map(lambda x:x[method]['security'])&g.attack.eq(0)
            segments+=int((active&~active.shift(fill_value=False)).sum())
        result['normal_security_segments'][method]=segments
    (root/'robustness_summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({'robustness_audit_pass':all(checks.values()),'actors':manifest['actors'],'metrics':result['metrics']['all']}))
    assert all(checks.values()),checks

if __name__=='__main__': run(Path(sys.argv[1]))
