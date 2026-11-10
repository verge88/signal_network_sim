"""Prospective paired comparison; no fitting or threshold selection on test data."""
import json
import hashlib
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from experiments.live_5g.evaluate import load_frozen, score_frame, metrics, KEYS
from experiments.freshness_5g.audit import read_predictions
from experiments.robustness_5g.audit import run as audit

PRIMARY = ['rules', 'IF_SBI', 'IF_CONTEXT', 'CARRY']

def run(root):
    audit(root)
    manifest=json.loads((root/'test/manifest.json').read_text())
    origin=json.loads((root/'frozen_origin.json').read_text())
    assert manifest['seed']==origin['source_seed']+31032
    assert manifest['control_acquisition']=='late'
    f=pd.read_csv(root/'test/features.csv',float_precision='round_trip')
    labels=pd.read_csv(root/'test/labels.csv').fillna({'episode_id':''})
    bundle=load_frozen(root/'frozen')
    assert not set(f.run_id)&set(bundle['manifest']['train_runs']+bundle['manifest']['calibration_runs'])
    assert f.timestamp.min()>bundle['manifest']['created_at']
    predictions=score_frame(bundle,f)
    online=pd.read_csv(root/'test/online_predictions.csv',float_precision='round_trip')
    joined=predictions.merge(online,on=KEYS+['method'],suffixes=('', '_online'),validate='one_to_one')
    assert len(joined)==len(predictions)==len(online)
    assert np.array_equal(joined.alarm,joined.alarm_online)
    assert np.allclose(joined.score,joined.score_online,rtol=0,atol=1e-12)
    predictions=joined[KEYS+['method','alarm','score','decision_timestamp']].copy()
    predictions['availability']=False
    refined=read_predictions(root/'test/refined_predictions.jsonl')
    model=json.loads((root/'frozen/refinement.json').read_text())
    for method in ['CARRY','CARRY_without_witness']:
        part=refined[KEYS].copy()
        part['method']=method
        if method=='CARRY':
            part['alarm']=refined.outputs.map(lambda x:x['CARRY']['security']).astype(int)
            part['availability']=refined.outputs.map(lambda x:x['CARRY']['availability'])
        else:
            part['alarm']=refined.signals.map(lambda x:x['separate_tail'] or x['semantic'] or x['carry']>model['carry_threshold']).astype(int)
            part['availability']=False
        # Binary output has no continuous anomaly ranking: AP is binary only.
        part['score']=part.alarm.astype(float)
        part['decision_timestamp']=refined.decision_time
        predictions=pd.concat([predictions,part],ignore_index=True)
    frame=predictions.merge(labels,on=KEYS,validate='many_to_one')
    assert len(f)==3600*32 and len(refined)==len(f)
    totals=[]; strata=[]; episodes=[]
    for method,g in frame.groupby('method'):
        assert len(g)==len(f) and not g.duplicated(KEYS).any()
        totals.append(dict(method=method,primary=method in PRIMARY,**metrics(g)))
        for (regime,scenario),s in g.groupby(['load_regime','scenario']):
            strata.append(dict(method=method,load_regime=regime,scenario=scenario,**metrics(s)))
        for ep,s in g[g.attack==1].groupby('episode_id'):
            hits=s[s.alarm==1]; av=s[s.availability]
            start=s.window_id.min(); nf=s.nf_id.iloc[0]
            before=g[(g.nf_id==nf)&(g.window_id==start-1)]
            episodes.append(dict(method=method,episode=ep,scenario=s.scenario.iloc[0],variant=s.variant.iloc[0],
                detected=bool(len(hits)),availability=bool(len(av)),
                preexisting_security=bool(before.alarm.any()),
                delay_seconds=float(hits.decision_timestamp.min()-s.onset.min()) if len(hits) else None))
    pairs=[]
    for left,right in [('IF_SBI','IF_CONTEXT'),('IF_CONTEXT','hybrid_without_KIE'),('hybrid_without_KIE','hybrid'),('CARRY_without_witness','CARRY')]:
        a=frame[frame.method==left].set_index(KEYS); b=frame[frame.method==right].set_index(KEYS).loc[a.index]
        for attack in [0,1]:
            mask=a.attack==attack
            pairs.append(dict(left=left,right=right,attack=attack,n=int(mask.sum()),
                left_only=int((mask&a.alarm.eq(1)&b.alarm.eq(0)).sum()),right_only=int((mask&a.alarm.eq(0)&b.alarm.eq(1)).sum())))
    result=dict(seed=manifest['seed'],primary_methods=PRIMARY,metrics=totals,strata=strata,episodes=episodes,paired_discordance=pairs,
        rows_per_method=len(f),feature_sha256=hashlib.sha256((root/'test/features.csv').read_bytes()).hexdigest(),
        model_sha256=bundle['manifest']['model_sha256'],
        limitations=['Primary four methods differ in both features and algorithm; not a pure feature attribution.',
        'CARRY_without_witness removes live control/replay/availability only, retaining route/report feature checks.',
        'IF uses continuous margin AP; CARRY AP uses binary alarms and is not directly comparable as a ranking metric.',
        'CARRY_without_witness uses full-method decision time as a conservative proxy, not measured standalone runtime.',
        'Correlated windows: counts are descriptive; three runs do not support strong confidence claims.',
        'Availability is separate from security; missed episodes remain in denominators.'])
    out=root/'ablation';out.mkdir(exist_ok=True)
    (out/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    pd.DataFrame(totals).to_csv(out/'metrics.csv',index=False)
    pd.DataFrame(episodes).to_csv(out/'episodes.csv',index=False)
    frame.to_csv(out/'predictions.csv.gz',index=False,compression='gzip')
    print(json.dumps(dict(seed=result['seed'],metrics=[x for x in totals if x['primary']])))

if __name__=='__main__': run(Path(sys.argv[1]))
