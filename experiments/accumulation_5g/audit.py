"""Independent arithmetic replay and paired metrics on prospective data."""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from experiments.robustness_5g.audit import run as previous_audit
from experiments.resilient_5g.audit import mac
from experiments.live_5g.evaluate import metrics


def run(root):
    previous_audit(root)
    model = json.loads((root/'frozen/accumulation.json').read_text())
    manifest = json.loads((root/'test/manifest.json').read_text())
    origin = json.loads((root/'frozen_origin.json').read_text())
    assert manifest['seed'] == origin['source_seed'] + 41032
    assert manifest['accumulation'] == model and model['created_at'] < manifest['started_at']
    assert model['calibration_run'] != manifest['run_id']
    features = pd.read_csv(root/'test/features.csv',float_precision='round_trip')
    labels = pd.read_csv(root/'test/labels.csv')
    predictions = [json.loads(s) for s in (root/'test/refined_predictions.jsonl').read_text().splitlines()]
    old = {(r['window_id'],r['nf_id']):r for r in map(json.loads,(root/'test/resilient_predictions.jsonl').read_text().splitlines())}
    controls = {r['window_id']:r for r in map(json.loads,(root/'test/control_events.jsonl').read_text().splitlines())}
    key = bytes.fromhex((root/'test/lab_control_key.hex').read_text())
    decisions = {(r['window_id'],r['nf_id']):r for r in predictions}
    accumulators = {}
    for window, group in features.groupby('window_id',sort=True):
        for row in group.to_dict('records'):
            identity = row['nf_id']
            peers = group.loc[group.nf_id != identity,'request_count'].to_numpy()
            center = float(np.median(peers))
            scale = max(1.,float(np.median(np.abs(peers-center))))
            residual = (row['request_count']-center)/scale
            decision = decisions[window,identity]
            allowed = decision['signals']['permit_valid']  # Independently verified by previous_audit.
            value = 0. if allowed else max(0., .99*accumulators.get(identity,0.) + min(4.,max(-1.,residual)) - .5)
            accumulators[identity] = value
            measured = dict(row,error_count=round(row['request_count']*row['error_rate']))
            tail = any(measured[name]>limit for name,limit in model['feature_thresholds'].items())
            assert decision['accumulation'] == dict(residual=residual,value=value,tail=tail)
            security = tail or decision['signals']['semantic']
            state = 'alarm' if security else 'normal'
            if identity == 'slot-4':
                control = controls[window]
                envelope = control['delivered']
                if envelope is None:
                    state = 'alarm' if security else 'unknown'
                else:
                    report = envelope['report']
                    fresh = report['nonce']==control['nonce'] and report['instance']==control['instance'] and 0<=decision['evaluation_time']-report['timestamp']<=.25
                    security = security or not mac(envelope,key) or (fresh and report['config_hash']!=control['expected'])
                    state = 'alarm' if security else 'normal' if fresh else 'unknown'
            security = state=='alarm' or old[window,identity]['signals']['replay'] or value>model['threshold']
            availability = old[window,identity]['signals']['availability']
            expected = dict(security=bool(security),availability=availability,state='security' if security else 'availability' if availability else state)
            assert decision['outputs']['ACCUM']==expected
    frame = pd.DataFrame(predictions).merge(labels,on=['run_id','window_id','nf_id'],validate='one_to_one')
    totals=[]
    episodes=[]
    strata=[]
    for method in ['CARRY','ACCUM']:
        part=frame.copy()
        part['alarm']=part.outputs.map(lambda x:x[method]['security']).astype(int)
        part['score']=part.alarm.astype(float)
        totals.append(dict(method=method,**metrics(part)))
        for regime,group in part.groupby('load_regime'):
            strata.append(dict(method=method,load_regime=regime,**metrics(group)))
        for episode,group in part[part.attack==1].groupby('episode_id'):
            hits=group[group.alarm==1]
            episodes.append(dict(method=method,episode=episode,scenario=group.scenario.iloc[0],variant=group.variant.iloc[0],
                detected=bool(len(hits)),attack_windows=len(group),security_windows=int(group.alarm.sum()),
                availability=bool(group.outputs.map(lambda x:x[method]['availability']).any()),
                delay_seconds=float(hits.decision_time.min()-group.onset.min()) if len(hits) else None))
    result=dict(seed=manifest['seed'],source_commit=(root/'source_commit.txt').read_text().strip(),
        model=model,model_sha256=hashlib.sha256((root/'frozen/accumulation.json').read_bytes()).hexdigest(),
        independent_replay_pass=True,metrics=totals,strata=strata,episodes=episodes,
        limitations=['Replacement changes both load thresholds and trend, not solely temporal memory.',
        'Peer baseline assumes a minority of abnormal NF and comparable peer workload.',
        'Calibration used earlier test data and reset state across excluded windows.',
        'Binary AP and window confidence intervals are descriptive; windows are dependent.'])
    (root/'accumulation_summary.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps(dict(independent_replay_pass=True,seed=manifest['seed'],metrics=totals)))

if __name__=='__main__':run(Path(sys.argv[1]))
