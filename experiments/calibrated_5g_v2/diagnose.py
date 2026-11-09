"""Post-run diagnostics; never updates thresholds or measured decisions."""
import json
from pathlib import Path
import sys
import pandas as pd

root=Path(sys.argv[1]); out=Path(sys.argv[2]); reports=[]
for path in sorted(root.glob('calibrated-v2-5g-*')):
    if not path.is_dir():
        continue
    f=pd.read_csv(path/'test/features.csv',float_precision='round_trip')
    labels=pd.read_csv(path/'test/labels.csv').fillna({'episode_id':''})
    old=pd.read_json(path/'test/resilient_predictions.jsonl',lines=True)
    new=pd.read_json(path/'test/refined_predictions.jsonl',lines=True)
    keys=['run_id','window_id','nf_id']
    frame=f.merge(labels,on=keys,validate='one_to_one').merge(new,on=keys,validate='one_to_one',suffixes=('','_decision'))
    oldmap={tuple(r[k] for k in keys):r for r in old.to_dict('records')}
    for _,r in frame.iterrows():
        r.outputs['PREVIOUS']=oldmap[tuple(r[k] for k in keys)]['outputs']['KALL']
    false=[]
    for r in frame[frame.attack==0].itertuples():
        methods=[m for m,v in r.outputs.items() if v['security']]
        if not methods:
            continue
        history=frame[(frame.nf_id==r.nf_id)&(frame.window_id<=r.window_id)&(frame.window_id>=r.window_id-10)]
        false.append(dict(window=int(r.window_id),nf=r.nf_id,methods=methods,signals=r.signals,
            history=history[['window_id','request_count','error_rate','update_count','attack','scenario']].fillna('').to_dict('records')))
    growth=[]
    for episode,g in frame[frame.scenario=='G'].groupby('episode_id'):
        growth.append(dict(episode=episode,
            permitted_windows=int(g.signals.map(lambda s:s['permit_valid']).sum()),
            persistence_max=float(g.signals.map(lambda s:s['persistence']).max()),
            carry_max=float(g.signals.map(lambda s:s['carry']).max()),
            request_counts=g.sort_values('window_id').request_count.tolist(),
            detected={m:bool(g.outputs.map(lambda x:x[m]['security']).any()) for m in ['PREVIOUS','SPLIT','PERSIST','CARRY']}))
    normal_incidents=[]
    slot=frame[frame.nf_id=='slot-4'].sort_values('window_id')
    active=slot.outputs.map(lambda x:x['PERSIST']['availability']) & slot.attack.eq(0)
    groups=(active!=active.shift(fill_value=False)).cumsum()
    for _,g in slot[active].groupby(groups[active]):
        first=int(g.window_id.min()); previous=slot[slot.window_id==first-1]
        normal_incidents.append(dict(start=first,end=int(g.window_id.max()),windows=len(g),
            previous_scenario=None if previous.empty else previous.scenario.iloc[0],
            previous_attack=False if previous.empty else bool(previous.attack.iloc[0])))
    s4=frame[frame.scenario=='S4']
    reports.append(dict(run=path.name,false_alarms=false,growth=growth,normal_availability_segments=normal_incidents,
        refinement=json.loads((path/'frozen/refinement.json').read_text()),
        previous_thresholds=json.loads((path/'frozen/manifest.json').read_text())['thresholds'],
        s4_error_rate_min=float(s4.error_rate.min()),s4_error_rate_max=float(s4.error_rate.max()),
        s4_alarm_windows={m:int(s4.outputs.map(lambda x:x[m]['security']).sum()) for m in ['PREVIOUS','SPLIT','PERSIST','CARRY']}))
(out/'diagnostics.json').write_text(json.dumps(reports,indent=2))
print(json.dumps([dict(run=r['run'],false_windows=len(r['false_alarms']),s4=r['s4_alarm_windows']) for r in reports]))
