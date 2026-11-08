"""Explain all held-out false alarms and the S4 calibration regression."""
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from experiments.live_5g.evaluate import tail_scores

root=Path(sys.argv[1]); out=Path(sys.argv[2]); result=[]
for path in sorted(root.glob('resilient-5g-*')):
    if not path.is_dir():
        continue
    manifest=json.loads((path/'frozen/manifest.json').read_text())
    f=pd.read_csv(path/'test/features.csv')
    labels=pd.read_csv(path/'test/labels.csv').fillna({'episode_id':''})
    decisions=pd.read_json(path/'test/resilient_predictions.jsonl',lines=True)
    frame=f.merge(labels,on=['run_id','window_id','nf_id'],validate='one_to_one').merge(
        decisions,on=['run_id','window_id','nf_id'],validate='one_to_one',suffixes=('','_decision'))
    false=[]
    for r in frame[frame.attack==0].itertuples():
        methods=[m for m,v in r.outputs.items() if v['security']]
        if not methods:
            continue
        history=frame[(frame.nf_id==r.nf_id)&(frame.window_id<=r.window_id)&(frame.window_id>=r.window_id-10)]
        false.append(dict(window=int(r.window_id),nf=r.nf_id,methods=methods,signals=r.signals,
            history=history[['window_id','request_count','attack','scenario']].fillna('').to_dict('records')))
    notices=[]
    slot=frame[frame.nf_id=='slot-4'].sort_values('window_id')
    active=slot.outputs.map(lambda x:x['KALL']['availability']) & slot.attack.eq(0)
    groups=(active!=active.shift(fill_value=False)).cumsum()
    for _,part in slot[active].groupby(groups[active]):
        first=int(part.window_id.min()); previous=slot[slot.window_id==first-1]
        notices.append(dict(start=first,end=int(part.window_id.max()),windows=len(part),
            previous_scenario=None if previous.empty else previous.scenario.iloc[0],
            previous_attack=False if previous.empty else bool(previous.attack.iloc[0])))
    s4=frame[frame.scenario=='S4']
    cal=pd.read_csv(path/'cal/features.csv')
    cal['tail_score']=tail_scores(manifest['tail_model'],cal)
    s4scores=tail_scores(manifest['tail_model'],s4)
    result.append(dict(run=path.name,false_alarms=false,normal_availability_segments=notices,
        tail_threshold=manifest['thresholds']['robust_tail'],tail_model=manifest['tail_model'],
        s4_score_min=float(np.min(s4scores)),s4_score_max=float(np.max(s4scores)),
        s4_alarm_windows=int((s4scores>manifest['thresholds']['robust_tail']).sum()),
        calibration_extremes=cal.nlargest(3,'tail_score')[['window_id','nf_id','request_count','error_rate','update_count','tail_score']].to_dict('records')))
(out/'diagnostics.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
