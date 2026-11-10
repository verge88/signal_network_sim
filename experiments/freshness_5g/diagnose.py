"""Post-hoc boundary states, missed control cases, and paired input differences."""
import json
from pathlib import Path
import sys
import pandas as pd
import numpy as np
from experiments.robustness_5g.audit_archive import open_archive
from experiments.freshness_5g.audit import read_predictions

root=Path(sys.argv[1]); out=Path(sys.argv[2]); reports=[]; inputs={}
for path in sorted(root.glob('freshness-5g-*/freshness_summary.json')):
    summary=json.loads(path.read_text()); archive=open_archive(path.parent.with_suffix('.zip'))
    try:
        features=pd.read_csv(archive/'test/features.csv',float_precision='round_trip')
        labels=pd.read_csv(archive/'test/labels.csv')
        frame=features.merge(labels,on=['run_id','window_id','nf_id'],validate='one_to_one')
        columns=['request_count','error_rate','update_count','profile_policy_violation','forbidden_operation','route_mismatch','report_mismatch','auth_failure']
        inputs[(summary['source']['source_seed'],summary['control_acquisition'])]=features.set_index(['window_id','nf_id'])[columns].sort_index()
        predictions=read_predictions(archive/'test/refined_predictions.jsonl')
        frame=frame.merge(predictions,on=['run_id','window_id','nf_id'],validate='one_to_one')
        controls={c['window_id']:c for c in map(json.loads,(archive/'test/control_events.jsonl').read_text().splitlines())}
        decision_end_ages=[]; evaluation_ms=[]; expired_by_decision_end=0
        for r in frame[(frame.nf_id=='slot-4')&(frame.attack==0)].itertuples():
            c=controls[r.window_id]
            evaluation_ms.append(1000*(r.decision_time-r.evaluation_time))
            if c['original'] is not None:
                timestamp=c['original']['report']['timestamp']
                decision_end_ages.append(1000*(r.decision_time-timestamp))
                expired_by_decision_end+=int(0<=c['completed']-timestamp<=.25 and r.decision_time-timestamp>.25)
        cases=[]
        selected=[e for e in summary['episodes'] if e['method']=='CARRY' and ((e['scenario']=='C' and not e['security']) or (e['scenario']=='D' and e['security']))]
        for episode in selected:
            rows=[]
            for r in frame[frame.episode_id==episode['episode']].itertuples():
                c=controls[r.window_id]
                age=None if c['original'] is None else 1000*(r.evaluation_time-c['original']['report']['timestamp'])
                rows.append(dict(window_id=r.window_id,request_count=r.request_count,error_count=round(r.request_count*r.error_rate),
                    original_age_ms=age,delivered_missing=c['delivered'] is None,signals=r.signals,output=r.outputs['CARRY']))
            cases.append(dict(episode=episode,rows=rows))
        false_reason={}
        for r in frame[frame.attack==0].itertuples():
            if r.outputs['CARRY']['security']:
                reason='tail' if r.signals['separate_tail'] else 'semantic' if r.signals['semantic'] else 'temporal_or_control'
                false_reason.setdefault(r.load_regime,{}).setdefault(reason,0)
                false_reason[r.load_regime][reason]+=1
        reports.append(dict(run=path.parent.name,mode=summary['control_acquisition'],source_seed=summary['source']['source_seed'],
            cases=cases,false_reason_counts=false_reason,episode_boundary_states=summary['episode_boundary_states'],
            normal_report_age_at_decision_end_p95_ms=float(np.percentile(decision_end_ages,95)),
            normal_reports_aging_after_return_by_decision_end=expired_by_decision_end,
            normal_evaluation_duration_p95_ms=float(np.percentile(evaluation_ms,95)),
            normal_evaluation_duration_max_ms=float(max(evaluation_ms))))
    finally: archive.root.close()
pairs=[]
for seed in [33101,33201,33301]:
    early,late=inputs[(seed,'early')],inputs[(seed,'late')]
    assert early.index.equals(late.index)
    different=early.ne(late)
    pairs.append(dict(source_seed=seed,different_cells=int(different.sum().sum()),
        by_feature={k:int(v) for k,v in different.sum().items()},
        first_ten_different_rows=[dict(window_id=int(key[0]),nf_id=key[1],early=early.loc[key].to_dict(),late=late.loc[key].to_dict()) for key in different.index[different.any(axis=1)][:10]]))
(out/'diagnostics.json').write_text(json.dumps(dict(runs=reports,paired_inputs=pairs),indent=2))
print(json.dumps(dict(paired_inputs=pairs,cases_per_run=[dict(mode=r['mode'],seed=r['source_seed'],cases=len(r['cases'])) for r in reports]),indent=2))
