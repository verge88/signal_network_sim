"""Unchanged raw replay plus an independent audit of acquisition boundaries."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from experiments.robustness_5g.audit import run as replay


def run(root):
    replay(root)
    result=json.loads((root/'robustness_summary.json').read_text())
    manifest=json.loads((root/'test/manifest.json').read_text())
    mode=manifest['control_acquisition']
    f=pd.read_csv(root/'test/features.csv',float_precision='round_trip')
    labels=pd.read_csv(root/'test/labels.csv')
    online=pd.read_csv(root/'test/online_predictions.csv',float_precision='round_trip')
    predictions=pd.read_json(root/'test/refined_predictions.jsonl',lines=True)
    decisions={r.window_id:r for r in predictions[predictions.nf_id=='slot-4'].itertuples()}
    labels4=labels[labels.nf_id=='slot-4'].set_index('window_id')
    ends=f.groupby('window_id').timestamp.max()
    scored=online.groupby('window_id').decision_timestamp.max()
    controls=[json.loads(line) for line in (root/'test/control_events.jsonl').read_text().splitlines()]
    checks=result['checks']
    checks['one_control_observation_per_window']=len(controls)==3600 and {c['window_id'] for c in controls}==set(range(3600))
    checks['freshness_configuration']=mode in ['early','late'] and manifest['unchanged_freshness_limit_seconds']==.25
    checks['new_test_seed']=manifest['seed']==result['source']['source_seed']+21000+32 and manifest['actors']==32
    checks['acquisition_order']=all(
        (c['started']>=max(ends[c['window_id']],scored[c['window_id']]) if mode=='late' else c['completed']<=ends[c['window_id']])
        for c in controls)
    checks['evaluation_not_backdated']=all(c['started']<=c['completed']<=decisions[c['window_id']].evaluation_time<=decisions[c['window_id']].decision_time for c in controls)
    age=[]; wait=[]; arrived_fresh=aged_after=missing=already_late=late_at_evaluation=0
    boundary=[]
    for c in controls:
        w=c['window_id']; decision=decisions[w]
        if labels4.loc[w,'attack']!=0: continue
        if c['original'] is None:
            missing+=1; continue
        timestamp=c['original']['report']['timestamp']
        arrival_age=c['completed']-timestamp
        evaluated_age=decision.evaluation_time-timestamp
        was_fresh=0<=arrival_age<=.25
        arrived_fresh+=int(was_fresh)
        aged_after+=int(was_fresh and evaluated_age>.25)
        already_late+=int(arrival_age>.25)
        late_at_evaluation+=int(evaluated_age>.25)
        age.append(evaluated_age); wait.append(decision.evaluation_time-c['completed'])
    pred_lookup={(r.window_id,r.nf_id):r.outputs['CARRY'] for r in predictions.itertuples()}
    for episode,g in labels[labels.attack==1].groupby('episode_id'):
        start=int(g.window_id.min()); nf=g.nf_id.iloc[0]
        boundary.append(dict(episode=episode,nf_id=nf,start_window=start,
            before=pred_lookup[(start-1,nf)],first=pred_lookup[(start,nf)]))
    # Paired runners must execute the same realized statistical/semantic inputs.
    cols=['window_id','nf_id','request_count','error_rate','update_count','profile_policy_violation','forbidden_operation','route_mismatch','report_mismatch','auth_failure']
    workload=f.sort_values(['window_id','nf_id'])[cols].to_csv(index=False,float_format='%.17g').encode()
    result.update(control_acquisition=mode,realized_workload_sha256=hashlib.sha256(workload).hexdigest(),
        timing=dict(normal_control_windows=int((labels4.attack==0).sum()),normal_missing_original=missing,
            normal_reports_fresh_at_return=arrived_fresh,normal_reports_aging_after_return=aged_after,
            normal_reports_already_late_at_return=already_late,normal_reports_late_at_evaluation=late_at_evaluation,
            normal_report_age_p50_ms=float(np.median(age)*1000),normal_report_age_p95_ms=float(np.percentile(age,95)*1000),
            normal_post_return_wait_p50_ms=float(np.median(wait)*1000),normal_post_return_wait_p95_ms=float(np.percentile(wait,95)*1000)),
        episode_boundary_states=boundary)
    # Outcome criteria are reported, never used to discard an intact experiment.
    episodes=[e for e in result['episodes'] if e['method']=='CARRY']
    result['outcomes']=dict(normal_post_return_expiry_at_most_point_one_percent=aged_after/max(1,arrived_fresh)<=.001,
        C_F_R_each_four_security=all(sum(e['security'] for e in episodes if e['scenario']==s)==4 for s in ['C','F','R']),
        D_four_availability=sum(e['availability'] for e in episodes if e['scenario']=='D')==4)
    (root/'freshness_summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(freshness_audit_pass=all(checks.values()),mode=mode,timing=result['timing'],outcomes=result['outcomes'])))
    assert all(checks.values()),checks


if __name__=='__main__': run(Path(sys.argv[1]))
