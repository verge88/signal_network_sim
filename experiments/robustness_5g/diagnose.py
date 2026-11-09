"""Post-run timing diagnosis and deterministic false-alarm examples."""
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from experiments.robustness_5g.audit_archive import open_archive

root=Path(sys.argv[1]); out=Path(sys.argv[2]); reports=[]
for p in sorted(root.glob('robustness-5g-*')):
    if not p.is_dir() or not (p/'robustness_summary.json').exists(): continue
    run_name=p.name
    summary=json.loads((p/'robustness_summary.json').read_text())
    p=open_archive(p.with_suffix('.zip'))
    labels=pd.read_csv(p/'test/labels.csv')
    control_labels=labels[labels.nf_id=='slot-4'].set_index('window_id')
    normal_keys=set(zip(labels.loc[labels.attack==0,'window_id'],labels.loc[labels.attack==0,'nf_id']))
    boundaries=[]; boundary_outputs={}
    for episode,g in labels[labels.attack==1].groupby('episode_id'):
        boundaries.append(dict(episode=episode,nf_id=g.nf_id.iloc[0],start_window=int(g.window_id.min())))
    boundary_keys={(b['start_window']+delta,b['nf_id']) for b in boundaries for delta in [-1,0]}
    evaluation={}; examples={}; false_reasons={}
    for line in (p/'test/refined_predictions.jsonl').open():
        r=json.loads(line)
        if (r['window_id'],r['nf_id']) in boundary_keys:
            boundary_outputs[(r['window_id'],r['nf_id'])]=r['outputs']['CARRY']
        if r['nf_id']=='slot-4': evaluation[r['window_id']]=r['evaluation_time']
        if (r['window_id'],r['nf_id']) not in normal_keys: continue
        regime=['reference','higher_load','cyclic','bursty_recovery'][r['window_id']//900]
        for method in ['SPLIT','PERSIST','CARRY']:
            if not r['outputs'][method]['security']: continue
            key=regime+'/'+method
            reason='separate_tail' if r['signals']['separate_tail'] else 'semantic' if r['signals']['semantic'] else 'temporal_or_control'
            false_reasons.setdefault(key,{}).setdefault(reason,0); false_reasons[key][reason]+=1
            if len(examples.setdefault(key,[]))<5:
                examples[key].append(dict(window=r['window_id'],nf=r['nf_id'],signals=r['signals']))
    ages=[]; waits=[]; late=missing=aged_after_return=already_late=0
    for line in (p/'test/control_events.jsonl').open():
        c=json.loads(line)
        if control_labels.loc[c['window_id'],'attack']!=0: continue
        if c['original'] is None:
            missing+=1; continue
        age=evaluation[c['window_id']]-c['original']['report']['timestamp']
        ages.append(age); late+=int(age>.25)
        arrival_age=c['completed']-c['original']['report']['timestamp']
        waits.append(evaluation[c['window_id']]-c['completed'])
        aged_after_return+=int(age>.25 and 0<=arrival_age<=.25)
        already_late+=int(arrival_age>.25)
    reports.append(dict(run=run_name,actors=summary['actors'],source_seed=summary['source']['source_seed'],
        episode_boundary_states=[dict(b,before=boundary_outputs.get((b['start_window']-1,b['nf_id'])),first=boundary_outputs[(b['start_window'],b['nf_id'])]) for b in boundaries],
        normal_control_windows=len(control_labels[control_labels.attack==0]),
        normal_missing_original_reports=missing,normal_original_reports_older_than_250ms=late,
        normal_reports_aging_past_250ms_after_return=aged_after_return,
        normal_reports_already_late_at_return=already_late,
        normal_post_return_wait_p95_ms=float(np.percentile(waits,95)*1000) if waits else None,
        normal_report_age_p50_ms=float(np.median(ages)*1000) if ages else None,
        normal_report_age_p95_ms=float(np.percentile(ages,95)*1000) if ages else None,
        growth=[e for e in summary['episodes'] if e['method']=='CARRY' and e['scenario']=='G'],
        false_alarm_reason_counts=false_reasons,first_five_false_examples_per_regime_method=examples))
    p.root.close()
(out/'diagnostics.json').write_text(json.dumps(reports,indent=2))
print(json.dumps([{k:r[k] for k in ['run','normal_missing_original_reports','normal_original_reports_older_than_250ms','normal_report_age_p95_ms']} for r in reports],indent=2))
