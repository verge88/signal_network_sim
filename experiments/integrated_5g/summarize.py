"""Aggregate every planned runner without filtering unsuccessful metrics."""
import hashlib
import csv
import json
from pathlib import Path
import sys
import numpy as np

root = Path(sys.argv[1])
out = Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
paths = sorted(root.glob('integrated-5g-*/integrated_summary.json'))
if len(paths) != 3:
    raise SystemExit(f'Expected all 3 results, found {len(paths)}')
runs = [json.loads(p.read_text()) for p in paths]
if not all(all(r['checks'].values()) for r in runs):
    raise SystemExit('Independent replay failed; do not publish pooled success')
pooled = {'runs':3, 'sources':[{'name':p.parent.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths],
          'metrics':{}, 'episodes':[], 'overhead':[r['overhead'] for r in runs]}
for scope in ['all','slot-4']:
    pooled['metrics'][scope] = {}
    for method in ['H','O','K']:
        groups = [r['metrics'][scope][method] for r in runs]
        m = {k:sum(g[k] for g in groups) for k in ['tp','fp','fn','normal_windows','attack_windows','unknown_normal','unknown_attack']}
        m.update(recall=m['tp']/m['attack_windows'], fpr=m['fp']/m['normal_windows'],
                 normal_coverage=1-m['unknown_normal']/m['normal_windows'],
                 conditional_fpr=m['fp']/max(1,m['normal_windows']-m['unknown_normal']),
                 false_alarms_per_sampled_nf_hour=m['fp']/(m['normal_windows']/3600))
        m['scenarios']={}
        for scenario in groups[0]['scenarios']:
            m['scenarios'][scenario] = {k:sum(g['scenarios'][scenario][k] for g in groups) for k in ['n','alarms','unknown']}
        pooled['metrics'][scope][method] = m
pooled['episodes'] = [e for r in runs for e in r['episodes']]
pooled['episode_summary'] = {}
for method in ['H','O','K']:
    es = [e for e in pooled['episodes'] if e['method']==method]
    delays = [e['delay_seconds'] for e in es if e['detected']]
    pooled['episode_summary'][method] = dict(n=len(es),detected=len(delays),missed=len(es)-len(delays),
       median_detected_delay_s=float(np.median(delays)) if delays else None,
       p95_detected_delay_s=float(np.percentile(delays,95)) if delays else None,
       scenarios={s:dict(n=sum(e['scenario']==s for e in es),detected=sum(e['scenario']==s and e['detected'] for e in es)) for s in sorted({e['scenario'] for e in es})})
pooled['normal_control_unknown_reasons'] = {m:{} for m in ['O','K']}
for path in paths:
    td = path.parent/'test'
    controls = {r['window_id']:r for r in map(json.loads,(td/'control_events.jsonl').read_text().splitlines())}
    with (td/'labels.csv').open() as handle:
        normal_windows = {int(r['window_id']) for r in csv.DictReader(handle) if r['nf_id']=='slot-4' and r['attack']=='0'}
    for r in map(json.loads,(td/'integrated_predictions.jsonl').read_text().splitlines()):
        if r['nf_id']!='slot-4' or r['window_id'] not in normal_windows:
            continue
        c = controls[r['window_id']]
        envelope = c['delivered']
        if envelope is None:
            reason = 'report_missing'
        elif envelope['report']['nonce'] != c['nonce']:
            reason = 'nonce_mismatch'
        else:
            age_at_receipt = c['completed']-envelope['report']['timestamp']
            reason = 'stale_at_receipt' if age_at_receipt > .25 else 'stale_before_joint_decision'
        for method in ['O','K']:
            if r['states'][method]=='unknown':
                counts = pooled['normal_control_unknown_reasons'][method]
                counts[reason] = counts.get(reason,0)+1
(out/'summary.json').write_text(json.dumps(pooled,indent=2))
print(json.dumps({k:v for k,v in pooled.items() if k not in ['episodes','sources']},indent=2))
