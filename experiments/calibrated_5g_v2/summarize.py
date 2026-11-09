"""Pool all planned results and keep security separate from operational notices."""
import json
from pathlib import Path
import sys
import numpy as np

root=Path(sys.argv[1]); out=Path(sys.argv[2]); out.mkdir(parents=True,exist_ok=True)
paths=sorted(root.glob('calibrated-v2-5g-*/calibrated_summary.json'))
if len(paths)!=3:
    raise SystemExit(f'Expected three complete runners, found {len(paths)}')
runs=[json.loads(p.read_text()) for p in paths]
assert all(all(r['checks'].values()) for r in runs)
methods=list(runs[0]['metrics']['all'])
result=dict(runs=3,sources=[p.parent.name for p in paths],metrics={},episode_summary={},
            thresholds=[{k:v for k,v in r['thresholds'].items() if k!='source_hashes'} for r in runs],
            overhead=[r['overhead'] for r in runs],episodes=[e for r in runs for e in r['episodes']])
for scope in ['all','slot-3','slot-4']:
    result['metrics'][scope]={}
    for method in methods:
        records=[r['metrics'][scope][method] for r in runs]
        metrics={k:sum(x[k] for x in records) for k in records[0]}
        metrics.update(security_recall=metrics['security_tp']/metrics['attack_windows'],
            security_fpr=metrics['security_fp']/metrics['normal_windows'],
            security_fp_per_sampled_nf_hour=metrics['security_fp']/(metrics['normal_windows']/3600))
        result['metrics'][scope][method]=metrics
for method in methods:
    episodes=[e for e in result['episodes'] if e['method']==method]
    delays=[e['security_delay'] for e in episodes if e['security']]
    operational=[e['operational_delay'] for e in episodes if e['operational']]
    result['episode_summary'][method]=dict(n=len(episodes),
       security=sum(e['security'] for e in episodes),availability=sum(e['availability'] for e in episodes),
       operational=sum(e['operational'] for e in episodes),
       median_security_delay_s=float(np.median(delays)) if delays else None,
       p95_security_delay_s=float(np.percentile(delays,95)) if delays else None,
       median_operational_delay_s=float(np.median(operational)) if operational else None,
       scenarios={s:dict(n=sum(e['scenario']==s for e in episodes),
           security=sum(e['scenario']==s and e['security'] for e in episodes),
           availability=sum(e['scenario']==s and e['availability'] for e in episodes),
           operational=sum(e['scenario']==s and e['operational'] for e in episodes))
           for s in sorted({e['scenario'] for e in episodes})})
(out/'summary.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k not in ['episodes','sources']},indent=2))
