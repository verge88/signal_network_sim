import json
from pathlib import Path
import sys
import numpy as np

root=Path(sys.argv[1]); out=Path(sys.argv[2]); out.mkdir(parents=True,exist_ok=True)
paths=sorted(root.glob('robustness-5g-*/robustness_summary.json'))
assert len(paths)==9, f'Expected 9 runs, found {len(paths)}'
runs=[json.loads(p.read_text()) for p in paths]
assert all(all(r['checks'].values()) for r in runs)
assert {(r['actors'],r['source']['source_seed']) for r in runs}=={(a,s) for a in [8,16,32] for s in [33101,33201,33301]}
methods=['PREVIOUS','SPLIT','PERSIST','CARRY']
def pool(selected):
    result=dict(metrics={},episodes={},regimes={})
    for scope in ['all','slot-3','slot-4']:
        result['metrics'][scope]={}
        for m in methods:
            records=[r['metrics'][scope][m] for r in selected]
            x={k:sum(v[k] for v in records) for k in records[0]}
            x['security_fpr']=x['security_fp']/x['normal_windows']
            x['security_recall']=x['security_tp']/x['attack_windows']
            x['security_fp_per_sampled_nf_hour']=x['security_fp']/(x['normal_windows']/3600)
            result['metrics'][scope][m]=x
    for m in methods:
        e=[e for r in selected for e in r['episodes'] if e['method']==m]
        delays=[x['security_delay'] for x in e if x['security']]
        def count(group):
            return dict(n=len(group),security=sum(x['security'] for x in group),availability=sum(x['availability'] for x in group),operational=sum(x['operational'] for x in group))
        result['episodes'][m]=dict(**count(e),
            scenarios={s:count([x for x in e if x['scenario']==s]) for s in sorted({x['scenario'] for x in e})},
            growth={v:count([x for x in e if x['scenario']=='G' and x['variant']==v]) for v in ['fast','slow','intermittent','inside_permit']},
            median_security_delay_s=float(np.median(delays)) if delays else None,
            p95_security_delay_s=float(np.percentile(delays,95)) if delays else None,
            normal_security_segments=sum(r['normal_security_segments'][m] for r in selected))
    for regime in ['reference','higher_load','cyclic','bursty_recovery']:
        result['regimes'][regime]={}
        for m in methods:
            records=[r['regime_metrics'][regime][m] for r in selected]
            x={k:sum(v[k] for v in records) for k in records[0]}
            x['security_fpr']=x['security_fp']/x['normal_windows']
            result['regimes'][regime][m]=x
    return result
result=dict(runs=9,overall=pool(runs),by_actors={str(a):pool([r for r in runs if r['actors']==a]) for a in [8,16,32]},
    run_details=[dict(source=p.parent.name,actors=r['actors'],source_seed=r['source']['source_seed'],test_seed=r['test_seed'],
        thresholds={k:v for k,v in r['thresholds'].items() if k!='source_hashes'},
        overhead=r['overhead'],interval_p50_s=r['observation_interval_p50_s'],interval_p95_s=r['observation_interval_p95_s'],
        metrics=r['metrics'],normal_security_segments=r['normal_security_segments']) for p,r in zip(paths,runs)],
    episodes=[dict(e,actors=r['actors'],source_seed=r['source']['source_seed']) for r in runs for e in r['episodes']])
(out/'summary.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result['overall'],indent=2))
