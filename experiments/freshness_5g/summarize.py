"""Pool all six preregistered runs, retaining matched seed and runner details."""
import json
from pathlib import Path
import sys
import numpy as np

root=Path(sys.argv[1]); out=Path(sys.argv[2]); out.mkdir(parents=True,exist_ok=True)
paths=sorted(root.glob('freshness-5g-*/freshness_summary.json'))
assert len(paths)==6, f'Expected six independently replayed runs, got {len(paths)}'
runs=[json.loads(p.read_text()) for p in paths]
assert {(r['source']['source_seed'],r['control_acquisition']) for r in runs}=={(s,m) for s in [33101,33201,33301] for m in ['early','late']}
assert all(all(r['checks'].values()) for r in runs)
by_mode={}
for mode in ['early','late']:
    selected=[r for r in runs if r['control_acquisition']==mode]
    metrics={}
    for scope in ['all','slot-3','slot-4']:
        metrics[scope]={}
        for method in ['PREVIOUS','SPLIT','PERSIST','CARRY']:
            records=[r['metrics'][scope][method] for r in selected]
            x={k:sum(v[k] for v in records) for k in records[0]}
            x.update(security_fpr=x['security_fp']/x['normal_windows'],security_recall=x['security_tp']/x['attack_windows'],
                security_precision=x['security_tp']/max(1,x['security_tp']+x['security_fp']))
            metrics[scope][method]=x
    timing={k:sum(r['timing'][k] for r in selected) for k in selected[0]['timing'] if not k.endswith('_ms')}
    timing['post_return_expiry_fraction']=timing['normal_reports_aging_after_return']/max(1,timing['normal_reports_fresh_at_return'])
    episodes={}
    for method in ['PREVIOUS','SPLIT','PERSIST','CARRY']:
        e=[e for r in selected for e in r['episodes'] if e['method']==method]
        def count(g):
            return dict(n=len(g),security=sum(x['security'] for x in g),availability=sum(x['availability'] for x in g),operational=sum(x['operational'] for x in g))
        delays=[x['security_delay'] for x in e if x['security']]
        episodes[method]=dict(**count(e),scenarios={s:count([x for x in e if x['scenario']==s]) for s in sorted({x['scenario'] for x in e})},
            growth={v:count([x for x in e if x['scenario']=='G' and x['variant']==v]) for v in ['fast','slow','intermittent','inside_permit']},
            security_delay_p50_s=float(np.median(delays)) if delays else None,security_delay_p95_s=float(np.percentile(delays,95)) if delays else None)
    regimes={}
    for regime in ['reference','higher_load','cyclic','bursty_recovery']:
        records=[r['regime_metrics'][regime]['CARRY'] for r in selected]
        regimes[regime]={k:sum(v[k] for v in records) for k in records[0]}
    boundaries=[b for r in selected for b in r['episode_boundary_states']]
    by_mode[mode]=dict(metrics=metrics,timing=timing,episodes=episodes,regimes=regimes,
        preregistered_outcomes_all_three={k:all(r['outcomes'][k] for r in selected) for k in selected[0]['outcomes']},
        preexisting_availability_episodes=sum(b['before']['availability'] for b in boundaries),
        preexisting_security_episodes=sum(b['before']['security'] for b in boundaries))
pairs=[]
for seed in [33101,33201,33301]:
    pair={r['control_acquisition']:r for r in runs if r['source']['source_seed']==seed}
    early,late=pair['early'],pair['late']
    pairs.append(dict(source_seed=seed,test_seed=early['test_seed'],realized_workload_matches=early['realized_workload_sha256']==late['realized_workload_sha256'],
        early_timing=early['timing'],late_timing=late['timing'],
        normal_availability_difference_late_minus_early=late['metrics']['slot-4']['CARRY']['availability_normal_windows']-early['metrics']['slot-4']['CARRY']['availability_normal_windows']))
result=dict(run_id=42771575096,source_commit='5e5767ea52a54743e1458eb270336754cf3dd2b6',runs=6,by_mode=by_mode,pairs=pairs,
    run_details=[dict(source=p.parent.name,mode=r['control_acquisition'],source_seed=r['source']['source_seed'],test_seed=r['test_seed'],
        timing=r['timing'],outcomes=r['outcomes'],metrics=r['metrics'],overhead=r['overhead'],
        interval_p50_s=r['observation_interval_p50_s'],interval_p95_s=r['observation_interval_p95_s'],
        realized_workload_sha256=r['realized_workload_sha256']) for p,r in zip(paths,runs)],
    episodes=[dict(e,mode=r['control_acquisition'],source_seed=r['source']['source_seed']) for r in runs for e in r['episodes']])
(out/'summary.json').write_text(json.dumps(result,indent=2))
print(json.dumps(dict(pairs=pairs,by_mode={m:dict(timing=v['timing'],outcomes=v['preregistered_outcomes_all_three'],episodes=v['episodes']['CARRY']) for m,v in by_mode.items()}),indent=2))
