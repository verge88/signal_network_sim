"""Synthetic fixtures only; never experimental evidence."""
import json
import pytest
from experiments.ablation_5g.summarize import run

def test_aggregation_keeps_misses_and_rejects_missing_run(tmp_path):
    source=tmp_path/'inputs';source.mkdir()
    methods=['rules','IF_SBI','IF_CONTEXT','CARRY']
    for seed in [64133,64233,64333]:
        rows=[];episodes=[]
        for method in methods:
            hit=method=='CARRY'
            rows.append(dict(method=method,tp=36 if hit else 0,fp=0,fn=0 if hit else 36,tn=115164,n=115200,
                recall=float(hit),fpr=0,precision=1 if hit else None))
            for scenario in ['S1','S2','S3','S4','C','F','R','D','G']:
                for i in range(4):episodes.append(dict(method=method,scenario=scenario,detected=hit,availability=False,delay_seconds=1 if hit else None))
        d=source/str(seed);d.mkdir()
        (d/'report.json').write_text(json.dumps(dict(seed=seed,rows_per_method=115200,primary_methods=methods,
            metrics=rows,episodes=episodes,paired_discordance=[])))
    run(source,tmp_path/'output')
    result=json.loads((tmp_path/'output/summary.json').read_text())
    assert result['rows_per_method']==345600
    assert next(m for m in result['metrics'] if m['method']=='IF_SBI')['detected_episodes']==0
    assert next(m for m in result['metrics'] if m['method']=='CARRY')['detected_episodes']==108
    (source/'64333/report.json').unlink()
    with pytest.raises(AssertionError):run(source,tmp_path/'incomplete')
