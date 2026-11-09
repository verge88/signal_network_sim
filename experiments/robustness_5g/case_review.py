"""Post-hoc evidence for missed C, incidental D alarm and permitted fast G."""
import json,sys
from pathlib import Path
import pandas as pd
from experiments.robustness_5g.audit_archive import open_archive
root=Path(sys.argv[1]); out=Path(sys.argv[2]); result=[]
s=json.loads((out/'summary.json').read_text())
cases=[e for e in s['episodes'] if e['method']=='CARRY' and (
    (e['scenario']=='C' and not e['security']) or
    (e['scenario']=='D' and e['security']) or
    (e['variant']=='fast' and not e['security']))]
for e in cases:
    archive=next(root.glob(f"robustness-5g-{e['actors']}-{e['source_seed']}-*.zip"))
    p=open_archive(archive)
    try:
        labels=pd.read_csv(p/'test/labels.csv'); target=labels[labels.episode_id==e['episode']]
        ids=set(target.window_id); nf=target.nf_id.iloc[0]
        features=pd.read_csv(p/'test/features.csv',float_precision='round_trip')
        features=features[(features.nf_id==nf)&features.window_id.isin(ids)].set_index('window_id')
        controls={r['window_id']:r for r in map(json.loads,(p/'test/control_events.jsonl').read_text().splitlines()) if r['window_id'] in ids}
        rows=[]
        for r in map(json.loads,(p/'test/refined_predictions.jsonl').read_text().splitlines()):
            if r['window_id'] not in ids or r['nf_id']!=nf: continue
            c=controls[r['window_id']]; f=features.loc[r['window_id']]
            age=None if c['original'] is None else r['evaluation_time']-c['original']['report']['timestamp']
            rows.append(dict(window_id=r['window_id'],request_count=int(f.request_count),
                error_count=round(f.request_count*f.error_rate),update_count=int(f.update_count),
                original_report_age_ms=1000*age if nf=='slot-4' and age is not None else None,
                delivered_missing=c['delivered'] is None if nf=='slot-4' else None,
                signals=r['signals'],output=r['outputs']['CARRY']))
        result.append(dict(episode=e,nf_id=nf,rows=rows))
    finally: p.root.close()
(out/'case_review.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
