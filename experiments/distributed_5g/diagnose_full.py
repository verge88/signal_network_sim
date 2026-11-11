"""Descriptive normal-regime breakdown on the already frozen full experiment."""
import argparse
import json
from pathlib import Path

import pandas as pd

from experiments.live_5g.evaluate import KEYS, sha256


def diagnose(root, output):
    labels = pd.read_csv(root/'test/labels.csv')
    online = pd.read_csv(root/'test/online_predictions.csv',usecols=KEYS+['method','alarm'])
    online = online[online.method.isin(['rules','IF_SBI','IF_CONTEXT'])]
    records = []
    with (root/'test/distributed_predictions.jsonl').open() as stream:
        for line in stream:
            window = json.loads(line)
            for nf, decision in window['decisions'].items():
                records.append(dict(window_id=window['window'],nf_id=nf,
                    FULL_HYBRID=decision['security'],ORIGINAL_HYBRID=decision['original_security'],
                    availability=decision['availability'],unknown=decision['unknown']))
    records = pd.DataFrame(records)
    merged = labels.merge(records,on=['window_id','nf_id'],validate='one_to_one')
    baselines = online.pivot(index=KEYS,columns='method',values='alarm').reset_index()
    merged = merged.merge(baselines,on=KEYS,validate='one_to_one')
    assert len(merged) == 115200
    normal = merged[merged.attack == 0]
    result = []
    for regime, group in normal.groupby('load_regime'):
        for method in ['rules','IF_SBI','IF_CONTEXT','FULL_HYBRID','ORIGINAL_HYBRID']:
            fp = int(group[method].sum())
            result.append(dict(load_regime=regime,method=method,normal_windows=len(group),fp=fp,fpr=fp/len(group)))
    positive = []
    for (scenario,variant), group in merged[merged.attack == 1].groupby(['scenario','variant']):
        positive.append(dict(scenario=scenario,variant=variant,windows=len(group),
            security=int(group.FULL_HYBRID.sum()),availability=int(group.availability.sum()),unknown=int(group.unknown.sum())))
    report = dict(normal_regimes=result,positive_states=positive,
        source_labels_sha256=sha256(root/'test/labels.csv'),
        note='Descriptive stratification after acquisition; no detector or labels modified.')
    output.write_text(json.dumps(report,indent=2))
    print(json.dumps(report))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('root',type=Path)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    diagnose(args.root,args.output)
