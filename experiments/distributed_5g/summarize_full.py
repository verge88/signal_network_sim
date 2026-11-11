"""Aggregate complete full-method repetitions without dropping failures or misses."""
import argparse
import json
from pathlib import Path
import shutil

import pandas as pd

from experiments.live_5g.evaluate import sha256


def summarize(source, output):
    reports, episodes = [], []
    output.mkdir(parents=True, exist_ok=True)
    for seed in (89501, 89502, 89503):
        directory = source/str(seed)
        report = json.loads((directory/'report.json').read_text())
        assert report['seed'] == seed and report['audit_pass'] and report['rows'] == 115200
        reports.append(report)
        frame = pd.read_csv(directory/'episodes.csv')
        assert len(frame) == 36*6 and not frame.duplicated(['method','episode']).any()
        frame['seed'] = seed
        episodes.append(frame)
        shutil.copyfile(directory/'report.json', output/f'{seed}-report.json')
        shutil.copyfile(directory/'model_manifest.json', output/f'{seed}-model.json')
        shutil.copyfile(directory/'integrity.json', output/f'{seed}-integrity.json')
    assert len({r['source_commit'] for r in reports}) == 1
    assert len({r['model_sha256'] for r in reports}) == 3
    assert len({r['zone_models_sha256'] for r in reports}) == 1
    combined = pd.concat(episodes, ignore_index=True)
    metrics = []
    names = [m['method'] for m in reports[0]['metrics']]
    for name in names:
        values = [next(m for m in r['metrics'] if m['method'] == name) for r in reports]
        total = dict(method=name, **{key:sum(m[key] for m in values) for key in
            ('tp','fp','fn','tn','episodes','episodes_with_alarm','episodes_with_new_alarm')})
        total['recall'] = total['tp']/(total['tp']+total['fn'])
        total['fpr'] = total['fp']/(total['fp']+total['tn'])
        total['precision'] = total['tp']/(total['tp']+total['fp']) if total['tp']+total['fp'] else None
        total['episode_recall_new_alarm'] = total['episodes_with_new_alarm']/total['episodes']
        total['per_run_average_precision'] = [m['average_precision'] for m in values]
        # Fresh models have different score scales; pooled AP is not inferred from run AP.
        total['per_run_recall'] = [m['recall'] for m in values]
        total['per_run_fpr'] = [m['fpr'] for m in values]
        delays = combined.loc[(combined.method == name)&combined.new_alarm, 'new_alarm_delay_seconds'].dropna()
        total['detected_episode_delay_median_seconds'] = float(delays.median()) if len(delays) else None
        total['detected_episode_delay_max_seconds'] = float(delays.max()) if len(delays) else None
        metrics.append(total)
    strata = combined.groupby(['method','scenario','variant']).agg(
        episodes=('episode','size'),new_alarms=('new_alarm','sum'),alarm_present=('alarm_present','sum')).reset_index()
    summary = dict(source_commit=reports[0]['source_commit'],runs=3,nf_windows=sum(r['rows'] for r in reports),
        metrics=metrics,scenario_episodes=strata.to_dict('records'),
        data_mining_increment={k:sum(r['data_mining_increment'][k] for r in reports) for k in ('additional_tp','additional_fp')},
        per_run_overhead=[dict(seed=r['seed'],**r['overhead']) for r in reports],
        availability_windows=sum(r['availability_windows'] for r in reports),
        unknown_windows=sum(r['unknown_windows'] for r in reports),
        fpr_target_met_all_runs=all(r['fpr_target_met'] for r in reports),
        source_report_sha256={str(seed):sha256(source/str(seed)/'report.json') for seed in (89501,89502,89503)},
        limitations=reports[0]['limitations'])
    (output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    combined.to_csv(output/'episodes.csv',index=False)
    strata.to_csv(output/'scenario_episodes.csv',index=False)
    print(json.dumps(summary),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('source',type=Path)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    summarize(args.source,args.output)
