"""Predeclared bounded discovery-workload validation with frozen detectors."""
import argparse
import json
from pathlib import Path

import pandas as pd
from experiments.distributed_5g.paired import BASELINES, summarize
from experiments.live_5g.evaluate import KEYS, sha256

SEEDS = (87301, 87302, 87303, 88401, 88402, 88403)
FPR_INCREMENT_LIMIT = .005


def workload(index, window, seed):
    profile = SEEDS.index(seed) % 3
    baseline = (2, 5, 2 + window % 12 // 3)[profile]
    for cycle, start in enumerate((24, 77, 130)):
        if start <= window < start + 24 and index % 8 == 5 + cycle:
            kind = ('slow', 'fast', 'intermittent')[(index // 8 + cycle) % 3]
            elapsed = window - start
            extra = {'slow': 1 + elapsed // 3, 'fast': 4 + elapsed,
                     'intermittent': 3 + elapsed // 2 if elapsed % 4 < 2 else 0}[kind]
            return baseline + extra, f'zone-{index//8}-{cycle}-{kind}', kind
    return baseline, '', 'normal'


def evaluate(capture, output):
    manifest = json.loads((capture/'manifest.json').read_text())
    assert manifest['complete'] and manifest['load_experiment'] and manifest['seed'] in SEEDS
    assert manifest['windows'] == 180 and manifest['actors'] == 32
    events = pd.read_json(capture/'workload_events.jsonl', lines=True)
    assert len(events) == 5760 and not events.duplicated(KEYS).any()
    events['attack'] = events.episode_id.ne('').astype(int)
    events['variant'] = 'live_authorized_load_change'
    events['onset'] = events.groupby('episode_id').started.transform('min')
    online = pd.read_csv(capture/'online_predictions.csv')
    baseline = online[online.method.isin(BASELINES)].pivot(index=KEYS, columns='method', values='alarm').reset_index()
    records = []
    for row in map(json.loads, (capture/'distributed_predictions.jsonl').read_text().splitlines()):
        assert row['complete'] and len(row['decisions']) == 32
        for nf, flag in row['decisions'].items():
            record = dict(run_id=manifest['run_id'], window_id=row['window'], nf_id=nf,
                ACCUM=flag['accum_security'], ACCUM_OR_IF_CONTEXT=flag['accum_security'] or flag['if_context_alarm'],
                ACCUM_CONFIRM_IF_CONTEXT=flag.get('original_security', flag['security']), decision_time=row['decision_time'])
            if 'original_security' in flag:
                record['RECOVERY_HYBRID'] = flag['security']
            records.append(record)
    paired = events.merge(baseline,on=KEYS,validate='one_to_one').merge(pd.DataFrame(records),on=KEYS,validate='one_to_one')
    assert len(paired) == 5760
    methods = (*BASELINES, 'ACCUM', 'ACCUM_OR_IF_CONTEXT', 'ACCUM_CONFIRM_IF_CONTEXT')
    if 'RECOVERY_HYBRID' in paired:
        methods = (*methods, 'RECOVERY_HYBRID')
    metrics, episodes = summarize(paired, methods, include_delay=True)
    for episode in episodes:
        rows = paired[(paired.episode_id==episode['episode']) & paired[episode['method']].astype(bool)]
        times = online[online.method==episode['method']][KEYS+['decision_timestamp']] if episode['method'] in BASELINES else paired[KEYS+['decision_time']].rename(columns={'decision_time':'decision_timestamp'})
        hits = rows.merge(times,on=KEYS,validate='one_to_one')
        episode['first_alarm_delay_seconds'] = float(hits.decision_timestamp.min()-hits.onset.iloc[0]) if len(hits) else None
    by_method = {row['method']:row for row in metrics}
    base, hybrid = by_method['ACCUM'], by_method['ACCUM_CONFIRM_IF_CONTEXT']
    delta = hybrid['fpr']-base['fpr']
    report = dict(seed=manifest['seed'],run_id=manifest['run_id'],metrics=metrics,
        fpr_increment_limit=FPR_INCREMENT_LIMIT,fpr_increment=delta,fpr_limit_met=delta<=FPR_INCREMENT_LIMIT,
        features_sha256=sha256(capture/'features.csv'),workload_events_sha256=sha256(capture/'workload_events.jsonl'),
        scope='Prospective authorized NSSF discovery workload changes on an isolated Open5GS lab. No feature editing. Three baseline profiles. All 12 scheduled episodes retained.')
    output.mkdir(parents=True,exist_ok=True)
    (output/'load_report.json').write_text(json.dumps(report,indent=2)+'\n')
    pd.DataFrame(episodes).to_csv(output/'load_episodes.csv',index=False)
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('capture',type=Path)
    parser.add_argument('output',type=Path)
    evaluate(**vars(parser.parse_args()))
