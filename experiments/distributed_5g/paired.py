"""Compare the frozen data-mining branches and zonal fusion on the same archive."""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np
import pandas as pd

from experiments.distributed_5g.replay import frame, read_json
from experiments.live_5g.evaluate import KEYS, load_frozen, score_frame, sha256


BASELINES = ('rules', 'IF_SBI', 'IF_CONTEXT')


def run(evidence, deployment, output):
    output.mkdir(parents=True, exist_ok=True)
    with ZipFile(evidence) as archive, TemporaryDirectory(dir=output.resolve()) as directory:
        features = frame(archive, 'test/features.csv')
        labels = frame(archive, 'test/labels.csv').fillna({'episode_id': ''})
        manifest = read_json(archive, 'test/manifest.json')
        for name in ('model.joblib', 'manifest.json'):
            (Path(directory) / name).write_bytes(archive.read('frozen/' + name))
        bundle = load_frozen(directory)
        training_runs = bundle['manifest']['train_runs'] + bundle['manifest']['calibration_runs']
        assert set(features.run_id) == {manifest['run_id']} and manifest['run_id'] not in training_runs
        assert features.timestamp.min() > bundle['manifest']['created_at']
        recomputed = score_frame(bundle, features)
        recomputed = recomputed.loc[recomputed.method.isin(BASELINES), KEYS + ['method', 'alarm', 'score']]
        online = frame(archive, 'test/online_predictions.csv')
        online = online.loc[online.method.isin(BASELINES), KEYS + ['method', 'alarm', 'score']]
        checked = recomputed.merge(online, on=KEYS + ['method'], validate='one_to_one', suffixes=('', '_live'))
        assert len(checked) == len(online) == len(recomputed) == len(features) * len(BASELINES)
        assert np.array_equal(checked.alarm, checked.alarm_live)
        assert np.allclose(checked.score, checked.score_live, rtol=0, atol=1e-12)

    with ZipFile(deployment) as archive:
        replay = read_json(archive, 'report.json')
        assert replay['exact_reference_match'] and replay['verified_nf_windows'] == len(features)
        assert sha256(evidence) in replay['source_archives'].values()
        records = []
        with archive.open('decisions.jsonl') as stream:
            for line in stream:
                window = json.loads(line)
                assert window['complete'] and len(window['decisions']) == manifest['actors']
                records.extend({'run_id': manifest['run_id'], 'window_id': window['window'], 'nf_id': identity,
                    'ACCUM': flags['security'], 'availability': flags['availability'], 'unknown': flags['unknown']}
                    for identity, flags in window['decisions'].items())
    baselines = checked.pivot(index=KEYS, columns='method', values='alarm').reset_index()
    paired = labels.merge(baselines, on=KEYS, validate='one_to_one').merge(
        pd.DataFrame(records), on=KEYS, validate='one_to_one')
    assert len(labels) == len(paired) == len(features) == len(records)
    paired['ACCUM_OR_IF_CONTEXT'] = paired.ACCUM | paired.IF_CONTEXT.astype(bool)
    paired = paired.sort_values(['nf_id', 'window_id'])
    truth = paired.attack.eq(1)
    assert paired.loc[truth, 'episode_id'].ne('').all()
    totals, episode_rows = [], []
    for method in (*BASELINES, 'ACCUM', 'ACCUM_OR_IF_CONTEXT'):
        alarm = paired[method].astype(bool)
        prev = alarm.groupby(paired.nf_id).shift(fill_value=False)
        adjacent = paired.groupby('nf_id').window_id.diff().eq(1)
        rising = alarm & ~(prev & adjacent)
        episodes = paired.loc[truth].assign(alarm=alarm[truth], rising=rising[truth])
        for episode, rows in episodes.groupby('episode_id'):
            episode_rows.append(dict(method=method, episode=episode, scenario=rows.scenario.iloc[0],
                variant=rows.variant.iloc[0], alarm_present=bool(rows.alarm.any()),
                new_alarm=bool(rows.rising.any())))
        method_episodes = [row for row in episode_rows if row['method'] == method]
        tp, fp = int((truth & alarm).sum()), int((~truth & alarm).sum())
        totals.append(dict(method=method, tp=tp, fp=fp, fn=int(truth.sum())-tp, tn=int((~truth).sum())-fp,
            recall=tp/int(truth.sum()), fpr=fp/int((~truth).sum()), episodes=len(method_episodes),
            episodes_with_alarm=sum(row['alarm_present'] for row in method_episodes),
            episodes_with_new_alarm=sum(row['new_alarm'] for row in method_episodes)))
    report = dict(run_id=manifest['run_id'], rows_per_method=len(paired), model_sha256=bundle['manifest']['model_sha256'],
        source_sha256=sha256(evidence), zonal_replay_sha256=sha256(deployment), frozen_score_replay_match=True,
        metrics=totals, additional_tp=int((truth & ~paired.ACCUM & paired.IF_CONTEXT.astype(bool)).sum()),
        additional_fp=int((~truth & ~paired.ACCUM & paired.IF_CONTEXT.astype(bool)).sum()),
        availability_rows=int(paired.availability.sum()), unknown_rows=int(paired.unknown.sum()),
        scope='Retrospective pairing on a previously analyzed acquisition; candidate OR fusion had no threshold tuning here.',
        timing='Episode hit means an alarm was present; a new alarm requires a rising edge within the episode. No comparable live hybrid delay was recorded.')
    (output/'paired_report.json').write_text(json.dumps(report, indent=2) + '\n')
    pd.DataFrame(episode_rows).to_csv(output/'paired_episodes.csv', index=False)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('evidence', type=Path)
    parser.add_argument('deployment', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    run(args.evidence, args.deployment, args.output)
