"""Frozen confirmation experiment on fresh SBI features and labelled offline alterations."""
import argparse
import json
from pathlib import Path

import pandas as pd

from experiments.distributed_5g.deploy import Deployment, configure
from experiments.distributed_5g.node import CONFIRMATION_FRACTION
from experiments.distributed_5g.paired import BASELINES, summarize
from experiments.live_5g.evaluate import KEYS, load_frozen, score_frame, sha256


SEED = 86241
WINDOWS = 180
STARTS = (24, 77, 130)
DURATION = 24
PATTERNS = ('slow_growth', 'fast_growth', 'intermittent_growth', 'error_burst')
METHODS = (*BASELINES, 'ACCUM', 'ACCUM_OR_IF_CONTEXT', 'ACCUM_CONFIRM_IF_CONTEXT')


def alterations(features, labels, topology):
    data = features.set_index(KEYS).copy()
    truth = labels.set_index(KEYS).copy()
    run_id = features.run_id.iloc[0]
    plan = []
    for cycle, start in enumerate(STARTS):
        for zone_index, (zone, members) in enumerate(sorted(topology.items())):
            identity = members[1 + cycle]
            pattern = PATTERNS[(cycle + zone_index) % len(PATTERNS)]
            episode = f'{zone}-{cycle}-{pattern}'
            for elapsed in range(DURATION):
                key = (run_id, start + elapsed, identity)
                if pattern == 'slow_growth':
                    requests, errors = 1 + elapsed // 3, 0
                elif pattern == 'fast_growth':
                    requests, errors = 4 + elapsed, 0
                elif pattern == 'intermittent_growth':
                    requests, errors = (3 + elapsed // 2 if elapsed % 4 < 2 else 0), 0
                else:
                    requests, errors = 3, 3
                old_count = data.at[key, 'request_count']
                old_errors = round(old_count * data.at[key, 'error_rate'])
                timestamp = data.at[key, 'timestamp']
                data.at[key, 'request_count'] = old_count + requests
                data.at[key, 'error_rate'] = (old_errors + errors) / max(1, old_count + requests)
                for width in (10, 60):
                    count_column, rate_column = f'request_count_{width}s', f'error_rate_{width}s'
                    for later in range(start + elapsed, WINDOWS):
                        later_key = (run_id, later, identity)
                        if data.at[later_key, 'timestamp'] - timestamp >= width:
                            break
                        count = data.at[later_key, count_column]
                        prior_errors = round(count * data.at[later_key, rate_column])
                        data.at[later_key, count_column] = count + requests
                        data.at[later_key, rate_column] = (prior_errors + errors) / max(1, count + requests)
                truth.loc[key, ['attack', 'scenario', 'episode_id', 'variant']] = (1, pattern, episode, 'offline')
            plan.append(dict(episode=episode, pattern=pattern, nf_id=identity, zone=zone,
                             first_window=start, last_window=start + DURATION - 1))
    return data.reset_index(), truth.reset_index(), plan


def lines(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream]


def decisions(features, bundle, models, capture, work):
    topology = {zone: model['members'] for zone, model in models.items()}
    computed = score_frame(bundle, features)
    baselines = computed[computed.method.isin(BASELINES)].pivot(index=KEYS, columns='method', values='alarm')
    controls = {row['window_id']: row for row in lines(capture/'control_events.jsonl')}
    grants = {row['window_id']: row['envelope'] for row in lines(capture/'load_grants.jsonl')}
    reference_time = {(row['window_id'], row['nf_id']): row['evaluation_time']
                      for row in lines(capture/'refined_predictions.jsonl')}
    run_id = features.run_id.iloc[0]
    configure(work, topology, models, run_id, (capture/'lab_operator_key.hex').read_text(),
              (capture/'lab_control_key.hex').read_text(), bundle['extensions'], True,
              include_if_context=True, confirm_if_context=True)
    records = []
    with Deployment(work) as deployment:
        for window, group in features.groupby('window_id', sort=True):
            observations = {}
            for row in group.to_dict('records'):
                identity = row['nf_id']
                observations[identity] = dict(row=row, grant=grants[window],
                    control=controls[window] if identity == 'slot-4' else None,
                    evaluation_time=reference_time[window, identity],
                    if_context_alarm=bool(baselines.at[(run_id, window, identity), 'IF_CONTEXT']))
            result = deployment.window(int(window), observations)
            assert result['complete'] and set(result['decisions']) == set(observations)
            for identity, flags in result['decisions'].items():
                context = observations[identity]['if_context_alarm']
                assert flags['if_context_alarm'] == context
                assert flags['security'] == (flags['accum_security'] or flags['if_context_confirmed'])
                records.append(dict(run_id=run_id, window_id=window, nf_id=identity, ACCUM=flags['accum_security'],
                    ACCUM_OR_IF_CONTEXT=flags['accum_security'] or context, ACCUM_CONFIRM_IF_CONTEXT=flags['security']))
    return baselines.reset_index().merge(pd.DataFrame(records), on=KEYS, validate='one_to_one')


def run(capture, frozen, zone_models, work, output):
    manifest = json.loads((capture/'manifest.json').read_text())
    assert manifest['complete'] and manifest['normal_smoke'] and not manifest['schedule']
    assert manifest['windows'] == WINDOWS and manifest['seed'] == SEED and manifest['actors'] == 32
    features = pd.read_csv(capture/'features.csv', float_precision='round_trip')
    labels = pd.read_csv(capture/'labels.csv').fillna({'episode_id': ''})
    assert labels.attack.eq(0).all() and len(labels) == len(features) == WINDOWS * 32
    bundle = load_frozen(frozen)
    bundle['extensions'] = json.loads((frozen/'extensions.json').read_text())
    provenance = bundle['manifest']
    assert set(features.run_id) == {manifest['run_id']}
    assert manifest['run_id'] not in provenance['train_runs'] + provenance['calibration_runs']
    assert features.timestamp.min() > provenance['created_at']
    models = json.loads(zone_models.read_text())
    assert all(model['calibration_run'] != manifest['run_id'] for model in models.values())
    altered, altered_labels, plan = alterations(features, labels, {zone: m['members'] for zone, m in models.items()})
    metrics, episodes = {}, []
    for case, frame, truth in (('normal', features, labels), ('altered', altered, altered_labels)):
        results = decisions(frame, bundle, models, capture, work/case)
        paired = truth.merge(results, on=KEYS, validate='one_to_one')
        assert len(results) == len(paired) == len(labels)
        metrics[case], case_episodes = summarize(paired, METHODS, include_delay=True)
        episodes.extend(case_episodes)
        print(f'Verified {case}: {len(paired)} NF windows', flush=True)
    report = dict(run_id=manifest['run_id'], seed=SEED, windows=WINDOWS, nf_windows=WINDOWS*32,
        confirmation_fraction=CONFIRMATION_FRACTION, model_sha256=provenance['model_sha256'],
        zone_models_sha256=sha256(zone_models), fresh_features_sha256=sha256(capture/'features.csv'),
        episode_plan=plan, metrics=metrics,
        scope='Fresh normal Open5GS SBI measurements plus offline alterations to request/error counts and their rolling summaries. Two paired HTTP replays. Other measurements and signed control evidence are held fixed.')
    output.mkdir(parents=True, exist_ok=True)
    (output/'confirmation_report.json').write_text(json.dumps(report, indent=2)+'\n')
    pd.DataFrame(episodes).to_csv(output/'confirmation_episodes.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('capture', 'frozen', 'zone_models', 'work', 'output'):
        parser.add_argument(name, type=Path)
    run(**vars(parser.parse_args()))
