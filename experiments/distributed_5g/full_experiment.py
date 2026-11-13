"""Prospective full-method protocol: fresh IF fitting and paired audited evaluation."""
import argparse
import collections
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from experiments.live_5g.evaluate import KEYS, DETECTOR_CHOICES, fit, fit_supervised, fit_temporal_graph, fit_temporal_graph_v2, fit_temporal_graph_v3, fit_temporal_graph_v4, load_frozen, score_frame, sha256, metrics
from experiments.distributed_5g.paired import summarize
from experiments.robustness_5g.design import attack_schedule, s5_members
from experiments.resilient_5g.audit import mac


def read(path):
    return json.loads(path.read_text())


def lines(path):
    with path.open() as stream:
        for line in stream:
            yield json.loads(line)


def prepare(root, previous, seed, detector='iforest', development=(), reference_calibration=(),
            v3_cluster_order=2, v3_decay=.75, v3_alpha=.0002, v3_accumulator_multiplier=1.50,
            v4_cluster_order=2, v4_decay=.75, v4_alpha=.0001, v4_accumulator_multiplier=1.50):
    if detector == 'rf_supervised':
        manifest = fit_supervised(root/'train', root/'cal', development, root/'frozen', alpha=.001, seed=seed)
    elif detector == 'temporal_graph':
        manifest = fit_temporal_graph(root/'train', root/'cal', root/'frozen', seed=seed,
                                      point_alpha=.001, graph_alpha=.005)
    elif detector == 'temporal_graph_v2':
        manifest = fit_temporal_graph_v2(root/'train', root/'cal', development, root/'frozen',
                                         seed=seed, point_alpha=.001, graph_alpha=.005)
    elif detector == 'temporal_graph_v3':
        manifest = fit_temporal_graph_v3(
            root/'train', root/'cal', development, reference_calibration, root/'frozen',
            seed=seed, point_alpha=.001, graph_alpha=v3_alpha,
            cluster_order=v3_cluster_order, decay=v3_decay,
            accumulator_multiplier=v3_accumulator_multiplier)
    elif detector == 'temporal_graph_v4':
        manifest = fit_temporal_graph_v4(
            root/'train', root/'cal', development, reference_calibration, root/'frozen',
            seed=seed, point_alpha=.001, graph_alpha=v4_alpha,
            cluster_order=v4_cluster_order, decay=v4_decay,
            accumulator_multiplier=v4_accumulator_multiplier)
    else:
        manifest = fit(root/'train', root/'cal', root/'frozen', alpha=.001, seed=seed, detector=detector)
    assert manifest['threshold_sufficient']
    origin = {}
    for name in ('extensions.json', 'refinement.json', 'accumulation.json'):
        shutil.copyfile(previous/name, root/'frozen'/name)
        origin[name] = sha256(previous/name)
    (root/'frozen_origin.json').write_text(json.dumps(dict(
        fresh_data_mining=True, inherited_fixed_components=origin,
        detector=detector,
        development_runs=manifest.get('development_runs', []),
        calibration_reference_runs=manifest.get('calibration_reference_runs', []),
        explanation=(
            'Historical labeled development is used only for supervised detector fitting when declared; '
            'all operational thresholds are calibrated exclusively on normal calibration runs. '
            'Control deadlines, fusion parameters and accumulator parameters are frozen before held-out testing; '
            'zone models are separately hash-pinned.')), indent=2))


def audit(root):
    capture = root/'test'
    output = root/'summary'
    output.mkdir(exist_ok=True)
    manifest = read(capture/'manifest.json')
    deployment = read(capture/'smoke_report.json')
    confirmation_fraction = float(deployment.get('confirmation_fraction', .5))
    ml_method = deployment.get('ml_method', 'IF_CONTEXT')
    ml_confirmation_mode = deployment.get('ml_confirmation_mode', 'recovery')
    ml_persistence_window = int(deployment.get('ml_persistence_window', 5))
    ml_persistence_hits = int(deployment.get('ml_persistence_hits', 3))
    accumulator_multiplier = float(deployment.get('accumulator_multiplier', 1.0))
    assert 0 < confirmation_fraction <= 1
    assert accumulator_multiplier > 0
    assert ml_confirmation_mode in ('recovery', 'persistence', 'direct')
    assert 1 <= ml_persistence_hits <= ml_persistence_window
    assert manifest['complete'] and manifest['windows'] == 3600 and manifest['actors'] == 32
    assert manifest['schedule'] == [list(x) for x in attack_schedule(3600, manifest['seed'])]
    bundle = load_frozen(root/'frozen')
    features = pd.read_csv(capture/'features.csv', float_precision='round_trip')
    labels = pd.read_csv(capture/'labels.csv').fillna({'episode_id': ''})
    assert len(features) == 115200 and not features.duplicated(KEYS).any()
    assert features.groupby('window_id').size().eq(32).all()
    assert set(features.window_id) == set(range(3600))
    assert features.run_id.nunique() == 1 and features.run_id.iloc[0] == manifest['run_id']
    assert manifest['run_id'] not in (bundle['manifest']['train_runs'] + bundle['manifest']['calibration_runs'] +
                                      bundle['manifest'].get('development_runs', []))
    assert features.timestamp.min() > bundle['manifest']['created_at']
    for phase in ('train', 'cal'):
        fm = read(root/phase/'manifest.json')
        lab = pd.read_csv(root/phase/'labels.csv')
        assert fm['complete'] and len(lab) == 19200 and not lab.attack.any()
        assert fm['completed_at'] < bundle['manifest']['created_at']
    # Independent transaction-count/error/latency checks; no synthetic feature injection.
    observations = collections.defaultdict(lambda: [0, 0, 0.])
    for event in lines(capture/'events.jsonl'):
        if event['source'] == 'actor_transaction':
            item = observations[event['window_id'], event['slot']]
            item[0] += 1
            item[1] += int(event['status'] >= 400 or event['status'] == 0)
            item[2] += event['latency_ms']
    for row in features.itertuples():
        count, errors, latency = observations[row.window_id, row.nf_id]
        assert count == row.request_count
        assert abs(errors/max(1,count)-row.error_rate) < 1e-12
        assert abs(latency/max(1,count)-row.latency_ms) < 1e-9
    recomputed = score_frame(bundle, features)
    online = pd.read_csv(capture/'online_predictions.csv', float_precision='round_trip')
    checked = recomputed.merge(online, on=KEYS+['method'], validate='one_to_one', suffixes=('', '_live'))
    assert len(checked) == len(recomputed) == len(online)
    assert np.array_equal(checked.alarm, checked.alarm_live)
    assert np.allclose(checked.score, checked.score_live, rtol=0, atol=1e-12)
    # The schedule is used only here, after detector execution, to verify labels.
    expected = {}
    for entry in attack_schedule(3600, manifest['seed']):
        start, end, scenario, slot, step, variant = entry
        members = s5_members(entry) if scenario == 'S5' else (slot,)
        for window in range(start, end):
            for member in members:
                expected[window, f'slot-{member}'] = (scenario, variant)
    for row in labels.itertuples():
        item = expected.get((row.window_id, row.nf_id))
        assert bool(row.attack) == bool(item)
        if item:
            assert (row.scenario, row.variant) == item
    zones = read(root/'zone_models.json')
    grants = {r['window_id']: r['envelope'] for r in lines(capture/'load_grants.jsonl')}
    operator = bytes.fromhex((capture/'lab_operator_key.hex').read_text())
    accum, recovery, quiet = {}, {}, {}
    ml_history = {}
    records, timing, resources = [], [], []
    by_window = dict(tuple(features.groupby('window_id')))
    selected_ml = online[online.method == ml_method]
    assert len(selected_ml) == len(features)
    context = selected_ml.set_index(KEYS).alarm.to_dict()
    for result in lines(capture/'distributed_predictions.jsonl'):
        window = result['window']
        assert result['complete'] and len(result['decisions']) == 32
        group = by_window[window].set_index('nf_id')
        timing.append(result['transport_seconds'])
        if 'process_resources' in result:
            resources.append(result['process_resources'])
        for zone, model in zones.items():
            for identity in model['members']:
                row = group.loc[identity]
                decision = result['decisions'][identity]
                peers = group.loc[[nf for nf in model['members'] if nf != identity], 'request_count'].to_numpy()
                center = np.median(peers)
                residual = float((row.request_count-center)/max(1., np.median(abs(peers-center))))
                grant = grants[window]
                allowed = False
                if mac(grant, operator):
                    g = grant['report']
                    allowed = (g['run_id'] == row.run_id and g['nf_id'] == identity and
                               g['from'] <= row.timestamp <= g['until'] and row.request_count <= g['max_requests'])
                assert allowed == decision['allowed']
                increment = min(4., max(-1., residual)) - .5
                accum[identity] = 0. if allowed else max(0., .99*accum.get(identity,0.)+increment)
                recovery[identity] = 0. if allowed else max(0., .99*recovery.get(identity,0.)+increment)
                normal = residual <= 0 and not any(decision[k] for k in ('local_security','availability','unknown'))
                quiet[identity] = min(3, quiet.get(identity,0)+1) if normal else 0
                if quiet[identity] == 3:
                    recovery[identity] = 0.
                assert abs(accum[identity]-decision['accumulated']) < 1e-12
                assert abs(recovery[identity]-decision['recovery_accumulated']) < 1e-12
                assert quiet[identity] == decision['recovery_quiet_windows']
                ml = bool(context[row.run_id, window, identity])
                assert decision['if_context_alarm'] == ml
                without_ml = decision['local_security'] or recovery[identity] > model['threshold']
                budgeted_without_ml = (decision['local_security'] or
                                       recovery[identity] > accumulator_multiplier*model['threshold'])
                if ml_confirmation_mode == 'persistence':
                    history = list(ml_history.get(identity, []))
                    history.append(ml)
                    history = history[-ml_persistence_window:]
                    ml_history[identity] = history
                    confirmed = sum(history) >= ml_persistence_hits
                elif ml_confirmation_mode == 'direct':
                    confirmed = ml
                else:
                    confirmed = ml and recovery[identity] > confirmation_fraction*model['threshold']
                assert bool(decision['if_context_confirmed']) == bool(confirmed)
                full = budgeted_without_ml or confirmed
                assert full == decision['security']
                if 'budgeted_without_ml' in decision:
                    assert bool(decision['budgeted_without_ml']) == bool(budgeted_without_ml)
                records.append(dict(run_id=row.run_id, window_id=window, nf_id=identity,
                    FULL_HYBRID=bool(full), WITHOUT_MINING=bool(without_ml),
                    BUDGETED_WITHOUT_MINING=bool(budgeted_without_ml),
                    ORIGINAL_HYBRID=decision['original_security'],
                    availability=decision['availability'], unknown=decision['unknown'],
                    decision_time=result['decision_time']))
    distributed = pd.DataFrame(records)
    assert len(distributed) == len(features) and not distributed.duplicated(KEYS).any()
    baseline_methods = ['rules','IF_SBI','IF_CONTEXT']
    if ml_method not in baseline_methods:
        baseline_methods.append(ml_method)
    baselines = online[online.method.isin(baseline_methods)].pivot(index=KEYS, columns='method', values='alarm').reset_index()
    paired = labels.merge(baselines,on=KEYS,validate='one_to_one').merge(distributed,on=KEYS,validate='one_to_one')
    methods = tuple(baseline_methods + ['WITHOUT_MINING','BUDGETED_WITHOUT_MINING','ORIGINAL_HYBRID','FULL_HYBRID'])
    totals, episodes = summarize(paired, methods, include_delay=True)
    strata = []
    for metric in totals:
        name = metric['method']
        metric['precision'] = metric['tp']/(metric['tp']+metric['fp']) if metric['tp']+metric['fp'] else None
        pred = online[online.method == name][KEYS+['score','decision_timestamp']] if name in baseline_methods else distributed[KEYS+['decision_time']].rename(columns={'decision_time':'decision_timestamp'})
        part = paired.merge(pred,on=KEYS,validate='one_to_one')
        metric['average_precision'] = float(average_precision_score(part.attack,part.score)) if 'score' in part else None
        metric['ap_note'] = 'Continuous anomaly margin' if 'score' in part else 'No continuous score for stateful decision fusion; AP not fabricated from binary alarms.'
        part = part.sort_values(['nf_id','window_id'])
        part['rising'] = part[name].astype(bool) & ~part.groupby('nf_id')[name].shift(fill_value=False).astype(bool)
        for episode in (e for e in episodes if e['method'] == name):
            hits = part[(part.episode_id == episode['episode']) & part.rising]
            episode['new_alarm_delay_seconds'] = float(hits.decision_timestamp.min()-hits.onset.min()) if len(hits) else None
        for (scenario,variant), group in part[part.attack == 1].groupby(['scenario','variant']):
            strata.append(dict(method=name,scenario=scenario,variant=variant,windows=len(group),tp=int(group[name].sum())))
    report = dict(seed=manifest['seed'],run_id=manifest['run_id'],detector=bundle['manifest'].get('detector','iforest'),source_commit=(root/'source_commit.txt').read_text().strip(),
        audit_pass=True,rows=len(paired),metrics=totals,strata=strata,
        model_sha256=sha256(root/'frozen/model.joblib'),zone_models_sha256=sha256(root/'zone_models.json'),
        data_mining_increment=dict(
            additional_tp_over_budgeted=int((paired.attack.eq(1)&paired.FULL_HYBRID&~paired.BUDGETED_WITHOUT_MINING).sum()),
            additional_fp_over_budgeted=int((paired.attack.eq(0)&paired.FULL_HYBRID&~paired.BUDGETED_WITHOUT_MINING).sum())),
        full_vs_original_without=dict(
            tp_delta=int((paired.attack.eq(1)&paired.FULL_HYBRID).sum()-(paired.attack.eq(1)&paired.WITHOUT_MINING).sum()),
            fp_delta=int((paired.attack.eq(0)&paired.FULL_HYBRID).sum()-(paired.attack.eq(0)&paired.WITHOUT_MINING).sum())),
        availability_windows=int(paired.availability.sum()),unknown_windows=int(paired.unknown.sum()),
        confirmation_fraction=confirmation_fraction,
        accumulator_multiplier=accumulator_multiplier,
        ml_method=ml_method,
        ml_confirmation_mode=ml_confirmation_mode,
        ml_persistence_window=ml_persistence_window,
        ml_persistence_hits=ml_persistence_hits,
        fpr_target=.001,fpr_target_met=next(m['fpr'] for m in totals if m['method']=='FULL_HYBRID')<=.001,
        overhead=dict(hierarchy_transport_median_seconds=float(np.median(timing)),
            hierarchy_transport_p95_seconds=float(np.quantile(timing,.95)),
            hierarchy_cpu_seconds=sum(r['cpu_seconds'] for r in resources[-1]),
            hierarchy_sampled_peak_rss_bytes=max(sum(r['rss_bytes'] for r in sample) for sample in resources),
            collector_cpu_seconds=manifest['collector_cpu_seconds'],elapsed_seconds=manifest['elapsed_seconds']),
        limitations=['32 logical NF: native NSSF and registered research clients; no UE/RAN/PDU or hardware attestation.',
            'Four zones and coordinator use distinct network namespaces on one physical runner per repetition.',
            'Unknown/control loss is not counted as a security detection.',
            'Fresh detector fitting; fixed earlier control and zone calibration retained. No test tuning.',
            'Hierarchy arithmetic independently recomputed; local control protocol covered by regression tests, not an independent implementation in this audit.',
            'Resource snapshots exclude core/receiver/observer processes; no causal monitor-off latency comparison.',
            'Window metrics are dependent; three runs, not 345600 independent trials.'])
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    pd.DataFrame(episodes).to_csv(output/'episodes.csv',index=False)
    shutil.copyfile(root/'frozen/manifest.json',output/'model_manifest.json')
    shutil.copyfile(capture/'manifest.json',output/'test_manifest.json')
    shutil.copyfile(capture/'smoke_report.json',output/'deployment_report.json')
    (output/'integrity.json').write_text(json.dumps({str(p.relative_to(root)):sha256(p) for p in
        [root/'frozen/model.joblib',root/'zone_models.json',capture/'features.csv',capture/'labels.csv',capture/'online_predictions.csv',capture/'distributed_predictions.jsonl']},indent=2))
    print(json.dumps(report),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['prepare','audit'])
    parser.add_argument('root',type=Path)
    parser.add_argument('--previous',type=Path)
    parser.add_argument('--seed',type=int)
    parser.add_argument('--detector',choices=DETECTOR_CHOICES,default='iforest')
    parser.add_argument('--development',type=Path,nargs='*',default=[])
    parser.add_argument('--reference-calibration',type=Path,nargs='*',default=[])
    parser.add_argument('--v3-cluster-order',type=int,default=2)
    parser.add_argument('--v3-decay',type=float,default=.75)
    parser.add_argument('--v3-alpha',type=float,default=.0002)
    parser.add_argument('--v3-accumulator-multiplier',type=float,default=1.50)
    parser.add_argument('--v4-cluster-order',type=int,default=2)
    parser.add_argument('--v4-decay',type=float,default=.75)
    parser.add_argument('--v4-alpha',type=float,default=.0001)
    parser.add_argument('--v4-accumulator-multiplier',type=float,default=1.50)
    args=parser.parse_args()
    prepare(args.root,args.previous,args.seed,args.detector,args.development,
            args.reference_calibration,args.v3_cluster_order,args.v3_decay,
            args.v3_alpha,args.v3_accumulator_multiplier,
            args.v4_cluster_order,args.v4_decay,
            args.v4_alpha,args.v4_accumulator_multiplier) if args.action=='prepare' else audit(args.root)
