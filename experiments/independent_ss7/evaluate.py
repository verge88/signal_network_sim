"""Independent SS7 evaluation of replicas of the completed historical RF/stack job."""
from __future__ import annotations

import argparse
import ast
from dataclasses import asdict
import hashlib
import importlib
import json
from pathlib import Path
import pickle
import sys


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def export_functions(module):
    """Observe fitted objects at return, without changing historical fit math."""
    captures = []
    wanted = {'prepare_matrices', 'm_random_forest', 'm_isolation_forest',
              'm_pca_recon', 'm_hist_gb', 'm_meta_ensemble'}

    def record(name, values):
        captures.append((name, {k: values[k] for k in ('f', 'p', 'imp', 'sc', 'meta') if k in values}))

    tree = ast.parse(Path(module.__file__).read_text(encoding='utf-8'))
    module.__dict__['_export_record'] = record
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted:
            final = ast.parse(f"_export_record('{node.name}', locals())").body
            node.body = [ast.Try(body=node.body, handlers=[], orelse=[], finalbody=final)]
            code = ast.Module(body=[node], type_ignores=[])
            exec(compile(ast.fix_missing_locations(code), module.__file__, 'exec'), module.__dict__)
    return captures


def infer_one(name, model, X):
    import numpy as np
    if name == 'm_isolation_forest':
        return -model['f'].score_samples(X)
    if name == 'm_pca_recon':
        p = model['p']
        return np.mean((X - p.inverse_transform(p.transform(X))) ** 2, axis=1)
    return model['f'].predict_proba(X)[:, 1]


def infer(bundle, frame):
    import numpy as np
    pre = bundle['preprocessing']
    X = pre['sc'].transform(pre['imp'].transform(frame[bundle['columns']]))
    scores = {'random_forest': bundle['random_forest']['f'].predict_proba(X)[:, 1]}
    if bundle['meta_fallback']:
        scores['meta_ensemble'] = bundle['meta_bases'][0][1]['f'].predict_proba(X)[:, 1]
    else:
        Z = np.column_stack([np.nan_to_num(infer_one(name, model, X)) for name, model in bundle['meta_bases'][:4]])
        meta = bundle['meta']
        scores['meta_ensemble'] = meta['meta'].predict_proba(meta['sc'].transform(Z))[:, 1]
    return scores


def main():
    import numpy as np
    import pandas as pd
    from sklearn.metrics import roc_auc_score, average_precision_score

    parser = argparse.ArgumentParser()
    parser.add_argument('--historical', required=True)
    parser.add_argument('--out', default='evidence')
    args = parser.parse_args()
    sys.path.insert(0, str(Path('source_ss7/ss7').resolve()))
    mod = importlib.import_module('ss7_training_v7')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = mod.TrainingConfig(seed=42, days=6, transport_overhead_bytes=80.0,
                             use_torch=False, use_cache=False)
    print('Rebuilding historical six-day seed-42 training dataset', flush=True)
    frame = mod.load_or_build(cfg)
    columns = mod.resolve_feature_columns(frame, mod.FEATURE_SETS[cfg.feature_set])
    mod.assert_no_leakage(columns)
    sc = cfg.sim_config()
    tr, va, te = mod.blocked_split(frame, sc, mod.RngBook(sc.seed)['split'])
    captures = export_functions(mod)
    M = mod.prepare_matrices(tr, va, te, columns)
    pre = {key: value for key, value in captures.pop()[1].items() if key in ('imp', 'sc')}
    historical_scores, historical_rows, models = {}, [], {}
    for name, function in [('random_forest', mod.m_random_forest), ('meta_ensemble', mod.m_meta_ensemble)]:
        captures.clear()
        print(f'Replaying {name}', flush=True)
        s_va, s_te = function(M, cfg, cfg.seed)
        historical_scores[name] = s_te
        historical_rows.append(mod.score_row(name, name, M, s_va, s_te, cfg))
        models[name] = list(captures)
    meta = {key: value for key, value in models['meta_ensemble'][-1][1].items() if key in ('meta', 'sc')}
    bundle = dict(preprocessing=pre, columns=columns, random_forest=models['random_forest'][-1][1],
                  meta=meta, meta_bases=models['meta_ensemble'][:-1], meta_fallback='meta' not in meta,
                  thresholds={r['model']: r['threshold'] for r in historical_rows})
    # Verify that exported estimators reproduce the historical function outputs.
    replay_scores = infer(bundle, te)
    for name in replay_scores:
        if not np.allclose(replay_scores[name], historical_scores[name], atol=1e-12):
            raise ValueError(f'exported {name} does not reproduce historical inference')
    pd.DataFrame(historical_rows).to_csv(out / 'replayed_historical_metrics.csv', index=False)
    historic_paths = list(Path(args.historical).rglob('models.csv'))
    if len(historic_paths) != 1:
        raise ValueError(f'expected historical completed stack artifact: {historic_paths}')
    old = pd.read_csv(historic_paths[0]).set_index('model')
    audit = []
    for row in historical_rows:
        name = row['model']
        for metric in ('threshold', 'auc', 'ap', 'episode_recall', 'window_fpr'):
            before, after = float(old.loc[name, metric]), float(row[metric])
            audit.append(dict(model=name, metric=metric, historical=before, replayed=after,
                              matches=bool(np.isclose(before, after, atol=1e-9, rtol=1e-7, equal_nan=True))))
    pd.DataFrame(audit).to_csv(out / 'historical_metric_audit.csv', index=False)
    frozen_path = out / 'frozen_models.pkl'
    with open(frozen_path, 'wb') as stream:
        pickle.dump(bundle, stream)
    frozen_sha = sha(frozen_path)
    del frame, M, tr, va, te
    rows = []
    for test_seed in (1001, 1002, 1003):
        print(f'Generating independent two-day test seed {test_seed}', flush=True)
        test_cfg = mod.TrainingConfig(seed=test_seed, days=2, transport_overhead_bytes=80.0,
                                      use_torch=False, use_cache=False)
        test = mod.load_or_build(test_cfg)
        mod.assert_no_leakage(columns)
        labels = test['label'].to_numpy(int)
        for name, scores in infer(bundle, test).items():
            threshold = bundle['thresholds'][name]
            row = dict(model=name, train_seed=42, test_seed=test_seed, n=len(labels),
                       n_positive=int(labels.sum()), roc_auc=float(roc_auc_score(labels, scores)),
                       average_precision=float(average_precision_score(labels, scores)),
                       recall=float((scores[labels == 1] >= threshold).mean()),
                       fpr=float((scores[labels == 0] >= threshold).mean()), threshold=threshold)
            row.update(mod.incident_metrics(test, scores, threshold, cfg))
            row['historical_metric_match'] = all(a['matches'] for a in audit if a['model'] == name)
            rows.append(row)
        pd.DataFrame(rows).to_csv(out / 'independent_metrics.csv', index=False)
    if sha(frozen_path) != frozen_sha:
        raise ValueError('frozen artifact changed')
    provenance = dict(source_sha='b27957dd75819b8bd9ea8cab2a9617ae49c26435', source_run=40283788549,
                      completed_job='RF + meta ensemble verification, seed 42',
                      historical_weights_available=False, historical_predictions_available=False,
                      audit='archived metrics compared; this cannot prove historical weight identity',
                      model_kind='reconstructed replicas', frozen_sha256=frozen_sha,
                      historical_training_config=asdict(cfg), test_seeds=[1001,1002,1003],
                      test_days=2, test_labels_used_for_fit=False, test_labels_used_for_threshold=False,
                      meta_ensemble_fell_back_to_rf=bundle['meta_fallback'], external_real_traffic=False)
    (out / 'provenance.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    report = ['# Independent SS7 synthetic evaluation', '',
              'Reconstructed replicas of the completed historical RF/stack seed-42 job.',
              'Historical weights/predictions were not saved; historical_metric_audit.csv compares archived metrics only.',
              'Training: six days, seed 42. Independent test: two days each, seeds 1001-1003. Thresholds frozen on historical validation data.',
              f'Meta-ensemble used historical RF fallback: {bundle["meta_fallback"]}.',
              'This is within-simulator transfer, not external operational traffic.', '', '```',
              pd.DataFrame(rows).to_string(index=False), '```']
    (out / 'report.md').write_text('\n'.join(report), encoding='utf-8')
    print('\n'.join(report), flush=True)


if __name__ == '__main__':
    main()
