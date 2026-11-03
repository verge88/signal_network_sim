"""Consolidate independent test evidence without selecting a winning model."""
import argparse
from pathlib import Path

import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    frames = []
    for path in sorted(root.rglob('independent_metrics.csv')):
        df = pd.read_csv(path)
        df['suite'] = 'neural' if 'neural' in str(path.parent) else 'classical'
        frames.append(df)
    text = ['# Independent Diameter evaluation', '',
            'Source: training run 40538956475, commit 32755d6fb47e9df590934b20e5413afa583ab479.', '',
            'Historical fitted weights were not serialized. Results below concern reconstructed replicas; archived prediction agreement is audited per model.', '',
            'Test seeds: 1001, 1002, 1003. Six new synthetic datasets per frozen training replica: original schedules plus shifted attack/benign-event times and remapped targets within node type and role.', '',
            'All preprocessing and thresholds are frozen before test generation. Meta-Ensemble uses its prespecified 0.5 threshold; other models use validation-negative 1% FPR calibration.', '',
            'The six datasets are shared across training replicas. They are not 60 independent datasets; no confidence intervals treating repeated scores or correlated rows as independent are reported.', '',
            'This assesses transfer within the calibrated simulator family. It does not establish operational accuracy on external or real traffic.', '']
    complete = False
    if frames:
        df = pd.concat(frames, ignore_index=True)
        df.to_csv(out / 'all_independent_metrics.csv', index=False)
        summary = df.groupby(['suite', 'model', 'scenario'])[
            ['roc_auc', 'average_precision', 'recall', 'fpr', 'debounce2_recall', 'debounce2_fpr']
        ].agg(['mean', 'min', 'max', 'count'])
        summary.to_csv(out / 'aggregate.csv')
        means = df.groupby(['suite', 'model', 'scenario'])[['roc_auc', 'recall', 'fpr', 'debounce2_recall', 'debounce2_fpr']].mean()
        text += ['```', means.to_string(), '```', '']
        coverage = df.groupby(['suite', 'train_seed']).agg(test_seeds=('test_seed', 'nunique'), scenarios=('scenario', 'nunique'))
        complete = (len(coverage) == 11 and (coverage.test_seeds == 3).all() and (coverage.scenarios == 2).all())
        # Every model in every suite must cover all 6 preregistered datasets.
        complete = complete and (df.groupby(['suite','train_seed','model']).size() == 6).all()
        text += [f'Complete expected coverage: {bool(complete)}', '', '## Historical prediction replay', '']
        audits = []
        for path in sorted(root.rglob('replay_audit.csv')):
            audit = pd.read_csv(path)
            audit['artifact'] = path.parent.name
            audits.append(audit)
        if audits:
            audit = pd.concat(audits, ignore_index=True)
            audit.to_csv(out / 'replay_audit.csv', index=False)
            text += ['```', audit.to_string(index=False), '```', '']
    else:
        text += ['No independent metrics produced. Inspect job failures.', '']
    (out / 'report.md').write_text('\n'.join(text), encoding='utf-8')
    print('\n'.join(text))
    if not complete:
        raise SystemExit('Independent evaluation is incomplete')


if __name__ == '__main__':
    main()
