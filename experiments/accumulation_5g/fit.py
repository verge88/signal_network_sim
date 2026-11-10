"""Use fully benign windows from the disclosed former test run for calibration."""
import collections
import hashlib
import json
import sys
import time
from pathlib import Path
import pandas as pd
from experiments.live_5g.evaluate import conformal_threshold
from experiments.resilient_5g.detector import permit_valid
from experiments.accumulation_5g.detector import advance, peer_residuals


def fit(source, target):
    target = Path(target)
    if target.exists():
        raise FileExistsError(target)
    features = pd.read_csv(source/'test/features.csv', float_precision='round_trip')
    labels = pd.read_csv(source/'test/labels.csv')
    attacked = set(labels.loc[labels.attack == 1, 'window_id'])
    normal = features[~features.window_id.isin(attacked)].copy()
    grants = {r['window_id']: r['envelope'] for r in map(json.loads, (source/'test/load_grants.jsonl').read_text().splitlines())}
    key = bytes.fromhex((source/'test/lab_operator_key.hex').read_text())
    scores = []
    accumulators = collections.defaultdict(float)
    previous_window = None
    for window, group in normal.groupby('window_id', sort=True):
        if previous_window is None or window != previous_window + 1:
            accumulators.clear()
        rows = group.to_dict('records')
        residuals = peer_residuals(rows)
        for row in rows:
            identity = row['nf_id']
            allowed = permit_valid(grants[window], key, row['run_id'], identity, row['timestamp'], row['request_count'])
            value = advance(accumulators[identity], residuals[identity], allowed)
            accumulators[identity] = value
            scores.append(value)
        previous_window = window
    normal['error_count'] = (normal.request_count * normal.error_rate).round().astype(int)
    alpha = .001 / 4
    model = dict(created_at=time.time(), version=1, alpha_component=alpha, decay=.99, drift=.5,
        residual_clip=[-1.,4.], threshold=conformal_threshold(scores,alpha),
        feature_thresholds={name:conformal_threshold(normal[name],alpha) for name in ['request_count','error_count','update_count']},
        calibration_run=features.run_id.iloc[0],calibration_windows=int(normal.window_id.nunique()),
        calibration_rows=len(normal), excluded_attack_windows=len(attacked),
        source_hashes={name:hashlib.sha256((source/'test'/name).read_bytes()).hexdigest()
                      for name in ['features.csv','labels.csv','load_grants.jsonl','lab_operator_key.hex']},
        disclosure='Former ablation test data now used for calibration. Entire attack windows excluded. Accumulators reset across excluded gaps. Correlated windows do not establish an operational FPR guarantee.')
    target.write_text(json.dumps(model,indent=2,allow_nan=False))
    print(json.dumps(model))

if __name__ == '__main__':
    fit(Path(sys.argv[1]), Path(sys.argv[2]))
