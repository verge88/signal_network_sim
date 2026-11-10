"""Peer-normalized accumulation calibrated before prospective capture."""
import collections
import numpy as np
from experiments.calibrated_5g_v2.detector import Engine as PreviousEngine
from experiments.integrated_5g.control import combine


def advance(previous, residual, allowed):
    if allowed:
        return 0.0
    return max(0.0, 0.99 * previous + min(4.0, max(-1.0, residual)) - 0.5)


def peer_residuals(rows):
    counts = np.array([row['request_count'] for row in rows], dtype=float)
    result = {}
    for index, row in enumerate(rows):
        peers = np.delete(counts, index)
        center = float(np.median(peers))
        scale = max(1.0, float(np.median(np.abs(peers - center))))
        result[row['nf_id']] = (row['request_count'] - center) / scale
    return result


class Engine(PreviousEngine):
    def __init__(self, previous_model, control_key, operator_key, accumulation_model):
        super().__init__(previous_model, control_key, operator_key)
        self.accumulation_model = accumulation_model
        self.accumulators = collections.defaultdict(float)
        self.residuals = {}

    def set_window(self, rows):
        self.residuals = peer_residuals(rows)

    def step(self, row, old, control, grant, now):
        result = super().step(row, old, control, grant, now)
        identity = row['nf_id']
        residual = self.residuals[identity]
        accumulated = advance(self.accumulators[identity], residual, result['signals']['permit_valid'])
        self.accumulators[identity] = accumulated
        values = dict(row, error_count=round(row['request_count'] * row['error_rate']))
        tail = any(values[name] > threshold for name, threshold in self.accumulation_model['feature_thresholds'].items())
        base = tail or result['signals']['semantic']
        state = 'alarm' if base else 'normal'
        if control is not None:
            state = combine(base, control['delivered'], control['nonce'], control['expected'],
                            control['instance'], self.control_key, now)['K']
        security = state == 'alarm' or old['signals']['replay'] or accumulated > self.accumulation_model['threshold']
        availability = old['signals']['availability']
        result['outputs']['ACCUM'] = dict(security=bool(security), availability=availability,
            state='security' if security else 'availability' if availability else state)
        result['accumulation'] = dict(residual=residual, value=accumulated, tail=bool(tail))
        return result
