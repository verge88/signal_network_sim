"""Causal cross-scenario NF detector. One row per NF and one decision per window."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.ensemble import HistGradientBoostingClassifier


SLOTS = tuple(f"slot-{i}" for i in range(32))
MEASUREMENTS = ("request_count", "update_count", "error_rate")
HISTORY = 12
METHOD = "CROSS_SCENARIO"


def telemetry(values):
    data = np.asarray(values, dtype=np.float64)
    if (data.ndim != 3 or data.shape[1:] != (32, len(MEASUREMENTS)) or not len(data)
            or not np.isfinite(data).all() or (data < 0).any() or (data[:, :, 2] > 1).any()):
        raise ValueError("Expected finite nonnegative telemetry for 32 NFs in every window")
    return data


def flags(values, shape):
    result = np.asarray(values)
    if result.shape != shape or not np.isin(result, (0, 1)).all():
        raise ValueError("Missing or invalid per-NF decision flags")
    return result.astype(bool)


def feature_block(values, previous_counts=None):
    """Use only the current and preceding observations; keep the last 12 counts per NF."""
    values = telemetry(values)
    counts = values[:, :, 0].reshape(-1, 4, 8)
    previous = (np.repeat(counts[:1], HISTORY, axis=0) if previous_counts is None
                else previous_counts)
    history = np.concatenate([previous, counts])
    series = sliding_window_view(history, HISTORY + 1, axis=0)
    center = np.median(series, axis=2, keepdims=True)
    residual = series - center
    current = residual[..., -1]
    order = np.sort(current, axis=2)
    lower = order[:, :, 1:2]
    upper = order[:, :, 6:7]
    wide = np.broadcast_to(upper - lower, counts.shape)
    # Rank with midranks for ties. Features never depend on slot order within a zone.
    diff = current[:, :, :, None] - current[:, :, None, :]
    rank = (diff > 0).mean(axis=3) + .5 * (diff == 0).mean(axis=3)
    parts = [counts, values[:, :, 1].reshape(counts.shape),
             values[:, :, 2].reshape(counts.shape), current, rank, wide,
             current - lower, current - upper, current - residual[..., -2]]
    for width in (4, HISTORY):
        recent = residual[..., -width:]
        mean = recent.mean(axis=-1)
        parts += [mean, recent.std(axis=-1), recent.min(axis=-1),
                  recent.max(axis=-1), current - mean]
        # An NF may move with a small group while the rest of its zone is quiet.
        # Compare each history with every peer after removing the common zone load.
        changes = np.diff(residual[..., -width-1:], axis=-1)
        changes -= changes.mean(axis=-1, keepdims=True)
        norm = np.sqrt(np.sum(changes * changes, axis=-1))
        denominator = norm[..., :, None] * norm[..., None, :]
        product = changes @ changes.swapaxes(-2, -1)
        corr = np.divide(product, denominator, out=np.zeros_like(product), where=denominator > 1e-10)
        diag = np.arange(8)
        corr[:, :, diag, diag] = -2.
        corr.sort(axis=-1)
        parts.extend((corr[..., -1], corr[..., -2], corr[..., -2] - corr[..., 4]))
        distance = np.abs(recent[:, :, :, None, :] - recent[:, :, None, :, :]).mean(axis=-1)
        distance[:, :, diag, diag] = np.inf
        distance.sort(axis=-1)
        parts.extend((distance[..., 0], distance[..., 1], distance[..., 3]))
    matrix = np.stack(parts, axis=-1).reshape(len(counts), 32, -1)
    return matrix, history[-HISTORY:].copy()


def feature_panel(values):
    blocks, previous = [], None
    for start in range(0, len(values), 128):
        block, previous = feature_block(values[start:start+128], previous)
        blocks.append(block)
    return np.concatenate(blocks)


def operating_threshold(scores, truth, local, allowed, budget):
    """Lowest strict threshold satisfying the empirical all-normal-window budget."""
    normal = ~np.asarray(truth, dtype=bool)
    if not normal.any() or not 0 < budget < 1:
        raise ValueError("Invalid calibration population or FPR budget")
    room = int(np.floor(budget * normal.sum())) - int((local & normal).sum())
    if room < 0:
        raise ValueError("Local alarms alone exceed the calibration budget")
    eligible = scores[normal & ~local & ~allowed]
    if not np.isfinite(eligible).all():
        raise ValueError("Nonfinite calibration scores")
    if len(eligible) <= room:
        return -1.0
    cut = len(eligible) - room - 1
    return float(np.partition(eligible, cut)[cut])


@dataclass
class CrossScenarioDetector:
    classifier: HistGradientBoostingClassifier
    threshold: float
    fit_runs: tuple[str, ...]
    calibration_runs: tuple[str, ...]

    def score_features(self, matrix):
        flat = matrix.reshape(-1, matrix.shape[-1])
        return self.classifier.predict_proba(flat)[:, 1].reshape(matrix.shape[:-1])

    def decide(self, scores, local, allowed):
        local = flags(local, scores.shape)
        allowed = flags(allowed, scores.shape)
        return local | (~allowed & (scores > self.threshold))


class WindowScorer:
    """Online adapter; the same transformation and decision rule are used by archived replay."""

    def __init__(self, detector):
        self.detector = detector
        self.run_id = None
        self.last_window = None
        self.previous = None

    def score(self, rows, decisions):
        by_nf = {row["nf_id"]: row for row in rows}
        if len(rows) != len(SLOTS) or set(by_nf) != set(SLOTS) or set(decisions) != set(SLOTS):
            raise ValueError("Expected one observation and decision per NF")
        runs = {row["run_id"] for row in rows}
        windows = {row["window_id"] for row in rows}
        if len(runs) != 1 or len(windows) != 1:
            raise ValueError("A scoring call must contain one run and window")
        run_id, window = runs.pop(), windows.pop()
        same_run = run_id == self.run_id
        if (same_run and window != self.last_window + 1) or (not same_run and window != 0):
            raise ValueError("Expected contiguous windows starting at zero")
        values = [[[by_nf[nf][k] for k in MEASUREMENTS] for nf in SLOTS]]
        matrix, previous = feature_block(values, self.previous if same_run else None)
        scores = self.detector.score_features(matrix)[0]
        alarms = self.detector.decide(scores, [decisions[n]["local_security"] for n in SLOTS],
                                    [decisions[n]["allowed"] for n in SLOTS])
        self.run_id, self.last_window, self.previous = run_id, window, previous
        return {nf: bool(alarm) for nf, alarm in zip(SLOTS, alarms)}
