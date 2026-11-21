"""Passive NF-window scoring with learned count trajectories and peer context."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .cross_scenario_detector import MEASUREMENTS, SLOTS, feature_block, flags, telemetry


METHOD = "PER_TYPE_RECALL"
CONTROL_NF = "slot-4"  # Control-observer placement in the deployed 32-NF topology.
LOOKBACK = 18


def missing_live_receipt(record, window, decision_time):
    """At the collector, compare local process liveness with receipt of this reply."""
    if record is None:
        return False
    completed = float(record["completed"])
    if (record["window_id"] != window or type(record["observer_alive"]) is not bool
            or not np.isfinite([completed, decision_time]).all() or completed > decision_time):
        raise ValueError("Control observation unavailable at the decision time")
    return record["observer_alive"] and record["delivered"] is None


def repeated_trajectories(captures):
    """Only trajectories seen in at least two distinct development acquisitions."""
    support = {}
    for counts, scenarios in captures:
        seen = set()
        for scenario in ("G", "S5"):
            active = scenarios == scenario
            for column in np.flatnonzero(active.any(axis=0)):
                mask = active[:, column]
                starts = np.flatnonzero(mask & ~np.r_[False, mask[:-1]])
                ends = np.flatnonzero(mask & ~np.r_[mask[1:], False]) + 1
                for start, end in zip(starts, ends):
                    path = tuple(counts[start:end, column].tolist())
                    if len(path) >= 2:
                        seen.add(path)
        for path in seen:
            support[path] = support.get(path, 0) + 1
    return [path for path, runs in sorted(support.items()) if runs >= 2]


class PrefixTracker:
    def __init__(self, paths):
        self.expected = np.concatenate(paths)
        self.length = np.concatenate([np.arange(1, len(path) + 1) for path in paths])
        self.first = self.length == 1
        self.active = np.zeros((32, len(self.expected)), bool)
        self.permit_age = np.zeros(32, int)

    def advance(self, counts, allowed):
        # A match must still explain the CURRENT observation. No scheduled start or alarm hold.
        self.active[:, 1:] = self.active[:, :-1]
        self.active[:, self.first] = True
        self.active &= counts[:, None] == self.expected
        self.permit_age = np.where(allowed, self.permit_age + 1, 0)
        nf, stage = np.nonzero(self.active)
        commonness = np.searchsorted([1, 3, 7, 15], self.active.sum(axis=0)[stage])
        age = self.permit_age[nf]
        length = self.length[stage]
        permit_phase = np.where(age == 0, 0, np.where(age == length, 1, np.where(age < length, 2, 3)))
        return nf, stage, commonness * 4 + permit_phase


@dataclass
class ProfileModel:
    paths: list
    scores: np.ndarray

    @classmethod
    def fit(cls, sets):
        paths = repeated_trajectories([(counts, labels) for counts, _, labels in sets])
        if not paths:
            return cls([], np.empty((0, 20)))
        tracker = PrefixTracker(paths)
        benign_hits = np.zeros((len(tracker.expected), 20))
        for counts, allowed, labels in sets:
            tracker = PrefixTracker(paths)
            for window, (row, permit) in enumerate(zip(counts, allowed)):
                nf, stage, context = tracker.advance(row, permit)
                normal = labels[window, nf] == "N1"
                np.add.at(benign_hits, (stage[normal], context[normal]), 1)
        # Score sustained matches by how often healthy NFs produced the same match
        # under comparable concurrent peer traffic and authorization.
        support = 1 - np.exp(-tracker.length[:, None] / 4)
        scores = support / np.sqrt(1 + benign_hits / len(sets))
        scores[tracker.first] = 0
        return cls(paths, scores)

    def score_window(self, tracker, counts, allowed):
        result = np.zeros(32)
        if tracker is not None:
            nf, stage, context = tracker.advance(counts, allowed)
            np.maximum.at(result, nf, self.scores[stage, context])
        return result

    def tracker(self):
        return PrefixTracker(self.paths) if self.paths else None


def shape_block(values, previous=None):
    values = telemetry(values)
    counts = values[:, :, 0].reshape(-1, 4, 8)
    prior = np.repeat(counts[:1], LOOKBACK, axis=0) if previous is None else previous
    graph, _ = feature_block(values, prior[-12:])
    history = np.concatenate((prior, counts))
    raw = sliding_window_view(history, LOOKBACK + 1, axis=0)
    now = raw[..., -1]
    net = raw.reshape(len(raw), 32, LOOKBACK + 1)
    net_center = np.median(net, axis=1)[:, None, None, -1]
    features = [raw[..., -2], raw[..., -3], now - net_center]
    for width in (3, 7, LOOKBACK):
        recent = raw[..., -width:]
        delta = np.diff(raw[..., -width-1:], axis=-1)
        sd = recent.std(axis=-1)
        mean = recent.mean(axis=-1)
        drift = delta.mean(axis=-1)
        features.extend((sd, np.ptp(recent, axis=-1), now - mean, drift,
                         (delta == 0).mean(axis=-1), np.abs(delta).max(axis=-1),
                         sd - np.median(sd, axis=2, keepdims=True)))
        # The full ordered peer distribution retains both a small cohort and the
        # quiet majority. Slot identity or the numeric order within a zone is never a feature.
        ordered = np.sort(mean, axis=2)
        for rank in (0, 2, 4, 5, 6, 7):
            features.append(mean - ordered[:, :, rank:rank + 1])
        pair = np.abs(delta[:, :, :, None, :] - delta[:, :, None, :, :])
        agreement = (pair <= 1).mean(axis=-1)
        diag = np.arange(8)
        agreement[:, :, diag, diag] = -1
        agreement.sort(axis=-1)
        features.extend((agreement[..., -2], agreement[..., -2] - agreement[..., 3]))
    shape = np.stack(features, axis=-1).reshape(len(values), 32, -1)
    return np.concatenate((graph, shape), axis=-1), history[-LOOKBACK:].copy()


def shape_panel(values):
    blocks, previous = [], None
    for start in range(0, len(values), 128):
        block, previous = shape_block(values[start:start+128], previous)
        blocks.append(block)
    return np.concatenate(blocks)


@dataclass
class PerTypeModel:
    classifier: object
    profiles: ProfileModel
    graph_cut: float
    profile_cut: float
    fit_runs: tuple
    calibration_runs: tuple

    def graph_scores(self, matrix):
        return self.classifier.predict_proba(matrix.reshape(-1, matrix.shape[-1]))[:, 1].reshape(matrix.shape[:-1])

    def score_panel(self, values, allowed):
        scores = self.graph_scores(shape_panel(values))
        tracker = self.profiles.tracker()
        profiles = np.stack([self.profiles.score_window(tracker, row[:, 0], grant)
                             for row, grant in zip(values, allowed, strict=True)])
        return scores, profiles

    def decide(self, local, loss, allowed, graph, profile):
        return local | loss | (~allowed & (graph > self.graph_cut)) | (profile > self.profile_cut)


class OnlineScorer:
    def __init__(self, model):
        self.model = model
        self.run, self.last, self.previous = None, None, None
        self.tracker = None

    def score(self, rows, decisions, control, decision_time, control_nf=CONTROL_NF):
        by_nf = {r["nf_id"]: r for r in rows}
        if len(rows) != 32 or set(by_nf) != set(SLOTS) or set(decisions) != set(SLOTS):
            raise ValueError("Expected one complete NF window")
        runs = {r["run_id"] for r in rows}
        windows = {r["window_id"] for r in rows}
        if len(runs) != 1 or len(windows) != 1 or control_nf not in by_nf:
            raise ValueError("Invalid scorer identities or control-observer placement")
        run, window = runs.pop(), windows.pop()
        new = run != self.run
        if window != (0 if new else self.last + 1):
            raise ValueError("Expected contiguous windows starting at zero")
        loss = np.zeros(32, bool)
        loss[SLOTS.index(control_nf)] = missing_live_receipt(control, window, decision_time)
        allowed = flags([decisions[n]["allowed"] for n in SLOTS], (32,))
        local = flags([decisions[n]["local_security"] for n in SLOTS], (32,))
        values = np.array([[[by_nf[n][k] for k in MEASUREMENTS] for n in SLOTS]], dtype=float)
        matrix, previous = shape_block(values, None if new else self.previous)
        graph = self.model.graph_scores(matrix)[0]
        tracker = self.model.profiles.tracker() if new else self.tracker
        profile = self.model.profiles.score_window(tracker, values[0, :, 0], allowed)
        alarms = self.model.decide(local, loss, allowed, graph, profile)
        self.run, self.last, self.previous, self.tracker = run, window, previous, tracker
        return dict(zip(SLOTS, alarms.tolist()))
