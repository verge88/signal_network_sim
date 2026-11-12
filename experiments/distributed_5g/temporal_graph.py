"""Causal temporal-graph anomaly features for the 32-NF laboratory topology.

Nodes are logical NF slots. Edges connect the eight slots in each predeclared
zone. The detector uses only measured aggregate features and past/current peer
relations; labels, scenario names, timestamps and run/window identifiers never
enter the fitted estimator.
"""
from collections import defaultdict, deque
import copy

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

BASE = ("request_count", "interarrival_mean", "latency_ms", "error_rate")
GRAPH_FEATURES = (
    "count",
    "count_delta_1",
    "count_delta_3",
    "count_vs_self_mean5",
    "peer_count_mean",
    "peer_count_gap",
    "peer_delta_mean",
    "peer_delta_std",
    "delta_sign_agreement",
    "edge_coherence",
    "zone_count_std",
    "latency_delta_1",
    "peer_latency_delta_mean",
    "interarrival_delta_1",
    "peer_interarrival_delta_mean",
)


def slot_number(nf_id):
    text = str(nf_id)
    if not text.startswith("slot-"):
        raise ValueError(f"Temporal graph requires slot-N identifiers, got {text!r}")
    value = int(text.split("-", 1)[1])
    if not 0 <= value < 32:
        raise ValueError(f"Unexpected NF slot {value}")
    return value


def zone_number(nf_id):
    return slot_number(nf_id) // 8


class TemporalGraphDetector:
    """Isolation Forest over causal temporal and within-zone graph features."""

    def __init__(self, seed=7143, n_estimators=256, max_samples=256, history=8):
        self.seed = int(seed)
        self.n_estimators = int(n_estimators)
        self.max_samples = int(max_samples)
        self.history_size = int(history)
        self.model = None
        self.reset()

    def get_params(self, deep=True):
        return {
            "seed": self.seed,
            "n_estimators": self.n_estimators,
            "max_samples": self.max_samples,
            "history": self.history_size,
            "graph": "four fixed 8-NF zones; complete peer graph within each zone",
            "features": list(GRAPH_FEATURES),
        }

    def reset(self):
        self.history = defaultdict(lambda: deque(maxlen=self.history_size))

    @staticmethod
    def _prior(history, name, lag=1):
        if len(history) < lag:
            return None
        return float(history[-lag][name])

    @staticmethod
    def _sign(value, eps=1e-12):
        return 1 if value > eps else -1 if value < -eps else 0

    def _window_features(self, window):
        if len(window) != 32 or window["nf_id"].nunique() != 32:
            raise ValueError("Temporal graph scoring requires exactly 32 unique NF rows per window")
        rows = {str(row.nf_id): row for row in window.itertuples(index=False)}
        deltas = {}
        latency_deltas = {}
        interarrival_deltas = {}
        for nf, row in rows.items():
            hist = self.history[nf]
            prior_count = self._prior(hist, "request_count")
            prior_latency = self._prior(hist, "latency_ms")
            prior_interarrival = self._prior(hist, "interarrival_mean")
            deltas[nf] = 0.0 if prior_count is None else float(row.request_count) - prior_count
            latency_deltas[nf] = 0.0 if prior_latency is None else float(row.latency_ms) - prior_latency
            interarrival_deltas[nf] = 0.0 if prior_interarrival is None else float(row.interarrival_mean) - prior_interarrival

        feature_rows = []
        for nf, row in rows.items():
            zone = zone_number(nf)
            peers = [other for other in rows if other != nf and zone_number(other) == zone]
            peer_counts = np.asarray([float(rows[p].request_count) for p in peers], dtype=float)
            peer_delta = np.asarray([deltas[p] for p in peers], dtype=float)
            peer_latency_delta = np.asarray([latency_deltas[p] for p in peers], dtype=float)
            peer_interarrival_delta = np.asarray([interarrival_deltas[p] for p in peers], dtype=float)
            hist = self.history[nf]
            prior3 = self._prior(hist, "request_count", 3)
            self_mean5 = np.mean([float(item["request_count"]) for item in list(hist)[-5:]]) if hist else float(row.request_count)
            sign = self._sign(deltas[nf])
            agreement = np.mean([self._sign(x) == sign for x in peer_delta]) if sign else np.mean(np.abs(peer_delta) < 1e-12)
            peer_delta_mean = float(peer_delta.mean())
            vector = [
                float(row.request_count),
                float(deltas[nf]),
                0.0 if prior3 is None else float(row.request_count) - prior3,
                float(row.request_count) - float(self_mean5),
                float(peer_counts.mean()),
                float(row.request_count) - float(peer_counts.mean()),
                peer_delta_mean,
                float(peer_delta.std()),
                float(agreement),
                float(deltas[nf] * peer_delta_mean),
                float(np.std([float(rows[p].request_count) for p in rows if zone_number(p) == zone])),
                float(latency_deltas[nf]),
                float(peer_latency_delta.mean()),
                float(interarrival_deltas[nf]),
                float(peer_interarrival_delta.mean()),
            ]
            feature_rows.append((nf, vector))

        for nf, row in rows.items():
            self.history[nf].append({
                "request_count": float(row.request_count),
                "latency_ms": float(row.latency_ms),
                "interarrival_mean": float(row.interarrival_mean),
            })
        return feature_rows

    def transform(self, frame):
        frame = pd.DataFrame(frame).copy()
        required = {"run_id", "window_id", "nf_id", *BASE}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Missing temporal graph inputs: {sorted(missing)}")
        output = np.zeros((len(frame), len(GRAPH_FEATURES)), dtype=float)
        positions = {idx: pos for pos, idx in enumerate(frame.index)}
        for (_, window_id), part in frame.groupby(["run_id", "window_id"], sort=False):
            if not part["window_id"].eq(window_id).all():
                raise AssertionError("Window grouping failed")
            features = self._window_features(part)
            by_nf = {nf: vector for nf, vector in features}
            for idx, row in part.iterrows():
                output[positions[idx]] = by_nf[str(row["nf_id"])]
        if not np.isfinite(output).all():
            raise ValueError("Nonfinite temporal graph feature")
        return output

    def fit(self, frame):
        self.reset()
        graph = self.transform(frame)
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=min(self.max_samples, len(graph)),
            contamination="auto",
            random_state=self.seed,
            n_jobs=1,
        )
        self.model.fit(graph)
        self.reset()
        return self

    def decision_function(self, frame):
        if self.model is None:
            raise ValueError("TemporalGraphDetector must be fitted")
        graph = self.transform(frame)
        return -np.asarray(self.model.score_samples(graph), dtype=float)

    def clean_copy(self):
        clone = copy.deepcopy(self)
        clone.reset()
        return clone
