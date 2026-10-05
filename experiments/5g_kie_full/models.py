from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score


SBI_FEATURES = [
    "req_norm",
    "error_rate",
    "http2_rst_rate",
    "latency_ms",
    "streams_norm",
    "authz_403_rate",
    "token_req_rate",
    "discovery_rate",
    "profile_update_rate",
    "cross_slice_rate",
    "nrf_register_rate",
    "nrf_delete_rate",
    "scope_mismatch_rate",
]

TEMPORAL_FEATURES = ["temporal_req_z", "temporal_rst_z", "temporal_auth_z"]
ZONE_FEATURES = ["zone_req_z", "zone_error_z", "zone_rst_z", "zone_size"]
REPORT_FEATURES = ["report_observed_gap"]
KIE_FEATURES = ["kie_rtt_ms", "kie_loss", "kie_auth_fail", "heartbeat_age_s"]
ACTIVE_FEATURES = REPORT_FEATURES + KIE_FEATURES

FEATURE_SETS: Dict[str, List[str]] = {
    "SBI": SBI_FEATURES,
    "SBI_TEMPORAL": SBI_FEATURES + TEMPORAL_FEATURES,
    "SBI_ZONE": SBI_FEATURES + ZONE_FEATURES,
    "SBI_ACTIVE": SBI_FEATURES + ACTIVE_FEATURES,
    "FULL": SBI_FEATURES + TEMPORAL_FEATURES + ZONE_FEATURES + ACTIVE_FEATURES,
}


@dataclass
class IFComponent:
    model: IsolationForest
    features: List[str]
    center: float
    scale: float

    def zscore(self, df: pd.DataFrame) -> np.ndarray:
        raw = -self.model.score_samples(df[self.features].to_numpy(float))
        return (raw - self.center) / self.scale


@dataclass
class IFModel:
    """Calibrated group-wise Isolation-Forest fusion.

    Each evidence source (SBI, temporal, zone, active/KIE) is modeled by its own
    benign Isolation Forest. The final anomaly score is the maximum calibrated
    component score, then re-calibrated on benign data to the requested FPR.
    """

    components: List[IFComponent]
    threshold: float
    features: List[str]

    def score(self, df: pd.DataFrame) -> np.ndarray:
        z = np.column_stack([c.zscore(df) for c in self.components])
        return np.max(z, axis=1)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return (self.score(df) >= self.threshold).astype(int)


@dataclass
class RFModel:
    model: RandomForestClassifier
    features: List[str]
    threshold: float = 0.5

    def score(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(df[self.features].to_numpy(float))[:, 1]

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return (self.score(df) >= self.threshold).astype(int)


def _group_components(features: List[str]) -> List[List[str]]:
    f = set(features)
    groups: List[List[str]] = []
    if set(SBI_FEATURES).issubset(f):
        groups.append(SBI_FEATURES)
    if set(TEMPORAL_FEATURES).issubset(f):
        groups.append(TEMPORAL_FEATURES)
    if set(ZONE_FEATURES).issubset(f):
        groups.append(ZONE_FEATURES)
    if set(ACTIVE_FEATURES).issubset(f):
        groups.append(REPORT_FEATURES)
        groups.append(KIE_FEATURES)
    if not groups:
        groups = [features]
    return [list(g) for g in groups]


def fit_iforest(
    train_benign: pd.DataFrame,
    calibration_benign: pd.DataFrame,
    features: List[str],
    target_fpr: float,
    seed: int,
) -> IFModel:
    components: List[IFComponent] = []
    cal_z = []
    for idx, group in enumerate(_group_components(features)):
        model = IsolationForest(
            n_estimators=160,
            max_samples="auto",
            contamination="auto",
            random_state=seed + idx * 997,
            n_jobs=-1,
        )
        model.fit(train_benign[group].to_numpy(float))
        raw_cal = -model.score_samples(calibration_benign[group].to_numpy(float))
        center = float(np.median(raw_cal))
        mad = float(np.median(np.abs(raw_cal - center)))
        scale = max(1.4826 * mad, 1e-4)
        comp = IFComponent(model=model, features=group, center=center, scale=scale)
        components.append(comp)
        cal_z.append((raw_cal - center) / scale)

    combined_cal = np.max(np.column_stack(cal_z), axis=1)
    threshold = float(np.quantile(combined_cal, 1.0 - target_fpr))
    return IFModel(components=components, threshold=threshold, features=list(features))


def fit_random_forest(
    train_frame: pd.DataFrame,
    features: List[str],
    seed: int,
    n_estimators: int,
    n_jobs: int,
) -> RFModel:
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=seed,
        n_jobs=n_jobs,
    )
    model.fit(train_frame[features].to_numpy(float), train_frame["label"].to_numpy(int))
    return RFModel(model=model, features=list(features))


def binary_metrics(df: pd.DataFrame, score: np.ndarray, pred: np.ndarray) -> dict:
    y = df["label"].to_numpy(int)
    out = {
        "n": int(len(df)),
        "positives": int(y.sum()),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "fpr": float(((pred == 1) & (y == 0)).sum() / max((y == 0).sum(), 1)),
    }
    out["pr_auc"] = float(average_precision_score(y, score)) if y.min() != y.max() else float("nan")
    return out


def detection_latency_windows(df: pd.DataFrame, pred: np.ndarray) -> float:
    work = df[["window_id", "label"]].copy()
    work["pred"] = pred
    attacked = work[work["label"] == 1]
    if attacked.empty:
        return float("nan")
    start = int(attacked["window_id"].min())
    detected = attacked[attacked["pred"] == 1]
    if detected.empty:
        return float("nan")
    return float(int(detected["window_id"].min()) - start)
