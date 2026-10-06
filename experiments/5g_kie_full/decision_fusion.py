from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from models import (
    SBI_FEATURES,
    TEMPORAL_FEATURES,
    ZONE_FEATURES,
    REPORT_FEATURES,
    KIE_FEATURES,
)


CHANNEL_FEATURES: Dict[str, List[str]] = {
    "SBI": list(SBI_FEATURES),
    "TEMPORAL": list(TEMPORAL_FEATURES),
    "ZONE": list(ZONE_FEATURES),
    "REPORT": list(REPORT_FEATURES),
    "KIE": list(KIE_FEATURES),
}


def _split_calibration(calibration_benign: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split benign calibration by complete windows.

    The first half estimates each channel's empirical null CDF; the second half
    calibrates the final fusion threshold. This avoids using attack labels and
    avoids calibrating the fusion threshold on the exact same observations that
    define the channel percentile transforms.
    """
    ids = np.asarray(sorted(calibration_benign["window_id"].unique()))
    if len(ids) < 4:
        cut = max(1, len(calibration_benign) // 2)
        return calibration_benign.iloc[:cut].copy(), calibration_benign.iloc[cut:].copy()

    mid = len(ids) // 2
    left_ids = set(ids[:mid].tolist())
    right_ids = set(ids[mid:].tolist())
    left = calibration_benign[calibration_benign["window_id"].isin(left_ids)].copy()
    right = calibration_benign[calibration_benign["window_id"].isin(right_ids)].copy()
    if left.empty or right.empty:
        cut = max(1, len(calibration_benign) // 2)
        return calibration_benign.iloc[:cut].copy(), calibration_benign.iloc[cut:].copy()
    return left, right


@dataclass
class ChannelIF:
    name: str
    model: IsolationForest
    features: List[str]
    reference_scores: np.ndarray

    def raw_score(self, df: pd.DataFrame) -> np.ndarray:
        return -self.model.score_samples(df[self.features].to_numpy(float))

    def percentile_score(self, df: pd.DataFrame) -> np.ndarray:
        raw = self.raw_score(df)
        ref = self.reference_scores
        # Mid-rank empirical CDF in (0, 1), robust to scores beyond the
        # calibration range while avoiding infinities in tail transforms.
        ranks = np.searchsorted(ref, raw, side="right")
        q = (ranks.astype(float) + 0.5) / (len(ref) + 1.0)
        return np.clip(q, 1e-6, 1.0 - 1e-6)


@dataclass
class FusionModel:
    channels: Mapping[str, ChannelIF]
    mode: str
    threshold: float
    weights: Mapping[str, float] | None = None

    def channel_matrix(self, df: pd.DataFrame) -> np.ndarray:
        return np.column_stack([self.channels[k].percentile_score(df) for k in self.channels])

    def score(self, df: pd.DataFrame) -> np.ndarray:
        q = self.channel_matrix(df)
        if self.mode in {"maxq", "sidak_or"}:
            return np.max(q, axis=1)
        if self.mode == "weighted":
            # Equal-by-default combination of calibrated upper-tail evidence.
            tail = -np.log(np.clip(1.0 - q, 1e-9, 1.0))
            if self.weights:
                w = np.asarray([float(self.weights[k]) for k in self.channels], dtype=float)
                w = w / np.sum(w)
            else:
                w = np.full(q.shape[1], 1.0 / q.shape[1])
            return tail @ w
        if self.mode == "noisy_or":
            return 1.0 - np.prod(1.0 - q, axis=1)
        raise ValueError(f"unknown fusion mode: {self.mode}")

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return (self.score(df) >= self.threshold).astype(int)


def fit_decision_fusion_family(
    train_benign: pd.DataFrame,
    calibration_benign: pd.DataFrame,
    target_fpr: float,
    seed: int,
    channel_features: Mapping[str, List[str]] | None = None,
) -> Dict[str, FusionModel]:
    """Fit label-free channel detectors and several decision-level fusions."""
    channel_features = dict(channel_features or CHANNEL_FEATURES)
    cal_ref, cal_fusion = _split_calibration(calibration_benign)

    channels: Dict[str, ChannelIF] = {}
    for idx, (name, features) in enumerate(channel_features.items()):
        model = IsolationForest(
            n_estimators=160,
            max_samples="auto",
            contamination="auto",
            random_state=seed + 1009 * idx,
            n_jobs=-1,
        )
        model.fit(train_benign[features].to_numpy(float))
        ref = -model.score_samples(cal_ref[features].to_numpy(float))
        ref = np.sort(np.asarray(ref, dtype=float))
        channels[name] = ChannelIF(
            name=name,
            model=model,
            features=list(features),
            reference_scores=ref,
        )

    # Temporary models are used only to obtain null fusion-score distributions.
    temp = {
        "fusion_maxq": FusionModel(channels=channels, mode="maxq", threshold=np.inf),
        "fusion_weighted": FusionModel(channels=channels, mode="weighted", threshold=np.inf),
        "fusion_noisy_or": FusionModel(channels=channels, mode="noisy_or", threshold=np.inf),
    }

    out: Dict[str, FusionModel] = {}
    for name, model in temp.items():
        null_score = model.score(cal_fusion)
        threshold = float(np.quantile(null_score, 1.0 - target_fpr))
        out[name] = FusionModel(
            channels=channels,
            mode=model.mode,
            threshold=threshold,
        )

    # Sidak correction provides an explicit OR rule with approximately controlled
    # family-wise benign false-positive rate under independent channels. In the
    # real data channels are not independent, so its empirical FPR is reported.
    k = max(len(channels), 1)
    alpha_component = 1.0 - (1.0 - target_fpr) ** (1.0 / k)
    out["fusion_sidak_or"] = FusionModel(
        channels=channels,
        mode="sidak_or",
        threshold=1.0 - alpha_component,
    )
    return out
