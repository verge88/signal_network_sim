from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping

import numpy as np
import pandas as pd

from decision_fusion import ChannelIF, _split_calibration, fit_decision_fusion_family
from models import IFModel, FEATURE_SETS, fit_iforest


@dataclass
class GuardedCascadeModel:
    """SBI-first cascade with low-FPR auxiliary channels.

    Stage 1 keeps a dedicated SBI detector.
    Stage 2 adds report/KIE evidence.
    Stage 3 adds temporal/zone evidence.

    Auxiliary thresholds are calibrated only on benign data. Attack labels are
    never used to fit or tune the cascade.
    """

    sbi_model: IFModel
    channels: Mapping[str, ChannelIF]
    active_threshold: float
    context_threshold: float
    name: str

    def component_scores(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        q_sbi = self.channels["SBI"].percentile_score(df)
        q_report = self.channels["REPORT"].percentile_score(df)
        q_kie = self.channels["KIE"].percentile_score(df)
        q_temporal = self.channels["TEMPORAL"].percentile_score(df)
        q_zone = self.channels["ZONE"].percentile_score(df)
        return {
            "sbi_q": q_sbi,
            "active_q": np.maximum(q_report, q_kie),
            "context_q": np.maximum(q_temporal, q_zone),
        }

    def stage_predictions(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        scores = self.component_scores(df)
        sbi = self.sbi_model.predict(df).astype(bool)
        active = scores["active_q"] >= self.active_threshold
        context = scores["context_q"] >= self.context_threshold
        final = sbi | active | context
        return {
            "sbi": sbi,
            "active": active,
            "context": context,
            "final": final,
        }

    def score(self, df: pd.DataFrame) -> np.ndarray:
        # Label-free continuous ranking used for PR-AUC. Final classification
        # still follows the guarded stage-specific thresholds above.
        s = self.component_scores(df)
        return np.maximum.reduce([s["sbi_q"], s["active_q"], s["context_q"]])

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.stage_predictions(df)["final"].astype(int)


def _tail_threshold(values: np.ndarray, alpha: float) -> float:
    if alpha <= 0:
        return 1.0
    return float(np.quantile(values, 1.0 - alpha))



def fit_preserve_sbi_cascade(
    train_benign: pd.DataFrame,
    calibration_benign: pd.DataFrame,
    seed: int,
    active_fpr_budget: float,
    context_fpr_budget: float = 0.0005,
    sbi_target_fpr: float = 0.01,
    name: str | None = None,
) -> GuardedCascadeModel:
    """Fit one SBI-preserving cascade with explicit auxiliary FPR budgets.

    The SBI stage is calibrated exactly as the baseline. ACTIVE/KIE and CONTEXT
    thresholds are estimated from benign calibration data only.
    """
    fusion_family = fit_decision_fusion_family(
        train_benign,
        calibration_benign,
        target_fpr=sbi_target_fpr,
        seed=seed + 7000,
    )
    channels = fusion_family["fusion_maxq"].channels
    _, cal_tail = _split_calibration(calibration_benign)

    q_report = channels["REPORT"].percentile_score(cal_tail)
    q_kie = channels["KIE"].percentile_score(cal_tail)
    q_temporal = channels["TEMPORAL"].percentile_score(cal_tail)
    q_zone = channels["ZONE"].percentile_score(cal_tail)

    active_null = np.maximum(q_report, q_kie)
    context_null = np.maximum(q_temporal, q_zone)

    sbi = fit_iforest(
        train_benign,
        calibration_benign,
        features=FEATURE_SETS["SBI"],
        target_fpr=sbi_target_fpr,
        seed=seed,
    )
    model_name = name or f"cascade_active_{active_fpr_budget:.5f}"
    return GuardedCascadeModel(
        sbi_model=sbi,
        channels=channels,
        active_threshold=_tail_threshold(active_null, active_fpr_budget),
        context_threshold=_tail_threshold(context_null, context_fpr_budget),
        name=model_name,
    )

def fit_guarded_cascade_family(
    train_benign: pd.DataFrame,
    calibration_benign: pd.DataFrame,
    seed: int,
    target_fpr: float = 0.01,
) -> Dict[str, GuardedCascadeModel]:
    """Fit two cascades for the next falsification-oriented iteration.

    preserve_sbi:
        keeps the original 1% SBI detector unchanged and spends small extra
        benign-FPR budgets on ACTIVE and CONTEXT evidence. This tests whether
        hidden-compromise recall can be added without sacrificing standard SBI
        attack sensitivity.

    equal_fpr:
        allocates approximately the same 1% total nominal FPR across stages.
        This is the fair-FPR comparison against SBI-only.
    """

    # Reuse the label-free percentile calibration from the fusion experiment.
    fusion_family = fit_decision_fusion_family(
        train_benign,
        calibration_benign,
        target_fpr=target_fpr,
        seed=seed + 7000,
    )
    channels = fusion_family["fusion_maxq"].channels
    _, cal_tail = _split_calibration(calibration_benign)

    q_report = channels["REPORT"].percentile_score(cal_tail)
    q_kie = channels["KIE"].percentile_score(cal_tail)
    q_temporal = channels["TEMPORAL"].percentile_score(cal_tail)
    q_zone = channels["ZONE"].percentile_score(cal_tail)
    active_null = np.maximum(q_report, q_kie)
    context_null = np.maximum(q_temporal, q_zone)

    # Preserve the baseline detector exactly at the original 1% calibration.
    sbi_preserve = fit_iforest(
        train_benign,
        calibration_benign,
        features=FEATURE_SETS["SBI"],
        target_fpr=target_fpr,
        seed=seed,
    )
    preserve = GuardedCascadeModel(
        sbi_model=sbi_preserve,
        channels=channels,
        active_threshold=_tail_threshold(active_null, 0.0010),
        context_threshold=_tail_threshold(context_null, 0.0005),
        name="cascade_preserve_sbi",
    )

    # Equal-FPR budget: 0.85% SBI + 0.10% active + 0.05% context.
    sbi_equal = fit_iforest(
        train_benign,
        calibration_benign,
        features=FEATURE_SETS["SBI"],
        target_fpr=0.0085,
        seed=seed + 17,
    )
    equal = GuardedCascadeModel(
        sbi_model=sbi_equal,
        channels=channels,
        active_threshold=_tail_threshold(active_null, 0.0010),
        context_threshold=_tail_threshold(context_null, 0.0005),
        name="cascade_equal_fpr",
    )

    return {
        preserve.name: preserve,
        equal.name: equal,
    }
