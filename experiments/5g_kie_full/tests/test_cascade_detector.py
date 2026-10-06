from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cascade_detector import fit_guarded_cascade_family
from config import ExperimentConfig
from features import TemporalState, add_features
from models import FEATURE_SETS, fit_iforest
from simulator import AttackSpec, FiveGCoreSimulator


def _frame(cfg: ExperimentConfig, seed: int, attack, n: int):
    sim = FiveGCoreSimulator(cfg.fleet, seed)
    state = TemporalState()
    _ = add_features(sim.generate(cfg.warmup_windows), state)
    return add_features(sim.generate(n, attack), state)


def test_preserve_cascade_uses_identical_sbi_stage():
    cfg = ExperimentConfig(
        train_windows=90,
        calibration_windows=60,
        warmup_windows=30,
        eval_windows=30,
        seeds=(1,),
        severities=(1.0,),
        supervised_attack_windows=12,
        random_forest_trees=30,
    )
    train = _frame(cfg, 401, None, cfg.train_windows)
    cal = _frame(cfg, 402, None, cfg.calibration_windows)
    eval_frame = _frame(cfg, 403, AttackSpec("rapid_reset", 1.0), cfg.eval_windows)

    baseline = fit_iforest(
        train, cal,
        FEATURE_SETS["SBI"],
        target_fpr=cfg.contamination_target_fpr,
        seed=77,
    )
    cascade = fit_guarded_cascade_family(
        train.drop(columns=["label"]),
        cal.drop(columns=["label"]),
        seed=77,
        target_fpr=cfg.contamination_target_fpr,
    )["cascade_preserve_sbi"]

    assert np.array_equal(
        baseline.predict(eval_frame),
        cascade.stage_predictions(eval_frame)["sbi"].astype(int),
    )


def test_adaptive_lie_reduces_report_gap_but_remains_malicious():
    cfg = ExperimentConfig(
        train_windows=50,
        calibration_windows=40,
        warmup_windows=20,
        eval_windows=35,
        seeds=(1,),
        severities=(1.0,),
        supervised_attack_windows=10,
        random_forest_trees=20,
    )
    lie = _frame(cfg, 501, AttackSpec("compromised_lie", 1.0), cfg.eval_windows)
    adaptive = _frame(cfg, 501, AttackSpec("adaptive_lie", 1.0), cfg.eval_windows)

    lie_gap = lie.loc[lie["label"] == 1, "report_observed_gap"].mean()
    adaptive_gap = adaptive.loc[adaptive["label"] == 1, "report_observed_gap"].mean()

    assert adaptive["label"].sum() > 0
    assert adaptive_gap < lie_gap
    assert adaptive_gap > 0.0


def test_cascade_adds_hidden_detection_without_dropping_sbi_stage():
    cfg = ExperimentConfig(
        train_windows=100,
        calibration_windows=70,
        warmup_windows=30,
        eval_windows=35,
        seeds=(1,),
        severities=(1.0,),
        supervised_attack_windows=10,
        random_forest_trees=20,
    )
    train = _frame(cfg, 601, None, cfg.train_windows)
    cal = _frame(cfg, 602, None, cfg.calibration_windows)
    hidden = _frame(cfg, 603, AttackSpec("compromised_lie", 1.0), cfg.eval_windows)

    cascade = fit_guarded_cascade_family(
        train, cal,
        seed=31,
        target_fpr=0.02,
    )["cascade_preserve_sbi"]
    stages = cascade.stage_predictions(hidden)
    y = hidden["label"].to_numpy(int).astype(bool)

    sbi_recall = float((stages["sbi"] & y).sum() / max(y.sum(), 1))
    final_recall = float((stages["final"] & y).sum() / max(y.sum(), 1))
    assert final_recall >= sbi_recall
    assert np.all(stages["final"] | (~stages["sbi"]))
