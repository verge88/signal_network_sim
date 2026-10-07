from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cascade_detector import fit_preserve_sbi_cascade_sweep
from config import ExperimentConfig
from features import TemporalState, add_features
from simulator import AttackSpec, FiveGCoreSimulator


def _frame(cfg: ExperimentConfig, seed: int, attack, n: int):
    sim = FiveGCoreSimulator(cfg.fleet, seed)
    state = TemporalState()
    _ = add_features(sim.generate(cfg.warmup_windows), state)
    return add_features(sim.generate(n, attack), state)


def test_adaptive_alpha_endpoints_match_existing_attack_endpoints():
    cfg = ExperimentConfig(
        train_windows=40,
        calibration_windows=30,
        warmup_windows=20,
        eval_windows=20,
        seeds=(1,),
        severities=(1.0,),
        supervised_attack_windows=10,
        random_forest_trees=20,
    )

    lie = _frame(cfg, 901, AttackSpec("compromised_lie", 1.0), cfg.eval_windows)
    a0 = _frame(cfg, 901, AttackSpec("adaptive_lie", 1.0, report_alpha=0.0), cfg.eval_windows)
    truth = _frame(cfg, 902, AttackSpec("compromised_truth", 1.0), cfg.eval_windows)
    a1 = _frame(cfg, 902, AttackSpec("adaptive_lie", 1.0, report_alpha=1.0), cfg.eval_windows)

    assert np.allclose(lie["observed_rps"], a0["observed_rps"])
    assert np.allclose(lie["report_rps"], a0["report_rps"])
    assert np.allclose(truth["observed_rps"], a1["observed_rps"])
    assert np.allclose(truth["report_rps"], a1["report_rps"])


def test_active_budget_sweep_has_monotone_thresholds_and_preserves_sbi_stage():
    cfg = ExperimentConfig(
        train_windows=80,
        calibration_windows=60,
        warmup_windows=25,
        eval_windows=25,
        seeds=(1,),
        severities=(1.0,),
        supervised_attack_windows=10,
        random_forest_trees=20,
    )
    train = _frame(cfg, 911, None, cfg.train_windows)
    cal = _frame(cfg, 912, None, cfg.calibration_windows)
    eval_frame = _frame(
        cfg,
        913,
        AttackSpec("adaptive_lie", 1.0, report_alpha=0.95),
        cfg.eval_windows,
    )

    budgets = (0.0001, 0.001, 0.005)
    family = fit_preserve_sbi_cascade_sweep(
        train.drop(columns=["label"]),
        cal.drop(columns=["label"]),
        seed=37,
        active_fpr_budgets=budgets,
        context_fpr_budget=0.0005,
        sbi_target_fpr=0.01,
    )

    thresholds = [family[b].active_threshold for b in budgets]
    assert thresholds[0] >= thresholds[1] >= thresholds[2]

    sbi0 = family[budgets[0]].stage_predictions(eval_frame)["sbi"]
    for b in budgets[1:]:
        assert np.array_equal(
            sbi0,
            family[b].stage_predictions(eval_frame)["sbi"],
        )


def test_more_active_budget_cannot_remove_positive_predictions():
    cfg = ExperimentConfig(
        train_windows=70,
        calibration_windows=55,
        warmup_windows=25,
        eval_windows=25,
        seeds=(1,),
        severities=(1.0,),
        supervised_attack_windows=10,
        random_forest_trees=20,
    )
    train = _frame(cfg, 921, None, cfg.train_windows)
    cal = _frame(cfg, 922, None, cfg.calibration_windows)
    frame = _frame(
        cfg,
        923,
        AttackSpec("adaptive_lie", 1.0, report_alpha=0.9),
        cfg.eval_windows,
    )
    family = fit_preserve_sbi_cascade_sweep(
        train,
        cal,
        seed=41,
        active_fpr_budgets=(0.0001, 0.005),
    )
    strict = family[0.0001].predict(frame).astype(bool)
    loose = family[0.005].predict(frame).astype(bool)
    assert np.all((~strict) | loose)
