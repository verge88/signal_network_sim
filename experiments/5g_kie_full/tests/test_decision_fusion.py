from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import ExperimentConfig
from decision_fusion import fit_decision_fusion_family
from features import TemporalState, add_features
from simulator import AttackSpec, FiveGCoreSimulator


def _frame(cfg: ExperimentConfig, seed: int, attack, n: int):
    sim = FiveGCoreSimulator(cfg.fleet, seed)
    state = TemporalState()
    _ = add_features(sim.generate(cfg.warmup_windows), state)
    return add_features(sim.generate(n, attack), state)


def _recall(frame, pred) -> float:
    y = frame["label"].to_numpy(int)
    return float(((pred == 1) & (y == 1)).sum() / max(y.sum(), 1))


def _fpr(frame, pred) -> float:
    y = frame["label"].to_numpy(int)
    return float(((pred == 1) & (y == 0)).sum() / max((y == 0).sum(), 1))


def test_decision_fusion_is_label_free_and_preserves_two_channels():
    cfg = ExperimentConfig(
        train_windows=100,
        calibration_windows=70,
        warmup_windows=35,
        eval_windows=35,
        seeds=(11,),
        severities=(1.0,),
        supervised_attack_windows=15,
        random_forest_trees=50,
    )
    train = _frame(cfg, 201, None, cfg.train_windows)
    cal = _frame(cfg, 202, None, cfg.calibration_windows)

    # Training/calibration labels are deliberately removed: unsupervised fusion
    # must depend only on benign feature distributions.
    family = fit_decision_fusion_family(
        train.drop(columns=["label"]),
        cal.drop(columns=["label"]),
        target_fpr=0.02,
        seed=17,
    )
    model = family["fusion_maxq"]

    normal = _frame(cfg, 203, None, cfg.eval_windows)
    lie = _frame(cfg, 204, AttackSpec("compromised_lie", 1.5), cfg.eval_windows)
    rapid = _frame(cfg, 205, AttackSpec("rapid_reset", 1.5), cfg.eval_windows)

    assert _fpr(normal, model.predict(normal)) <= 0.08
    assert _recall(lie, model.predict(lie)) >= 0.65
    assert _recall(rapid, model.predict(rapid)) >= 0.65


def test_all_fusion_variants_score_and_predict():
    cfg = ExperimentConfig(
        train_windows=70,
        calibration_windows=50,
        warmup_windows=25,
        eval_windows=20,
        seeds=(1,),
        severities=(1.0,),
        supervised_attack_windows=10,
        random_forest_trees=20,
    )
    train = _frame(cfg, 301, None, cfg.train_windows)
    cal = _frame(cfg, 302, None, cfg.calibration_windows)
    eval_frame = _frame(cfg, 303, AttackSpec("slow_drift_lie", 1.0), cfg.eval_windows)

    family = fit_decision_fusion_family(train, cal, target_fpr=0.02, seed=23)
    assert {"fusion_maxq", "fusion_weighted", "fusion_noisy_or", "fusion_sidak_or"} <= set(family)

    for model in family.values():
        score = model.score(eval_frame)
        pred = model.predict(eval_frame)
        assert len(score) == len(eval_frame)
        assert len(pred) == len(eval_frame)
        assert set(pred.tolist()) <= {0, 1}
