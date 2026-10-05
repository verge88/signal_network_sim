from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import ExperimentConfig, FleetConfig
from features import TemporalState, add_features
from models import FEATURE_SETS, fit_iforest
from simulator import AttackSpec, FiveGCoreSimulator


def test_simulator_has_dynamic_zone_and_active_features():
    cfg = FleetConfig(churn_probability=0.05)
    sim = FiveGCoreSimulator(cfg, seed=7)
    raw = sim.generate(12, AttackSpec("compromised_lie", 1.0))
    feat = add_features(raw, TemporalState())
    required = {
        "zone_key",
        "report_observed_gap",
        "temporal_req_z",
        "zone_req_z",
        "kie_auth_fail",
        "heartbeat_age_s",
    }
    assert required.issubset(set(feat.columns))
    assert feat["zone_size"].min() >= 1
    assert feat["label"].sum() > 0


def test_active_features_help_lying_compromise():
    cfg = ExperimentConfig(
        train_windows=90,
        calibration_windows=45,
        warmup_windows=30,
        eval_windows=30,
        seeds=(11,),
        severities=(1.0,),
        supervised_attack_windows=15,
        random_forest_trees=50,
    )

    def feat(seed, attack, n):
        sim = FiveGCoreSimulator(cfg.fleet, seed)
        state = TemporalState()
        _ = add_features(sim.generate(cfg.warmup_windows), state)
        return add_features(sim.generate(n, attack), state)

    train = feat(101, None, cfg.train_windows)
    cal = feat(102, None, cfg.calibration_windows)
    attack = feat(103, AttackSpec("compromised_lie", 1.0), cfg.eval_windows)

    sbi = fit_iforest(train, cal, FEATURE_SETS["SBI"], 0.02, 1)
    full = fit_iforest(train, cal, FEATURE_SETS["FULL"], 0.02, 1)

    y = attack["label"].to_numpy(int)
    sbi_recall = ((sbi.predict(attack) == 1) & (y == 1)).sum() / max(y.sum(), 1)
    full_recall = ((full.predict(attack) == 1) & (y == 1)).sum() / max(y.sum(), 1)
    assert full_recall >= sbi_recall
    assert full_recall >= 0.70
