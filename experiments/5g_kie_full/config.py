from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass(frozen=True)
class FleetConfig:
    plmn: str = "00101"
    slices: Tuple[str, ...] = ("1-010203", "1-112233")
    localities: Tuple[str, ...] = ("edge-a", "edge-b")
    nf_types: Tuple[str, ...] = ("AMF", "SMF", "UDM", "AUSF", "PCF")
    replicas_per_zone: int = 3
    window_s: float = 30.0
    churn_probability: float = 0.015
    churn_hold_windows: int = 4


@dataclass(frozen=True)
class ExperimentConfig:
    train_windows: int = 350
    calibration_windows: int = 150
    warmup_windows: int = 80
    eval_windows: int = 90
    supervised_attack_windows: int = 55
    contamination_target_fpr: float = 0.01
    seeds: Tuple[int, ...] = (101, 202, 303, 404, 505)
    severities: Tuple[float, ...] = (0.55, 1.0, 1.5)
    fleet: FleetConfig = field(default_factory=FleetConfig)
    random_forest_trees: int = 250
    n_jobs: int = -1
