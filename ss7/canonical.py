"""Canonical scientific SS7 entry point.

Scientific experiments must use ss7/fix.py.  ss7_simulator_v7.py is legacy
until the GUI is migrated to the common simulator API.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

try:
    from .fix import (
        Episode,
        KeyCustody,
        RngBook,
        SimConfig,
        Sophistication,
        Ss7Sim,
        Topology,
        build_dataset,
    )
except ImportError:
    from fix import (
        Episode,
        KeyCustody,
        RngBook,
        SimConfig,
        Sophistication,
        Ss7Sim,
        Topology,
        build_dataset,
    )


@dataclass
class CanonicalRun:
    config: SimConfig
    dataframe: Any
    simulator: Ss7Sim
    episodes: list[Episode]


def simulate(
    config: Optional[SimConfig] = None,
    *,
    episodes_kw: Optional[dict] = None,
    topology: Optional[Topology] = None,
    size_capacities: bool = True,
    sim_template: Optional[Ss7Sim] = None,
) -> CanonicalRun:
    cfg = config or SimConfig()
    df, sim, episodes = build_dataset(
        cfg,
        episodes_kw=episodes_kw,
        topo=topology,
        size_capacities=size_capacities,
        sim_template=sim_template,
    )
    return CanonicalRun(cfg, df, sim, episodes)
