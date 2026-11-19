"""Pure V10 confidence-gated fusion rule."""
from __future__ import annotations

import numpy as np


def v10_fusion_margin(v4_ratio, group_ratio, pair_ratio, k4_ratio, k5_ratio,
                      group_floor=1.08, v4_floor=1.25):
    """Return the frozen V10 fusion margin; alarms require margin > 0."""
    return np.maximum.reduce([
        np.asarray(group_ratio,dtype=float)-float(group_floor),
        np.asarray(v4_ratio,dtype=float)-float(v4_floor),
        np.asarray(pair_ratio,dtype=float)-1.0,
        np.asarray(k4_ratio,dtype=float)-1.0,
        np.asarray(k5_ratio,dtype=float)-1.0,
    ])
