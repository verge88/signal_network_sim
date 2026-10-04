from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SBA = ROOT / "5g"
if str(SBA) not in sys.path:
    sys.path.insert(0, str(SBA))

from run_semantic_v9_sensitivity import (  # noqa: E402
    DEFAULT_SENSITIVITY_SEEDS,
    SENSITIVITY_GRID,
)


class V9SensitivityContractTests(unittest.TestCase):
    def test_grid_is_fixed_and_unique(self):
        ids = [p.point_id for p in SENSITIVITY_GRID]
        self.assertEqual(len(ids), 10)
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(ids[0], "baseline")

    def test_grid_covers_requested_axes(self):
        by_id = {p.point_id: p for p in SENSITIVITY_GRID}
        self.assertEqual(by_id["marker_060"].marker_recall, 0.60)
        self.assertEqual(by_id["marker_075"].marker_recall, 0.75)
        self.assertEqual(by_id["marker_095"].marker_recall, 0.95)
        self.assertEqual(by_id["recovery_060"].recovery_probability, 0.60)
        self.assertEqual(by_id["recovery_075"].recovery_probability, 0.75)
        self.assertEqual(by_id["recovery_slow_2x"].recovery_latency_multiplier, 2.0)
        self.assertEqual(by_id["recovery_slow_4x"].recovery_latency_multiplier, 4.0)
        self.assertEqual(by_id["corr_error_010"].correlation_error_probability, 0.10)

    def test_degraded_combo_is_jointly_harder_than_baseline(self):
        by_id = {p.point_id: p for p in SENSITIVITY_GRID}
        baseline = by_id["baseline"]
        degraded = by_id["degraded_combo"]
        self.assertLess(degraded.marker_recall, baseline.marker_recall)
        self.assertLess(degraded.recovery_probability, baseline.recovery_probability)
        self.assertGreater(degraded.recovery_latency_multiplier, baseline.recovery_latency_multiplier)
        self.assertGreater(degraded.correlation_error_probability, baseline.correlation_error_probability)

    def test_all_points_validate_as_operational_configs(self):
        for point in SENSITIVITY_GRID:
            cfg = point.config()
            cfg.validate()

    def test_default_sensitivity_seeds_are_exploratory_subset(self):
        self.assertEqual(DEFAULT_SENSITIVITY_SEEDS, (42, 47, 51))


if __name__ == "__main__":
    unittest.main()
