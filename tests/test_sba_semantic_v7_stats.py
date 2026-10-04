from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SBA = ROOT / "5g"
if str(SBA) not in sys.path:
    sys.path.insert(0, str(SBA))

from run_semantic_v7 import (  # noqa: E402
    bootstrap_delta_ci,
    holm_adjust,
    paired_signflip_p,
)


class SemanticV7StatisticsTests(unittest.TestCase):
    def test_bootstrap_constant_delta_is_constant(self):
        lo, hi = bootstrap_delta_ci([0.2] * 10, seed=42, n_boot=500)
        self.assertAlmostEqual(lo, 0.2)
        self.assertAlmostEqual(hi, 0.2)

    def test_signflip_zero_delta_returns_one(self):
        self.assertEqual(paired_signflip_p([0.0] * 10, seed=42), 1.0)

    def test_signflip_exact_for_four_equal_positive_pairs(self):
        # For four equal positive differences, only all-plus and all-minus are
        # as extreme as the observed mean: 2 / 2**4 = 0.125.
        p = paired_signflip_p([1.0, 1.0, 1.0, 1.0], seed=42)
        self.assertAlmostEqual(p, 0.125)

    def test_holm_adjust_is_monotone_in_sorted_p_order(self):
        raw = [0.01, 0.04, 0.02, float("nan")]
        adj = holm_adjust(raw)
        finite = [(raw[i], adj[i]) for i in range(3)]
        finite.sort()
        adjusted = [x[1] for x in finite]
        self.assertLessEqual(adjusted[0], adjusted[1])
        self.assertLessEqual(adjusted[1], adjusted[2])
        self.assertTrue(math.isnan(adj[3]))


if __name__ == "__main__":
    unittest.main()
