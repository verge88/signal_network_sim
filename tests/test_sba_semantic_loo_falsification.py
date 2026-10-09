from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "5g"))

from run_semantic_loo_falsification import (  # noqa: E402
    FACT, ORIGINS, conformal_p, intervene, score, wilson_interval,
)
from sba_semantic_v5 import OriginObservation  # noqa: E402
from sba_semantic_v8 import EvidenceState  # noqa: E402


class LooFalsificationTests(unittest.TestCase):
    def _reports(self):
        return {
            origin: OriginObservation(
                event_id="evt-1",
                fact=FACT,
                origin=origin,
                trust_root=origin.value,
                state=EvidenceState.CONSISTENT,
                observed_at=0.0,
            )
            for origin in ORIGINS
        }

    def test_single_origin_attack_is_erased_by_min_loo(self):
        base = self._reports()
        self.assertEqual(score(base), (0, 0))
        injected = intervene(base, list(ORIGINS), 1)
        self.assertEqual(score(injected), (1, 0))

    def test_two_and_three_origin_attacks(self):
        base = self._reports()
        self.assertEqual(score(intervene(base, list(ORIGINS), 2)), (2, 1))
        self.assertEqual(score(intervene(base, list(ORIGINS), 3)), (3, 2))

    def test_unknown_is_never_promoted(self):
        base = self._reports()
        first = ORIGINS[0]
        base[first] = base[first].__class__(
            **{**base[first].__dict__, "state": EvidenceState.UNKNOWN}
        )
        observed_consistent = [x for x in ORIGINS if x != first]
        out = intervene(base, observed_consistent, 1)
        self.assertEqual(out[first].state, EvidenceState.UNKNOWN)

    def test_conformal_ties_are_conservative(self):
        self.assertEqual(conformal_p([0] * 100, 0), 1.0)
        self.assertAlmostEqual(conformal_p([0] * 100, 1), 1 / 101)

    def test_wilson_interval_contains_empirical_rate(self):
        low, high = wilson_interval(10, 100)
        self.assertLessEqual(low, 0.1)
        self.assertGreaterEqual(high, 0.1)


if __name__ == "__main__":
    unittest.main()
