from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SBA = ROOT / "5g"
if str(SBA) not in sys.path:
    sys.path.insert(0, str(SBA))

from sba_semantic_v3 import (  # noqa: E402
    AttackFamily,
    CommunicationMode,
    CompromiseMode,
    SemanticFact,
    SemanticQuorumDetector,
    SemanticSbaSimulator,
    TrustDomain,
    empirical_auc,
)
from run_semantic_v3 import _event_metrics, wilson_interval  # noqa: E402


class SemanticV3Tests(unittest.TestCase):
    def _fit(self, seed=7):
        sim = SemanticSbaSimulator(seed=seed)
        contexts = list(sim.CONTROL_CONTEXTS)
        calib = [
            sim.generate_window(i, context=contexts[i % len(contexts)])
            for i in range(72)
        ]
        det = SemanticQuorumDetector(0.05).fit(calib)
        return sim, det

    def test_direct_allowed_is_not_route_violation(self):
        sim, det = self._fit(10)
        w = sim.generate_window(1000, context="direct_allowed")
        r = det.score(w)
        self.assertNotIn(SemanticFact.ROUTE, r.fired_facts)
        self.assertEqual(w.communication_mode, CommunicationMode.DIRECT_ALLOWED)

    def test_scp_bypass_fires_route_quorum(self):
        sim, det = self._fit(11)
        w = sim.generate_window(
            1001,
            attack_family=AttackFamily.SCP_BYPASS,
            hidden_calls=20,
            adaptive_budget=1.0,
        )
        r = det.score(w)
        self.assertTrue(r.alert)
        self.assertIn(SemanticFact.ROUTE, r.fired_facts)

    def test_one_suppressed_domain_does_not_defeat_route_quorum(self):
        sim, det = self._fit(12)
        w = sim.generate_window(
            1002,
            attack_family=AttackFamily.SCP_BYPASS,
            hidden_calls=20,
            compromised_domains=[TrustDomain.SCP],
            compromise_mode=CompromiseMode.SUPPRESS,
        )
        r = det.score(w)
        self.assertIn(SemanticFact.ROUTE, r.fired_facts)

    def test_no_token_uses_semantic_token_gate(self):
        sim, det = self._fit(13)
        w = sim.generate_window(
            1003,
            attack_family=AttackFamily.NO_TOKEN,
            hidden_calls=12,
        )
        r = det.score(w)
        self.assertIn(SemanticFact.TOKEN, r.fired_facts)

    def test_coarse_scope_fires_slice_gate(self):
        sim, det = self._fit(14)
        w = sim.generate_window(
            1004,
            attack_family=AttackFamily.COARSE_SCOPE,
            hidden_calls=12,
        )
        r = det.score(w)
        self.assertIn(SemanticFact.SLICE, r.fired_facts)

    def test_passive_compromise_is_not_an_observable_fingerprint(self):
        a = SemanticSbaSimulator(seed=123)
        b = SemanticSbaSimulator(seed=123)
        wa = a.generate_window(0)
        wb = b.generate_window(
            0,
            compromised_domains=[TrustDomain.SCP],
            compromise_mode=CompromiseMode.PASSIVE,
        )
        self.assertEqual(wa.transport_residuals, wb.transport_residuals)
        xa = [(o.event_id, o.fact, o.domain, o.inconsistent) for o in wa.observations]
        xb = [(o.event_id, o.fact, o.domain, o.inconsistent) for o in wb.observations]
        self.assertEqual(xa, xb)

    def test_null_world_auc_is_near_random(self):
        sim, det = self._fit(15)
        windows = [sim.generate_window(2000 + i) for i in range(120)]
        labels = [i % 2 for i in range(len(windows))]
        auc = empirical_auc(labels, [det.score(w).score for w in windows])
        self.assertGreater(auc, 0.30)
        self.assertLess(auc, 0.70)

    def test_event_metrics_use_only_appended_attack_events(self):
        sim, det = self._fit(16)
        w = sim.generate_window(
            3000,
            attack_family=AttackFamily.NO_TOKEN,
            hidden_calls=3,
        )
        metrics = _event_metrics(w, det.score(w))
        self.assertEqual(metrics["hidden_events"], 3.0)
        self.assertGreaterEqual(metrics["detected_attack_events"], 0.0)
        self.assertLessEqual(metrics["detected_attack_events"], 3.0)
        self.assertGreaterEqual(metrics["event_recall"], 0.0)
        self.assertLessEqual(metrics["event_recall"], 1.0)
        if math.isfinite(metrics["ttd_s"]):
            self.assertGreaterEqual(metrics["ttd_s"], 0.0)

    def test_wilson_interval_contains_target_for_observed_medium_run(self):
        lo, hi = wilson_interval(10, 1200)
        self.assertLess(lo, 0.01)
        self.assertGreater(hi, 0.01)
        self.assertAlmostEqual(10 / 1200, 0.008333333333333333)


if __name__ == "__main__":
    unittest.main()
