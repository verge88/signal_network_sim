from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SBA = ROOT / "5g"
if str(SBA) not in sys.path:
    sys.path.insert(0, str(SBA))

from sba_semantic_v4 import (  # noqa: E402
    AttackFamily,
    CompromiseMode,
    EvidenceState,
    SemanticFact,
    SemanticQuorumDetector,
    SemanticSbaSimulator,
    Sophistication,
    TrustDomain,
    empirical_auc,
)


class SemanticV4Tests(unittest.TestCase):
    def _fit(self, seed=41):
        sim = SemanticSbaSimulator(seed=seed)
        contexts = list(sim.CONTROL_CONTEXTS)
        calib = [
            sim.generate_window(i, context=contexts[i % len(contexts)])
            for i in range(120)
        ]
        det = SemanticQuorumDetector(0.05).fit(calib)
        return sim, det

    def test_evidence_has_unknown_state_and_latency(self):
        sim, _ = self._fit(42)
        windows = [sim.generate_window(1000 + i) for i in range(20)]
        observations = [o for w in windows for o in w.observations]
        self.assertTrue(any(o.state == EvidenceState.UNKNOWN for o in observations))
        for w in windows:
            for o in w.observations:
                event = next(e for e in w.events if e.event_id == o.event_id)
                self.assertGreaterEqual(o.observed_at, event.timestamp)

    def test_online_quorum_alert_time_is_after_attack_event(self):
        sim, det = self._fit(43)
        w = sim.generate_window(
            2000,
            attack_family=AttackFamily.SCP_BYPASS,
            hidden_calls=20,
            sophistication=Sophistication.NAIVE,
        )
        r = det.score(w)
        attack_ids = w.attack_event_ids
        detected = attack_ids & set(r.fired_event_ids)
        if detected:
            first_attack = min(e.timestamp for e in w.events if e.event_id in attack_ids)
            first_detection = min(r.event_alert_times[eid] for eid in detected)
            self.assertGreaterEqual(first_detection, first_attack)
            self.assertGreater(first_detection - first_attack, 0.0)

    def test_until_parameter_prevents_future_evidence_leakage(self):
        sim, det = self._fit(44)
        w = sim.generate_window(
            3000,
            attack_family=AttackFamily.COARSE_SCOPE,
            hidden_calls=10,
            sophistication=Sophistication.NAIVE,
        )
        first_attack = min(e.timestamp for e in w.events if e.event_id in w.attack_event_ids)
        early = det.score(w, until=first_attack)
        full = det.score(w)
        self.assertLessEqual(len(early.fired_event_ids), len(full.fired_event_ids))

    def test_adaptive_is_not_easier_than_naive_on_average(self):
        naive = SemanticSbaSimulator(seed=45)
        adaptive = SemanticSbaSimulator(seed=45)
        contexts = list(naive.CONTROL_CONTEXTS)
        calib_n = [naive.generate_window(i, context=contexts[i % len(contexts)]) for i in range(120)]
        calib_a = [adaptive.generate_window(i, context=contexts[i % len(contexts)]) for i in range(120)]
        det_n = SemanticQuorumDetector(0.05).fit(calib_n)
        det_a = SemanticQuorumDetector(0.05).fit(calib_a)

        naive_detected = 0
        adaptive_detected = 0
        n = 80
        for i in range(n):
            wn = naive.generate_window(
                4000 + i,
                attack_family=AttackFamily.DISCOVERY_POISON,
                hidden_calls=1,
                sophistication=Sophistication.NAIVE,
            )
            wa = adaptive.generate_window(
                4000 + i,
                attack_family=AttackFamily.DISCOVERY_POISON,
                hidden_calls=1,
                sophistication=Sophistication.ADAPTIVE,
            )
            naive_detected += int(det_n.score(wn).alert)
            adaptive_detected += int(det_a.score(wa).alert)
        self.assertGreaterEqual(naive_detected, adaptive_detected)

    def test_passive_compromise_does_not_create_fingerprint(self):
        a = SemanticSbaSimulator(seed=46)
        b = SemanticSbaSimulator(seed=46)
        wa = a.generate_window(0)
        wb = b.generate_window(
            0,
            compromised_domains=[TrustDomain.SCP],
            compromise_mode=CompromiseMode.PASSIVE,
        )
        self.assertEqual(wa.transport_residuals, wb.transport_residuals)
        xa = [(o.event_id, o.fact, o.domain, o.state, o.observed_at) for o in wa.observations]
        xb = [(o.event_id, o.fact, o.domain, o.state, o.observed_at) for o in wb.observations]
        self.assertEqual(xa, xb)

    def test_direct_allowed_is_not_route_attack(self):
        sim, det = self._fit(47)
        route_alerts = 0
        for i in range(40):
            w = sim.generate_window(5000 + i, context="direct_allowed")
            route_alerts += int(SemanticFact.ROUTE in det.score(w).fired_facts)
        self.assertLessEqual(route_alerts, 2)

    def test_suppressed_scp_reduces_route_evidence_but_not_all_evidence(self):
        plain, det_plain = self._fit(48)
        suppressed, det_suppressed = self._fit(48)
        plain_hits = 0
        suppressed_hits = 0
        n = 60
        for i in range(n):
            wp = plain.generate_window(
                6000 + i,
                attack_family=AttackFamily.SCP_BYPASS,
                hidden_calls=1,
                sophistication=Sophistication.ADAPTIVE,
            )
            ws = suppressed.generate_window(
                6000 + i,
                attack_family=AttackFamily.SCP_BYPASS,
                hidden_calls=1,
                sophistication=Sophistication.ADAPTIVE,
                compromised_domains=[TrustDomain.SCP],
                compromise_mode=CompromiseMode.SUPPRESS,
            )
            plain_hits += int(det_plain.score(wp).alert)
            suppressed_hits += int(det_suppressed.score(ws).alert)
        self.assertGreaterEqual(plain_hits, suppressed_hits)

    def test_null_world_auc_is_near_random(self):
        sim, det = self._fit(49)
        windows = [sim.generate_window(7000 + i) for i in range(140)]
        labels = [i % 2 for i in range(len(windows))]
        auc = empirical_auc(labels, [det.score(w).score for w in windows])
        self.assertGreater(auc, 0.30)
        self.assertLess(auc, 0.70)


if __name__ == "__main__":
    unittest.main()
