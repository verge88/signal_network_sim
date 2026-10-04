from __future__ import annotations

import math
import sys
import unittest
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SBA = ROOT / "5g"
if str(SBA) not in sys.path:
    sys.path.insert(0, str(SBA))

from sba_semantic_v6 import (  # noqa: E402
    SemanticQuorumDetector as V6SemanticQuorumDetector,
)
from sba_semantic_v7 import (  # noqa: E402
    AttackFamily,
    EvidenceOrigin,
    EvidenceState,
    LegitimateDisturbance,
    OperationalMarker,
    OperationalState,
    SemanticFact,
    SemanticQuorumDetector,
    SemanticSbaSimulator,
    Sophistication,
    TransportMode,
)


class SemanticV7Tests(unittest.TestCase):
    def _calibration(self, seed=71, n=120):
        sim = SemanticSbaSimulator(seed=seed)
        contexts = list(sim.CONTROL_CONTEXTS)
        return [
            sim.generate_window(i, context=contexts[i % len(contexts)])
            for i in range(n)
        ]

    def _fit(self, seed=71):
        calibration = self._calibration(seed)
        det = SemanticQuorumDetector(0.05).fit(calibration)
        return calibration, det

    @staticmethod
    def _force_states(window, event_id, fact, origins, state, at_offset):
        origins = set(origins)
        by_id = {e.event_id: e for e in window.events}
        event = by_id[event_id]
        out = []
        touched = set()
        for obs in window.observations:
            if (
                obs.event_id == event_id
                and obs.fact == fact
                and obs.origin in origins
            ):
                out.append(
                    replace(
                        obs,
                        state=state,
                        observed_at=float(event.timestamp + at_offset),
                        confidence=0.99,
                    )
                )
                touched.add(obs.origin)
            else:
                out.append(obs)
        if touched != origins:
            raise AssertionError(f"missing origins: {origins - touched}")
        return out

    def test_candidate_configuration_remains_frozen(self):
        det = SemanticQuorumDetector(0.05)
        self.assertEqual(det.profile.name, "q3of5")
        self.assertFalse(det.enable_soft_persistence)
        self.assertEqual(det.transport_mode, TransportMode.BYZ_3OF4)
        self.assertTrue(det.enable_operational_grace)

    def test_disturbance_is_label_free_and_has_recovery_context(self):
        sim = SemanticSbaSimulator(seed=72)
        w = sim.generate_window(
            1000,
            disturbance=LegitimateDisturbance.TOKEN_REFRESH_RACE,
            disturbance_events=1,
        )
        self.assertEqual(w.label, 0)
        self.assertFalse(w.attack_event_ids)
        self.assertEqual(len(w.disturbance_event_ids), 1)
        self.assertEqual(len(w.operational_markers), 1)
        marker = w.operational_markers[0]
        self.assertEqual(marker.state, OperationalState.TOKEN_REFRESH)
        self.assertEqual(marker.fact, SemanticFact.TOKEN)

        event_id = next(iter(w.disturbance_event_ids))
        obs = [
            o
            for o in w.observations
            if o.event_id == event_id and o.fact == SemanticFact.TOKEN
        ]
        inconsistent_times = [o.observed_at for o in obs if o.state == EvidenceState.INCONSISTENT]
        consistent_times = [o.observed_at for o in obs if o.state == EvidenceState.CONSISTENT]
        self.assertTrue(inconsistent_times)
        self.assertTrue(any(t > min(inconsistent_times) for t in consistent_times))
        self.assertLess(max(consistent_times), marker.expires_at)

    def test_no_marker_world_matches_v6_semantic_and_transport_decision(self):
        calibration = self._calibration(73)
        baseline = V6SemanticQuorumDetector(
            0.05,
            transport_mode=TransportMode.BYZ_3OF4,
        ).fit(calibration)
        candidate = SemanticQuorumDetector(0.05).fit(calibration)

        sim = SemanticSbaSimulator(seed=7300)
        worlds = [sim.generate_window(2000 + i) for i in range(10)]
        worlds += [
            sim.generate_window(
                3000 + i,
                attack_family=AttackFamily.NO_TOKEN,
                hidden_calls=2,
                sophistication=Sophistication.ADAPTIVE,
            )
            for i in range(10)
        ]
        for w in worlds:
            rb = baseline.score(w)
            rc = candidate.score(w)
            self.assertEqual(rb.alert, rc.alert)
            self.assertEqual(rb.semantic_alert, rc.semantic_alert)
            self.assertEqual(rb.transport_alert, rc.transport_alert)
            self.assertEqual(set(rb.fired_event_ids), set(rc.fired_event_ids))

    def test_resolved_quorum_inside_matching_grace_is_suppressed(self):
        calibration, candidate = self._fit(74)
        baseline = V6SemanticQuorumDetector(
            0.05,
            transport_mode=TransportMode.BYZ_3OF4,
        ).fit(calibration)
        sim = SemanticSbaSimulator(seed=7400)
        w = sim.generate_window(4000)
        event = w.events[0]
        origins = (
            EvidenceOrigin.NRF,
            EvidenceOrigin.POLICY,
            EvidenceOrigin.CONSUMER,
        )
        observations = self._force_states(
            w,
            event.event_id,
            SemanticFact.TOKEN,
            origins,
            EvidenceState.INCONSISTENT,
            0.10,
        )
        # A later consistent report from the same independent roots resolves
        # the transient before grace expiry.
        for origin in origins:
            source = next(
                o
                for o in observations
                if o.event_id == event.event_id
                and o.fact == SemanticFact.TOKEN
                and o.origin == origin
            )
            observations.append(
                replace(
                    source,
                    state=EvidenceState.CONSISTENT,
                    observed_at=float(event.timestamp + 0.50),
                )
            )
        marker = OperationalMarker(
            OperationalState.TOKEN_REFRESH,
            SemanticFact.TOKEN,
            event.correlation_id,
            event.timestamp - 0.1,
            event.timestamp + 2.0,
        )
        w = replace(
            w,
            observations=observations,
            operational_markers=(marker,),
        )
        rb = baseline.score(w)
        rc = candidate.score(w)
        self.assertIn(event.event_id, rb.fired_event_ids)
        self.assertNotIn(event.event_id, rc.fired_event_ids)
        self.assertGreaterEqual(rc.grace_deferred_keys, 1)
        self.assertGreaterEqual(rc.grace_resolved_keys, 1)

    def test_unresolved_attack_during_grace_alerts_at_expiry(self):
        _, det = self._fit(75)
        sim = SemanticSbaSimulator(seed=7500)
        w = sim.generate_window(
            5000,
            attack_family=AttackFamily.NO_TOKEN,
            hidden_calls=1,
            sophistication=Sophistication.ADAPTIVE,
        )
        attack_id = next(iter(w.attack_event_ids))
        attack = next(e for e in w.events if e.event_id == attack_id)
        origins = (
            EvidenceOrigin.NRF,
            EvidenceOrigin.POLICY,
            EvidenceOrigin.CONSUMER,
        )
        observations = self._force_states(
            w,
            attack_id,
            SemanticFact.TOKEN,
            origins,
            EvidenceState.INCONSISTENT,
            0.10,
        )
        w = replace(w, observations=observations)
        w = sim.cover_attack_with_operational_marker(w, attack_event_id=attack_id)
        marker = next(
            m for m in w.operational_markers if m.correlation_id == attack.correlation_id
        )
        r = det.score(w)
        self.assertIn(attack_id, r.fired_event_ids)
        self.assertGreaterEqual(r.event_alert_times[attack_id], marker.expires_at)
        self.assertGreaterEqual(r.grace_expired_alerts, 1)

    def test_grace_is_scoped_to_matching_correlation(self):
        _, det = self._fit(76)
        sim = SemanticSbaSimulator(seed=7600)
        w = sim.generate_window(6000)
        protected = w.events[0]
        target = w.events[1]
        origins = (
            EvidenceOrigin.NRF,
            EvidenceOrigin.POLICY,
            EvidenceOrigin.CONSUMER,
        )
        observations = self._force_states(
            w,
            target.event_id,
            SemanticFact.TOKEN,
            origins,
            EvidenceState.INCONSISTENT,
            0.10,
        )
        marker = OperationalMarker(
            OperationalState.TOKEN_REFRESH,
            SemanticFact.TOKEN,
            protected.correlation_id,
            min(protected.timestamp, target.timestamp) - 0.1,
            max(protected.timestamp, target.timestamp) + 10.0,
        )
        w = replace(
            w,
            observations=observations,
            operational_markers=(marker,),
        )
        r = det.score(w)
        self.assertIn(target.event_id, r.fired_event_ids)
        self.assertLess(
            r.event_alert_times[target.event_id],
            marker.expires_at,
        )

    def test_until_does_not_use_future_recovery(self):
        _, det = self._fit(77)
        sim = SemanticSbaSimulator(seed=7700)
        w = sim.generate_window(
            7000,
            disturbance=LegitimateDisturbance.POLICY_PROPAGATION_DELAY,
            disturbance_events=1,
        )
        marker = w.operational_markers[0]
        # During grace there must be no premature alert due only to future
        # convergence evidence.
        early = det.score(w, until=marker.start_at + 0.2)
        self.assertFalse(early.semantic_alert)
        full = det.score(w)
        self.assertLessEqual(full.grace_expired_alerts, full.grace_deferred_keys)


if __name__ == "__main__":
    unittest.main()
