from __future__ import annotations

import copy
import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SBA = ROOT / "5g"
if str(SBA) not in sys.path:
    sys.path.insert(0, str(SBA))

from sba_semantic_v5 import (  # noqa: E402
    AttackFamily,
    CompromiseMode,
    EvidenceOrigin,
    EvidenceState,
    OriginObservation,
    QUORUM_PROFILES,
    SemanticFact,
    SemanticQuorumDetector,
    SemanticSbaSimulator,
    Sophistication,
    V5Config,
)


class SemanticV5Tests(unittest.TestCase):
    def _fit(self, profile="q3of5", seed=71, persistence=True):
        sim = SemanticSbaSimulator(seed=seed)
        contexts = list(sim.CONTROL_CONTEXTS)
        calibration = [
            sim.generate_window(i, context=contexts[i % len(contexts)])
            for i in range(120)
        ]
        detector = SemanticQuorumDetector(
            profile,
            target_fpr=0.05,
            enable_soft_persistence=persistence,
        ).fit(calibration)
        return sim, detector

    def test_profiles_are_exactly_2of3_3of5_4of7(self):
        self.assertEqual((QUORUM_PROFILES["q2of3"].q, QUORUM_PROFILES["q2of3"].n), (2, 3))
        self.assertEqual((QUORUM_PROFILES["q3of5"].q, QUORUM_PROFILES["q3of5"].n), (3, 5))
        self.assertEqual((QUORUM_PROFILES["q4of7"].q, QUORUM_PROFILES["q4of7"].n), (4, 7))

    def test_selected_origins_are_distinct(self):
        _, detector = self._fit("q4of7", 72)
        for fact in (
            SemanticFact.TOKEN,
            SemanticFact.NOTIFICATION,
            SemanticFact.SLICE,
            SemanticFact.DISCOVERY,
            SemanticFact.PROCEDURE,
            SemanticFact.ROUTE,
        ):
            origins = detector.selected_origins(fact)
            self.assertEqual(len(origins), 7)

    def test_duplicate_same_trust_root_does_not_create_quorum(self):
        sim, detector = self._fit("q2of3", 73, persistence=False)
        window = sim.generate_window(
            1000,
            attack_family=AttackFamily.NO_TOKEN,
            hidden_calls=1,
            sophistication=Sophistication.NAIVE,
        )
        attack_id = next(iter(window.attack_event_ids))
        selected = list(detector.selected_origins(SemanticFact.TOKEN))
        origin = selected[0]
        template = next(
            o
            for o in window.observations
            if o.event_id == attack_id
            and o.fact == SemanticFact.TOKEN
            and o.origin == origin
        )
        window.observations = [
            OriginObservation(
                event_id=attack_id,
                fact=SemanticFact.TOKEN,
                origin=origin,
                trust_root=origin.value,
                state=EvidenceState.INCONSISTENT,
                observed_at=template.observed_at + i * 0.001,
                confidence=1.0,
            )
            for i in range(5)
        ]
        # Remove transport influence for this structural test.
        window.transport_residuals = {k: float("nan") for k in window.transport_residuals}
        result = detector.score(window)
        self.assertFalse(result.semantic_alert)

    def test_two_distinct_roots_trigger_q2of3_hard_fact(self):
        sim, detector = self._fit("q2of3", 74, persistence=False)
        window = sim.generate_window(
            1001,
            attack_family=AttackFamily.NO_TOKEN,
            hidden_calls=1,
            sophistication=Sophistication.NAIVE,
        )
        attack_id = next(iter(window.attack_event_ids))
        origins = list(detector.selected_origins(SemanticFact.TOKEN))[:2]
        base_time = next(e.timestamp for e in window.events if e.event_id == attack_id)
        window.observations = [
            OriginObservation(
                attack_id,
                SemanticFact.TOKEN,
                origin,
                origin.value,
                EvidenceState.INCONSISTENT,
                base_time + 0.1 + i * 0.1,
                1.0,
            )
            for i, origin in enumerate(origins)
        ]
        window.transport_residuals = {k: float("nan") for k in window.transport_residuals}
        result = detector.score(window)
        self.assertTrue(result.semantic_alert)
        self.assertIn(attack_id, result.fired_event_ids)

    def test_soft_fact_needs_persistence(self):
        sim, detector = self._fit("q2of3", 75, persistence=True)
        window = sim.generate_window(1002)
        event = window.events[0]
        origins = list(detector.selected_origins(SemanticFact.PROCEDURE))[:2]
        window.observations = [
            OriginObservation(
                event.event_id,
                SemanticFact.PROCEDURE,
                origin,
                origin.value,
                EvidenceState.INCONSISTENT,
                event.timestamp + 0.1 + i * 0.1,
                1.0,
            )
            for i, origin in enumerate(origins)
        ]
        window.transport_residuals = {k: float("nan") for k in window.transport_residuals}
        self.assertFalse(detector.score(window).semantic_alert)

    def test_soft_persistence_fires_on_second_distinct_event(self):
        sim, detector = self._fit("q2of3", 76, persistence=True)
        window = sim.generate_window(1003)
        e1, e2 = sorted(window.events[:2], key=lambda e: e.timestamp)
        # Force the two events to be within the persistence horizon.
        e2.timestamp = e1.timestamp + 1.0
        origins = list(detector.selected_origins(SemanticFact.PROCEDURE))[:2]
        observations = []
        for j, event in enumerate((e1, e2)):
            for i, origin in enumerate(origins):
                observations.append(
                    OriginObservation(
                        event.event_id,
                        SemanticFact.PROCEDURE,
                        origin,
                        origin.value,
                        EvidenceState.INCONSISTENT,
                        event.timestamp + 0.1 + i * 0.1,
                        1.0,
                    )
                )
        window.observations = observations
        window.transport_residuals = {k: float("nan") for k in window.transport_residuals}
        result = detector.score(window)
        self.assertTrue(result.semantic_alert)
        self.assertEqual(set(result.fired_event_ids), {e1.event_id, e2.event_id})

    def test_single_injector_cannot_semantically_alert_without_honest_false_votes(self):
        cfg = V5Config(false_evidence_prob=0.0)
        sim = SemanticSbaSimulator(seed=77, v5_cfg=cfg)
        contexts = list(sim.CONTROL_CONTEXTS)
        calibration = [sim.generate_window(i, context=contexts[i % len(contexts)]) for i in range(120)]
        for profile in QUORUM_PROFILES:
            detector = SemanticQuorumDetector(profile, 0.05, enable_soft_persistence=True).fit(calibration)
            test_sim = SemanticSbaSimulator(seed=78, v5_cfg=cfg)
            for origin in EvidenceOrigin:
                window = test_sim.generate_window(
                    2000 + list(EvidenceOrigin).index(origin),
                    compromised_origins=[origin],
                    compromise_mode=CompromiseMode.INJECT,
                )
                result = detector.score(window)
                self.assertFalse(
                    result.semantic_alert,
                    msg=f"{profile} incorrectly accepted one injector {origin.value}",
                )

    def test_passive_origin_compromise_has_identical_semantic_evidence(self):
        a = SemanticSbaSimulator(seed=79)
        b = SemanticSbaSimulator(seed=79)
        wa = a.generate_window(0)
        wb = b.generate_window(
            0,
            compromised_origins=[EvidenceOrigin.PRODUCER],
            compromise_mode=CompromiseMode.PASSIVE,
        )
        xa = [(o.event_id, o.fact, o.origin, o.state, o.observed_at) for o in wa.observations]
        xb = [(o.event_id, o.fact, o.origin, o.state, o.observed_at) for o in wb.observations]
        self.assertEqual(xa, xb)


if __name__ == "__main__":
    unittest.main()
