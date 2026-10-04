from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SBA = ROOT / "5g"
if str(SBA) not in sys.path:
    sys.path.insert(0, str(SBA))

from sba_semantic_v6 import SemanticQuorumDetector as V6Detector  # noqa: E402
from sba_semantic_v8 import (  # noqa: E402
    AttackFamily,
    EvidenceState,
    OperationalContextConfig,
    OperationalMarker,
    OperationalState,
    STATE_SPECS,
    SemanticQuorumDetector,
    SemanticSbaSimulator,
    Sophistication,
    TransportMode,
)


class SemanticV8Tests(unittest.TestCase):
    def _fit_pair(self, seed=81, cfg=None):
        sim = SemanticSbaSimulator(seed=seed, operational_cfg=cfg)
        contexts = list(sim.CONTROL_CONTEXTS)
        calibration = [
            sim.generate_window(
                i,
                context=contexts[i % len(contexts)],
                operational_state=OperationalState.NORMAL,
                false_marker=False,
            )
            for i in range(120)
        ]
        static = V6Detector(
            0.05,
            transport_mode=TransportMode.BYZ_3OF4,
        ).fit(calibration)
        noisy = SemanticQuorumDetector(0.05).fit(calibration)
        return sim, static, noisy

    def test_marker_can_be_missing_while_transient_exists(self):
        cfg = OperationalContextConfig(
            marker_recall=0.0,
            false_marker_probability=0.0,
            recovery_observation_probability=0.0,
            disturbance_manifest_probability=1.0,
        )
        sim = SemanticSbaSimulator(seed=82, operational_cfg=cfg)
        w = sim.generate_window(
            1000,
            operational_state=OperationalState.TOKEN_REFRESH,
            false_marker=False,
        )
        self.assertTrue(w.marker_expected)
        self.assertFalse(w.marker_emitted)
        self.assertTrue(w.transient_manifested)
        self.assertFalse(w.recovery_emitted)

    def test_false_marker_exists_without_operational_state(self):
        cfg = OperationalContextConfig(
            marker_recall=0.0,
            false_marker_probability=1.0,
        )
        sim = SemanticSbaSimulator(seed=83, operational_cfg=cfg)
        w = sim.generate_window(
            1001,
            operational_state=OperationalState.NORMAL,
            false_marker=True,
        )
        self.assertFalse(w.marker_expected)
        self.assertTrue(w.marker_emitted)
        self.assertTrue(any(m.is_false_marker for m in w.operational_markers))
        self.assertEqual(w.label, 0)
        self.assertFalse(w.attack_event_ids)

    def test_marker_correlation_can_be_wrong(self):
        cfg = OperationalContextConfig(
            marker_recall=1.0,
            false_marker_probability=0.0,
            correlation_error_probability=1.0,
            recovery_observation_probability=0.0,
            disturbance_manifest_probability=1.0,
        )
        sim = SemanticSbaSimulator(seed=84, operational_cfg=cfg)
        w = sim.generate_window(
            1002,
            operational_state=OperationalState.POLICY_SYNC,
            false_marker=False,
        )
        self.assertTrue(w.marker_emitted)
        self.assertFalse(w.marker_correct)
        marker = w.operational_markers[0]
        event_id = next(iter(w.operational_event_ids))
        event = next(e for e in w.events if e.event_id == event_id)
        self.assertNotEqual(marker.correlation_id, event.correlation_id)

    def test_recovery_is_not_forced_before_grace(self):
        cfg = OperationalContextConfig(
            marker_recall=1.0,
            false_marker_probability=0.0,
            recovery_observation_probability=1.0,
            recovery_latency_sigma=1e-6,
            recovery_latency_multiplier=20.0,
            disturbance_manifest_probability=1.0,
        )
        sim = SemanticSbaSimulator(seed=85, operational_cfg=cfg)
        w = sim.generate_window(
            1003,
            operational_state=OperationalState.TOKEN_REFRESH,
            false_marker=False,
        )
        self.assertTrue(w.recovery_emitted)
        event_id = next(iter(w.recovery_event_ids))
        event = next(e for e in w.events if e.event_id == event_id)
        spec = STATE_SPECS[OperationalState.TOKEN_REFRESH]
        inconsistent = [
            o for o in w.observations
            if o.event_id == event_id
            and o.fact == spec.fact
            and o.state == EvidenceState.INCONSISTENT
        ]
        last_bad = max(o.observed_at for o in inconsistent)
        recovered = [
            o for o in w.observations
            if o.event_id == event_id
            and o.fact == spec.fact
            and o.state == EvidenceState.CONSISTENT
            and o.observed_at > last_bad
        ]
        self.assertTrue(recovered)
        self.assertGreater(
            max(o.observed_at for o in recovered),
            event.timestamp + spec.grace_s,
        )

    def test_late_marker_cannot_retroactively_suppress_quorum(self):
        cfg = OperationalContextConfig(false_marker_probability=0.0)
        sim, _static, det = self._fit_pair(86, cfg)
        w = sim.generate_window(2000, false_marker=False)
        event = w.events[0]
        fact = STATE_SPECS[OperationalState.TOKEN_REFRESH].fact
        selected = list(det.selected_origins(fact))[:3]
        observations = list(w.observations)
        changed = 0
        max_bad = event.timestamp
        for i, obs in enumerate(observations):
            if obs.event_id == event.event_id and obs.fact == fact and obs.origin in selected:
                observations[i] = replace(
                    obs,
                    state=EvidenceState.INCONSISTENT,
                    observed_at=event.timestamp + 0.05 + 0.01 * changed,
                )
                max_bad = max(max_bad, observations[i].observed_at)
                changed += 1
                if changed == 3:
                    break
        self.assertEqual(changed, 3)
        marker = OperationalMarker(
            state=OperationalState.TOKEN_REFRESH,
            fact=fact,
            correlation_id=event.correlation_id,
            valid_from=event.timestamp - 0.1,
            expires_at=event.timestamp + 2.0,
            observed_at=max_bad + 0.5,
            source="test-late",
        )
        w.observations = observations
        w.operational_markers = (marker,)
        result = det.score(w)
        self.assertTrue(result.semantic_alert)
        self.assertIn(event.event_id, result.fired_event_ids)

    def test_early_marker_plus_recovery_resolves_quorum(self):
        cfg = OperationalContextConfig(false_marker_probability=0.0)
        sim, _static, det = self._fit_pair(87, cfg)
        w = sim.generate_window(2001, false_marker=False)
        event = w.events[0]
        fact = STATE_SPECS[OperationalState.TOKEN_REFRESH].fact
        selected = list(det.selected_origins(fact))[:3]
        observations = list(w.observations)
        changed_indices = []
        for i, obs in enumerate(observations):
            if obs.event_id == event.event_id and obs.fact == fact and obs.origin in selected:
                observations[i] = replace(
                    obs,
                    state=EvidenceState.INCONSISTENT,
                    observed_at=event.timestamp + 0.30 + 0.01 * len(changed_indices),
                )
                changed_indices.append(i)
                if len(changed_indices) == 3:
                    break
        self.assertEqual(len(changed_indices), 3)
        for j, idx in enumerate(changed_indices):
            obs = observations[idx]
            observations.append(
                replace(
                    obs,
                    state=EvidenceState.CONSISTENT,
                    observed_at=event.timestamp + 0.8 + 0.01 * j,
                )
            )
        marker = OperationalMarker(
            state=OperationalState.TOKEN_REFRESH,
            fact=fact,
            correlation_id=event.correlation_id,
            valid_from=event.timestamp - 0.1,
            expires_at=event.timestamp + 2.0,
            observed_at=event.timestamp + 0.1,
            source="test-early",
        )
        w.observations = observations
        w.operational_markers = (marker,)
        result = det.score(w)
        self.assertFalse(result.semantic_alert)
        self.assertGreaterEqual(result.grace_deferred_keys, 1)
        self.assertGreaterEqual(result.grace_resolved_keys, 1)

    def test_attack_during_real_state_is_not_permanently_whitelisted(self):
        cfg = OperationalContextConfig(
            marker_recall=1.0,
            false_marker_probability=0.0,
            correlation_error_probability=0.0,
            marker_latency_median_s=0.01,
            marker_latency_sigma=1e-6,
            recovery_observation_probability=0.0,
        )
        sim, static, noisy = self._fit_pair(88, cfg)
        static_hits = noisy_hits = 0
        expired = 0
        for i in range(40):
            w = sim.generate_attack_during_state(
                3000 + i,
                attack_family=AttackFamily.NO_TOKEN,
                hidden_calls=1,
                sophistication=Sophistication.ADAPTIVE,
            )
            rs = static.score(w)
            rn = noisy.score(w)
            static_hits += int(rs.alert)
            noisy_hits += int(rn.alert)
            expired += rn.grace_expired_alerts
        self.assertEqual(static_hits, noisy_hits)
        self.assertGreater(expired, 0)

    def test_detector_configuration_remains_frozen(self):
        det = SemanticQuorumDetector(0.01)
        self.assertEqual(det.profile.name, "q3of5")
        self.assertFalse(det.enable_soft_persistence)
        self.assertEqual(det.transport_mode, TransportMode.BYZ_3OF4)


if __name__ == "__main__":
    unittest.main()
