from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SBA = ROOT / "5g"
if str(SBA) not in sys.path:
    sys.path.insert(0, str(SBA))

from sba_semantic_v6 import (  # noqa: E402
    EvidenceOrigin,
    EvidenceState,
    LegitimateDisturbance,
    RobustDomainModel,
    SemanticQuorumDetector,
    SemanticSbaSimulator,
    TransportMode,
    TrustDomain,
)


TRANSPORT_DOMAINS = (
    TrustDomain.CONSUMER,
    TrustDomain.SCP,
    TrustDomain.PRODUCER,
    TrustDomain.NWDAF,
)


class SemanticV6Tests(unittest.TestCase):
    def _fit(self, seed=61, target_fpr=0.05):
        sim = SemanticSbaSimulator(seed=seed)
        contexts = list(sim.CONTROL_CONTEXTS)
        calibration = [
            sim.generate_window(i, context=contexts[i % len(contexts)])
            for i in range(120)
        ]
        det = SemanticQuorumDetector(
            target_fpr,
            transport_mode=TransportMode.BYZ_3OF4,
        ).fit(calibration)
        return sim, det

    def _force_unit_transport_model(self, det, context="normal_indirect"):
        models = {d: RobustDomainModel(0.0, 1.0) for d in TRANSPORT_DOMAINS}
        det.global_robust_models = dict(models)
        det.robust_models = {context: dict(models)}
        det.global_robust_threshold = 3.0
        det.robust_thresholds = {context: 3.0}

    def _set_transport(self, window, values):
        for domain, value in zip(TRANSPORT_DOMAINS, values):
            window.transport_residuals[domain] = float(value)
            window.transport_observed_at[domain] = window.window_end + 0.1

    def test_candidate_is_q3of5_without_global_persistence(self):
        det = SemanticQuorumDetector(0.05)
        self.assertEqual(det.profile.name, "q3of5")
        self.assertEqual(det.profile.q, 3)
        self.assertEqual(det.profile.n, 5)
        self.assertFalse(det.enable_soft_persistence)

    def test_legitimate_disturbance_is_attack_label_free(self):
        sim = SemanticSbaSimulator(seed=62)
        w = sim.generate_window(
            1000,
            disturbance=LegitimateDisturbance.TOKEN_REFRESH_RACE,
            disturbance_events=1,
        )
        self.assertEqual(w.label, 0)
        self.assertFalse(w.attack_event_ids)
        self.assertEqual(len(w.disturbance_event_ids), 1)

        disturbed_id = next(iter(w.disturbance_event_ids))
        inconsistent = [
            o
            for o in w.observations
            if o.event_id == disturbed_id
            and o.state == EvidenceState.INCONSISTENT
        ]
        self.assertGreaterEqual(len(inconsistent), 2)

    def test_disturbance_rng_does_not_change_transport_world(self):
        plain = SemanticSbaSimulator(seed=63)
        disturbed = SemanticSbaSimulator(seed=63)
        wa = plain.generate_window(0)
        wb = disturbed.generate_window(
            0,
            disturbance=LegitimateDisturbance.SCP_BENIGN_REROUTE,
        )
        self.assertEqual(set(wa.transport_residuals), set(wb.transport_residuals))
        for domain in wa.transport_residuals:
            a = wa.transport_residuals[domain]
            b = wb.transport_residuals[domain]
            if math.isnan(a) and math.isnan(b):
                continue
            self.assertEqual(a, b)

    def test_one_arbitrary_high_transport_report_cannot_alert(self):
        sim, det = self._fit(64)
        w = sim.generate_window(2000)
        self._force_unit_transport_model(det, w.context)
        self._set_transport(w, [100.0, 0.0, 0.0, 0.0])
        self.assertEqual(det.robust_transport_score(w), 0.0)
        self.assertFalse(det.score(w).transport_alert)

    def test_two_high_transport_reports_still_cannot_alert_3of4(self):
        sim, det = self._fit(65)
        w = sim.generate_window(2001)
        self._force_unit_transport_model(det, w.context)
        self._set_transport(w, [100.0, 100.0, 0.0, 0.0])
        self.assertEqual(det.robust_transport_score(w), 0.0)
        self.assertFalse(det.score(w).transport_alert)

    def test_three_high_transport_reports_do_alert_3of4(self):
        sim, det = self._fit(66)
        w = sim.generate_window(2002)
        self._force_unit_transport_model(det, w.context)
        self._set_transport(w, [100.0, 100.0, 100.0, 0.0])
        self.assertEqual(det.robust_transport_score(w), 100.0)
        self.assertTrue(det.score(w).transport_alert)

    def test_adversarial_calibration_score_is_conservative(self):
        sim, det = self._fit(67)
        w = sim.generate_window(2003)
        self._force_unit_transport_model(det, w.context)
        self._set_transport(w, [4.0, 3.0, 2.0, 1.0])
        self.assertEqual(det.robust_transport_score(w), 2.0)
        self.assertEqual(det.adversarial_calibration_score(w), 3.0)
        self.assertGreaterEqual(
            det.adversarial_calibration_score(w),
            det.robust_transport_score(w),
        )

    def test_legacy_transport_ablation_remains_available(self):
        sim = SemanticSbaSimulator(seed=68)
        contexts = list(sim.CONTROL_CONTEXTS)
        calibration = [
            sim.generate_window(i, context=contexts[i % len(contexts)])
            for i in range(90)
        ]
        legacy = SemanticQuorumDetector(
            0.05,
            transport_mode=TransportMode.LEGACY_MEDIAN,
        ).fit(calibration)
        robust = SemanticQuorumDetector(
            0.05,
            transport_mode=TransportMode.BYZ_3OF4,
        ).fit(calibration)
        w = sim.generate_window(3000)
        self.assertIsInstance(legacy.score(w).transport_alert, bool)
        self.assertIsInstance(robust.score(w).transport_alert, bool)


if __name__ == "__main__":
    unittest.main()
