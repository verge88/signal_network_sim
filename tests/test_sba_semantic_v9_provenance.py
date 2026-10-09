"""Regression tests for observational provenance in operational transients.

These tests are intentionally independent of the detector's statistical
performance: no UNKNOWN evidence may be invented, and only mutated reports
may receive synthetic recovery observations.
"""
from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SBA = ROOT / "5g"
if str(SBA) not in sys.path:
    sys.path.insert(0, str(SBA))

from sba_semantic_v6 import (  # noqa: E402
    DISTURBANCE_SPECS,
    SemanticSbaSimulator as V6Simulator,
)
from sba_semantic_v8 import (  # noqa: E402
    EvidenceState,
    OperationalContextConfig,
    OperationalState,
    STATE_SPECS,
    SemanticSbaSimulator,
)


class ProvenanceRegressionTests(unittest.TestCase):
    def _setup(self, seed=901):
        cfg = OperationalContextConfig(
            disturbance_manifest_probability=1.0,
            recovery_observation_probability=1.0,
            marker_recall=0.0,
            false_marker_probability=0.0,
        )
        sim = SemanticSbaSimulator(seed=seed, operational_cfg=cfg)
        base = V6Simulator(seed=seed).generate_window(1000)
        event = base.events[0]
        spec = STATE_SPECS[OperationalState.TOKEN_REFRESH]
        return sim, base, event, spec

    def test_all_unknown_stays_unknown_and_no_recovery(self):
        sim, base, event, spec = self._setup()
        base.observations = [
            replace(o, state=EvidenceState.UNKNOWN)
            if o.event_id == event.event_id and o.fact == spec.fact else o
            for o in base.observations
        ]
        before = list(base.observations)
        after, ids, changed = sim._manifest_transient(base, event, spec)
        self.assertEqual(after, before)
        self.assertFalse(ids)
        self.assertEqual(changed, ())
        recovered, emitted = sim._add_recovery(after, event, spec, changed)
        self.assertFalse(emitted)
        self.assertEqual(recovered, before)

    def test_recovery_only_follows_exact_new_mutations(self):
        sim, base, event, spec = self._setup(902)
        primary = set(DISTURBANCE_SPECS[spec.disturbance].primary_origins)
        base.observations = [
            replace(
                o,
                state=(
                    EvidenceState.CONSISTENT
                    if o.origin in primary else EvidenceState.INCONSISTENT
                ),
            )
            if o.event_id == event.event_id and o.fact == spec.fact else o
            for o in base.observations
        ]
        expected = [
            o for o in base.observations
            if o.event_id == event.event_id
            and o.fact == spec.fact
            and o.origin in primary
        ]
        preexisting = [
            o for o in base.observations
            if o.event_id == event.event_id
            and o.fact == spec.fact
            and o.origin not in primary
        ]
        self.assertTrue(expected)
        self.assertTrue(preexisting)
        after, ids, changed = sim._manifest_transient(base, event, spec)
        self.assertEqual(ids, frozenset([event.event_id]))
        self.assertEqual(len(changed), len(expected))
        self.assertTrue(all(after[i].state == EvidenceState.INCONSISTENT for i in changed))
        recovered, emitted = sim._add_recovery(after, event, spec, changed)
        self.assertTrue(emitted)
        added = recovered[len(after):]
        self.assertEqual(len(added), len(expected))
        self.assertTrue(all(o.origin in primary for o in added))
        self.assertTrue(all(o.state == EvidenceState.CONSISTENT for o in added))
        self.assertFalse(any(o.origin not in primary for o in added))

    def test_unknown_primary_is_never_promoted(self):
        sim, base, event, spec = self._setup(903)
        primary = set(DISTURBANCE_SPECS[spec.disturbance].primary_origins)
        base.observations = [
            replace(o, state=EvidenceState.UNKNOWN)
            if o.event_id == event.event_id and o.fact == spec.fact
            and o.origin in primary else o
            for o in base.observations
        ]
        before = list(base.observations)
        after, _ids, changed = sim._manifest_transient(base, event, spec)
        for original, current in zip(before, after):
            if original.state == EvidenceState.UNKNOWN:
                self.assertEqual(current.state, EvidenceState.UNKNOWN)
        self.assertTrue(all(before[i].state == EvidenceState.CONSISTENT for i in changed))


if __name__ == "__main__":
    unittest.main()
