from __future__ import annotations

import unittest
from pathlib import Path

from scientific.contract import Capability, CompromiseState, elapsed_since

ROOT = Path(__file__).resolve().parents[1]


class CompromiseContractTests(unittest.TestCase):
    def test_elapsed_since_zero(self):
        self.assertEqual(elapsed_since(0, 0), 0)
        self.assertEqual(elapsed_since(0, 1), 1)
        self.assertEqual(elapsed_since(0, 100), 100)

    def test_elapsed_none(self):
        self.assertEqual(elapsed_since(None, 100), 0)

    def test_compromise_is_capability(self):
        state = CompromiseState(
            compromised_since=0,
            capabilities=frozenset({Capability.FORGE_LOCAL_REPORT}),
        )
        self.assertTrue(state.compromised)
        self.assertTrue(state.can(Capability.FORGE_LOCAL_REPORT))
        self.assertEqual(state.elapsed(20), 20)


class SourceContractTests(unittest.TestCase):
    def test_ss7_canonical_entry(self):
        text = (ROOT / "ss7" / "canonical.py").read_text(encoding="utf-8")
        self.assertIn("build_dataset", text)
        self.assertIn(".fix", text)

    def test_diameter_no_direct_fingerprints(self):
        text = (ROOT / "diameter" / "diameter_sim_v1.py").read_text(encoding="utf-8")
        for needle in [
            "processing_delay_ms += self.rng.uniform(0.5, 3.0)",
            "fail_prob = 0.30",
            "+ drift + 0.8",
            "(node.compromised_since or interval_idx)",
        ]:
            self.assertNotIn(needle, text)
        self.assertIn("def _nominal_counterfactual_report", text)

    def test_sip_keeps_ack_invariant(self):
        text = (ROOT / "sip" / "sip_sim_v4_claude.py").read_text(encoding="utf-8")
        self.assertIn("n_ack_transit", text)
        self.assertIn("cu_ack_closure_mismatch", text)
        self.assertNotIn("phys_leak = {", text)
        self.assertNotIn("(node.compromised_since or interval_idx)", text)

    def test_legacy_ss7_t0_bug_removed(self):
        text = (ROOT / "ss7" / "ss7_simulator_v7.py").read_text(encoding="utf-8")
        self.assertNotIn("(node.compromised_since or interval_idx)", text)

    def test_baseline_utf8(self):
        text = (ROOT / "experiments" / "baseline.py").read_text(encoding="utf-8")
        self.assertIn('PYTHONIOENCODING", "utf-8"', text)
        self.assertIn('encoding="utf-8"', text)


if __name__ == "__main__":
    unittest.main()
