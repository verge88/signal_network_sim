from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SBA = ROOT / "5g"
if str(SBA) not in sys.path:
    sys.path.insert(0, str(SBA))

from run_semantic_v8 import (  # noqa: E402
    ARM_NOISY,
    ARM_STATIC,
    paired_statistics,
)


class SemanticV8StatisticsTests(unittest.TestCase):
    def test_paired_statistics_preserve_seed_pairing(self):
        summaries = []
        attacks = []
        operational = []
        state_attacks = []
        for seed in range(42, 47):
            for arm, shift in ((ARM_STATIC, 0.0), (ARM_NOISY, -0.02)):
                summaries.append({
                    "seed": seed,
                    "arm": arm,
                    "benign_fpr": 0.01,
                    "mean_attack_window_recall": 0.80,
                    "mean_attack_event_recall": 0.30,
                    "mean_operational_fpr": 0.08 + shift,
                    "worst_operational_fpr": 0.10 + shift,
                    "mean_semantic_operational_fpr": 0.06 + shift,
                    "state_attack_window_recall": 0.40,
                    "state_attack_event_recall": 0.35,
                    "state_attack_event_latency_s": 1.0 + (0.5 if arm == ARM_NOISY else 0.0),
                })
                attacks.append({
                    "seed": seed,
                    "arm": arm,
                    "family": "no_token",
                    "window_rate": 0.8,
                    "event_recall": 0.3,
                })
                operational.append({
                    "seed": seed,
                    "arm": arm,
                    "state": "token_refresh",
                    "operational_fpr": 0.08 + shift,
                    "semantic_operational_fpr": 0.06 + shift,
                })
                state_attacks.append({
                    "seed": seed,
                    "arm": arm,
                    "family": "no_token",
                    "window_rate": 0.4,
                    "event_recall": 0.35,
                    "event_latency_median_s": 1.0 + (0.5 if arm == ARM_NOISY else 0.0),
                })

        stats = paired_statistics(
            pd.DataFrame(summaries),
            pd.DataFrame(attacks),
            pd.DataFrame(operational),
            pd.DataFrame(state_attacks),
            seed=42,
            n_boot=500,
            n_perm=500,
        )
        row = stats[
            (stats["scope"] == "overall")
            & (stats["metric"] == "mean_operational_fpr")
        ].iloc[0]
        self.assertEqual(int(row["n_seeds"]), 5)
        self.assertAlmostEqual(float(row["delta_v8_minus_static"]), -0.02, places=12)
        self.assertLess(float(row["delta_ci_hi"]), 0.0)

    def test_holm_column_is_bounded_when_pvalues_are_finite(self):
        summaries = []
        attacks = []
        operational = []
        state_attacks = []
        for seed in (1, 2, 3):
            for arm in (ARM_STATIC, ARM_NOISY):
                candidate = arm == ARM_NOISY
                summaries.append({
                    "seed": seed,
                    "arm": arm,
                    "benign_fpr": 0.01,
                    "mean_attack_window_recall": 0.8,
                    "mean_attack_event_recall": 0.3,
                    "mean_operational_fpr": 0.02 if candidate else 0.08,
                    "worst_operational_fpr": 0.03 if candidate else 0.10,
                    "mean_semantic_operational_fpr": 0.01 if candidate else 0.06,
                    "state_attack_window_recall": 0.4,
                    "state_attack_event_recall": 0.35,
                    "state_attack_event_latency_s": 2.0 if candidate else 1.0,
                })
                attacks.append({"seed": seed, "arm": arm, "family": "x", "window_rate": 0.8, "event_recall": 0.3})
                operational.append({"seed": seed, "arm": arm, "state": "x", "operational_fpr": 0.02 if candidate else 0.08, "semantic_operational_fpr": 0.01 if candidate else 0.06})
                state_attacks.append({"seed": seed, "arm": arm, "family": "x", "window_rate": 0.4, "event_recall": 0.35, "event_latency_median_s": 2.0 if candidate else 1.0})
        stats = paired_statistics(
            pd.DataFrame(summaries), pd.DataFrame(attacks),
            pd.DataFrame(operational), pd.DataFrame(state_attacks),
            seed=7, n_boot=200, n_perm=200,
        )
        finite = stats[stats["holm_p"].notna()]["holm_p"]
        self.assertTrue(len(finite) > 0)
        self.assertTrue(all(0.0 <= float(x) <= 1.0 for x in finite))


if __name__ == "__main__":
    unittest.main()
