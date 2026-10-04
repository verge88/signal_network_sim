from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SBA = ROOT / "5g"
if str(SBA) not in sys.path:
    sys.path.insert(0, str(SBA))

from run_semantic_v9_confirmatory import (  # noqa: E402
    ARM_NOISY,
    ARM_STATIC,
    DEFAULT_HOLDOUT_SEEDS,
    PRIMARY_ENDPOINTS,
    build_analysis_plan,
    classify_primary,
    primary_statistics,
    write_locked_plan,
)
from sba_semantic_v8 import OperationalContextConfig  # noqa: E402


class V9ConfirmatoryStatisticsTests(unittest.TestCase):
    def test_primary_family_is_fixed_to_four_endpoints(self):
        self.assertEqual(len(PRIMARY_ENDPOINTS), 4)
        self.assertEqual(
            [x.metric for x in PRIMARY_ENDPOINTS],
            [
                "mean_operational_fpr",
                "mean_attack_window_recall",
                "state_attack_window_recall",
                "state_attack_event_latency_s",
            ],
        )

    def test_default_confirmatory_seeds_are_fresh_holdout(self):
        self.assertEqual(DEFAULT_HOLDOUT_SEEDS, tuple(range(52, 62)))
        self.assertTrue(set(DEFAULT_HOLDOUT_SEEDS).isdisjoint(set(range(42, 52))))

    def test_analysis_plan_declares_primary_holm_family_only(self):
        plan = build_analysis_plan(
            seeds=DEFAULT_HOLDOUT_SEEDS,
            cfg=OperationalContextConfig(),
            target_fpr=0.01,
            calib_windows=1200,
            eval_windows=200,
            operational_windows=200,
            state_attack_windows=100,
            hidden=5,
            bootstrap=5000,
            permutations=10000,
            alpha=0.05,
            recall_margin=0.01,
        )
        self.assertEqual(plan["multiplicity"]["method"], "Holm")
        self.assertEqual(plan["multiplicity"]["family_size"], 4)
        self.assertEqual(len(plan["primary_endpoints"]), 4)
        self.assertNotIn("mean_semantic_operational_fpr", [x["metric"] for x in plan["primary_endpoints"]])

    def test_locked_plan_refuses_changed_plan_in_same_directory(self):
        plan = build_analysis_plan(
            seeds=DEFAULT_HOLDOUT_SEEDS,
            cfg=OperationalContextConfig(),
            target_fpr=0.01,
            calib_windows=1200,
            eval_windows=200,
            operational_windows=200,
            state_attack_windows=100,
            hidden=5,
            bootstrap=100,
            permutations=100,
            alpha=0.05,
            recall_margin=0.01,
        )
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            write_locked_plan(out, plan)
            changed = dict(plan)
            changed["alpha"] = 0.01
            with self.assertRaises(RuntimeError):
                write_locked_plan(out, changed)

    def test_primary_statistics_holm_is_not_diluted_by_secondary_endpoints(self):
        rows = []
        for seed in range(52, 62):
            rows.extend([
                {
                    "seed": seed,
                    "arm": ARM_STATIC,
                    "mean_operational_fpr": 0.06,
                    "mean_attack_window_recall": 0.87,
                    "state_attack_window_recall": 0.38,
                    "state_attack_event_latency_s": 0.4,
                },
                {
                    "seed": seed,
                    "arm": ARM_NOISY,
                    "mean_operational_fpr": 0.02,
                    "mean_attack_window_recall": 0.87,
                    "state_attack_window_recall": 0.38,
                    "state_attack_event_latency_s": 5.2,
                },
            ])
        stats = primary_statistics(
            pd.DataFrame(rows),
            seed=52,
            n_boot=1000,
            n_perm=1000,
            alpha=0.05,
            recall_margin=0.01,
        )
        self.assertEqual(len(stats), 4)
        h1 = stats[stats["hypothesis"] == "H1"].iloc[0]
        h2 = stats[stats["hypothesis"] == "H2"].iloc[0]
        h3 = stats[stats["hypothesis"] == "H3"].iloc[0]
        h4 = stats[stats["hypothesis"] == "H4"].iloc[0]
        self.assertLess(h1["holm_p"], 0.05)
        self.assertEqual(h1["decision"], "PASS_SUPERIORITY")
        self.assertEqual(h2["decision"], "PASS_NONINFERIORITY")
        self.assertEqual(h3["decision"], "PASS_NONINFERIORITY")
        self.assertEqual(h4["decision"], "LATENCY_COST_CONFIRMED")

    def test_noninferiority_uses_ci_margin_not_zero_difference_p_value(self):
        endpoint = PRIMARY_ENDPOINTS[1]
        row = {
            "delta_ci_lo": -0.004,
            "delta_ci_hi": 0.002,
            "holm_p": 1.0,
        }
        result = classify_primary(row, endpoint, alpha=0.05, margin=0.01)
        self.assertTrue(result["decision_pass"])
        self.assertEqual(result["decision"], "PASS_NONINFERIORITY")


if __name__ == "__main__":
    unittest.main()
