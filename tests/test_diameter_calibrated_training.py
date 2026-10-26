import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "diameter")]

from diameter.diameter_training_calibrated import (  # noqa: E402
    CascadeCompromiseDetector,
    DataPipeline,
    TrainingConfig,
)
from diameter.run_seeds_diameter_calibrated import (  # noqa: E402
    DIAMETER_CALIBRATED,
    OPEN5GS_CALIBRATION,
)
from diameter.ms_multiseed_core import (  # noqa: E402
    count_false_incidents,
    normalize_score01,
    normalize_score01_from_reference,
)


class DiameterCalibratedTrainingTests(unittest.TestCase):
    def _df(self):
        rows = []
        # 40 normal time blocks and 12 attack episodes across 4 types.
        for interval in range(0, 4800, 20):
            for node in (7, 8, 9):
                rows.append({
                    "interval": interval,
                    "time_s": interval * 15,
                    "node_id": node,
                    "node_type": "MME",
                    "attack_type": "none",
                    "is_anomaly": 0,
                })

        episodes = [
            ("location_tracking", 7, 100, 109),
            ("location_tracking", 8, 700, 709),
            ("location_tracking", 9, 1300, 1309),
            ("sms_data_interception", 7, 300, 309),
            ("sms_data_interception", 8, 900, 909),
            ("sms_data_interception", 9, 1500, 1509),
            ("signaling_dos", 7, 500, 509),
            ("signaling_dos", 8, 1100, 1109),
            ("signaling_dos", 9, 1700, 1709),
            ("node_compromise", 7, 1900, 1939),
            ("node_compromise", 8, 2300, 2339),
            ("node_compromise", 9, 2700, 2739),
        ]
        for attack, node, start, end in episodes:
            for interval in range(start, end + 1):
                rows.append({
                    "interval": interval,
                    "time_s": interval * 15,
                    "node_id": node,
                    "node_type": "MME",
                    "attack_type": attack,
                    "is_anomaly": 1,
                })
        return pd.DataFrame(rows)

    def test_open5gs_profile_is_fixed_in_protocol_spec(self):
        self.assertEqual(
            DIAMETER_CALIBRATED.sim_config_overrides,
            OPEN5GS_CALIBRATION,
        )
        self.assertAlmostEqual(
            OPEN5GS_CALIBRATION["s6a_transaction_size_bytes"],
            794.2608695652174,
        )
        self.assertAlmostEqual(
            OPEN5GS_CALIBRATION[
                "s6a_command_transaction_size_bytes"
            ]["AIR"],
            584.9230769230769,
        )
        self.assertAlmostEqual(
            OPEN5GS_CALIBRATION[
                "s6a_command_transaction_size_bytes"
            ]["ULR"],
            1066.4,
        )

    def test_grouped_split_keeps_groups_and_episodes_disjoint(self):
        cfg = TrainingConfig(
            seed=42,
            split_normal_block_intervals=120,
            split_min_test_episodes=2,
            split_min_test_attack_types=2,
        )
        pipe = DataPipeline(cfg)
        df = pipe._annotate_groups(self._df())
        train, val, test, _ = pipe._grouped_split(df)

        gs = [
            set(x["split_group"].astype(str))
            for x in (train, val, test)
        ]
        self.assertFalse(gs[0] & gs[1])
        self.assertFalse(gs[0] & gs[2])
        self.assertFalse(gs[1] & gs[2])

        for part in (train, val, test):
            self.assertEqual(part["is_anomaly"].nunique(), 2)

        test_eps = set(
            test.loc[test["is_anomaly"] == 1, "attack_episode_id"]
            .astype(str)
        )
        self.assertGreaterEqual(len(test_eps), 2)
        self.assertTrue(
            (train["attack_type"].astype(str) == "node_compromise").any()
        )
        self.assertTrue(
            (test["attack_type"].astype(str) == "node_compromise").any()
        )

    def test_stacking_normalizer_sanitizes_nonfinite_scores(self):
        out = normalize_score01([float("nan"), 1.0, 2.0, float("inf")])
        self.assertEqual(len(out), 4)
        self.assertTrue(pd.Series(out).map(lambda x: x == x).all())
        self.assertTrue(((out >= 0.0) & (out <= 1.0)).all())


    def test_cascade_stage2_degrades_to_stage1_on_one_class(self):
        import numpy as np

        cfg = TrainingConfig(
            cascade_rf_trees=5,
            cascade_gb_estimators=5,
        )
        model = CascadeCompromiseDetector(cfg, seed=123)
        x_full = np.array([
            [0.0, 0.0],
            [0.1, 0.0],
            [1.0, 1.0],
            [1.1, 1.0],
        ])
        y = np.array([0, 0, 1, 1])
        x_comp = x_full.copy()
        y_comp = np.zeros(4, dtype=int)

        model.fit(x_full, y, x_comp, y_comp)
        self.assertFalse(model.stage2_enabled)
        self.assertEqual(model.stage2_classes, [0])

        p1 = model.predict_proba_stage1(x_full)
        p2 = model.predict_proba_stage2(x_comp)
        pc = model.predict_proba(x_full, x_comp)
        self.assertTrue(np.allclose(p2, 0.0))
        self.assertTrue(np.allclose(pc, p1))


    def test_reference_normalizer_uses_validation_quantiles(self):
        import numpy as np

        ref = np.array([0.0, 1.0, 2.0, 3.0])
        test = np.array([0.0, 3.0, 30.0])
        out = normalize_score01_from_reference(ref, test)
        self.assertEqual(len(out), 3)
        self.assertGreater(out[1], out[0])
        self.assertEqual(out[2], 1.0)


    def test_false_incidents_group_contiguous_fp_windows(self):
        import numpy as np

        df = pd.DataFrame({
            "node_id": [7, 7, 7, 7, 8, 8],
            "interval": [10, 11, 13, 14, 10, 11],
        })
        y = np.zeros(len(df), dtype=int)
        yp = np.array([1, 1, 1, 0, 1, 0])
        # node 7: intervals 10-11 = one incident; interval 13 = another.
        # node 8: interval 10 = one incident.
        self.assertEqual(count_false_incidents(df, yp, y), 3)

    def test_attack_episode_is_not_split_by_normal_time_block(self):
        cfg = TrainingConfig(seed=7, split_normal_block_intervals=10)
        pipe = DataPipeline(cfg)
        df = pd.DataFrame([
            {
                "interval": i,
                "time_s": i * 15,
                "node_id": 7,
                "node_type": "MME",
                "attack_type": "location_tracking",
                "is_anomaly": 1,
            }
            for i in range(95, 115)
        ])
        out = pipe._annotate_groups(df)
        self.assertEqual(out["attack_episode_id"].nunique(), 1)
        self.assertEqual(out["split_group"].nunique(), 1)


if __name__ == "__main__":
    unittest.main()
