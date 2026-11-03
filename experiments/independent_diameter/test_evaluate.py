from dataclasses import dataclass
import unittest

from experiments.independent_diameter.evaluate import (
    check_test_seed, fixed_metrics, shift_scenarios, validate_features,
)


class IndependentContractTests(unittest.TestCase):
    def test_training_seed_is_rejected(self):
        for seed in range(42, 52):
            with self.assertRaises(ValueError):
                check_test_seed(seed)
        check_test_seed(1001)

    def test_fixed_threshold_is_not_recalibrated_from_test_labels(self):
        a = fixed_metrics([0, 0, 1, 1], [0.1, 0.8, 0.6, 0.9], 0.7)
        b = fixed_metrics([1, 0, 0, 1], [0.1, 0.8, 0.6, 0.9], 0.7)
        self.assertEqual(a["threshold"], b["threshold"])
        self.assertEqual(a["recall"], 0.5)
        self.assertEqual(a["fpr"], 0.5)

    def test_ground_truth_is_rejected(self):
        validate_features(["obs_total_messages", "node_type_enc"])
        for column in ("gt_count", "attack_type", "is_anomaly", "split_group"):
            with self.assertRaises(ValueError):
                validate_features([column])

    def test_nonfinite_scores_are_not_silently_imputed(self):
        for bad in (float('inf'), float('nan')):
            with self.assertRaises(ValueError):
                fixed_metrics([0, 1], [0.1, bad], 0.7)

    def test_shift_preserves_duration_and_original_scenarios(self):
        @dataclass
        class Scenario:
            target_nodes: list
            start_interval: int
            end_interval: int
        original = Scenario([7], 100, 150)
        result = shift_scenarios([original], {7: 9}, 137)[0]
        self.assertEqual((result.target_nodes, result.start_interval, result.end_interval), ([9], 237, 287))
        self.assertEqual(original.target_nodes, [7])
        self.assertEqual(original.start_interval, 100)


if __name__ == "__main__":
    unittest.main()
