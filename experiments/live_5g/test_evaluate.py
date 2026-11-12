"""Scientific contract tests; fixtures are not experimental evidence."""
import math
import copy
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .evaluate import FULL, KEYS, METHODS, TAIL_FEATURES, add_episode_metrics, aggregate, attach_online_decisions, conformal_threshold, fit, fit_tail_model, load_frozen, metrics, require_disjoint, score, score_frame, tail_scores
except ImportError:
    from evaluate import FULL, KEYS, METHODS, TAIL_FEATURES, add_episode_metrics, aggregate, attach_online_decisions, conformal_threshold, fit, fit_tail_model, load_frozen, metrics, require_disjoint, score, score_frame, tail_scores


def fixture(run, n=1000):
    rng = np.random.default_rng(913)
    frame = pd.DataFrame({key: rng.uniform(0, .1, n) for key in FULL})
    for col in ("profile_policy_violation", "forbidden_operation", "route_mismatch", "report_mismatch", "auth_failure"):
        frame[col] = 0
    frame["run_id"] = run
    frame["window_id"] = np.arange(n)
    frame["nf_id"] = "test-fixture"
    frame["timestamp"] = time.time() + np.arange(n)
    lab = frame[KEYS].copy()
    lab["attack"] = 0
    lab["scenario"] = "N1"
    lab["onset"] = np.nan
    lab["episode_id"] = ""
    return frame, lab


def write_fixture(path, run, n=1000):
    path.mkdir()
    frame, labels = fixture(run, n)
    frame.to_csv(path / "features.csv", index=False)
    labels.to_csv(path / "labels.csv", index=False)
    return frame, labels


class ScientificContractTests(unittest.TestCase):
    def test_tail_is_frozen_train_only_one_sided_and_calibrated(self):
        train = pd.DataFrame({"request_count": [2., 3., 4.], "error_rate": [0., 0., 0.],
                              "update_count": [1., 1., 1.]})
        model = fit_tail_model(train)
        self.assertEqual(model["center"], [3., 0., 1.])
        self.assertEqual(model["resolution_floor"], [1., .25, 1.])
        below = pd.DataFrame({key: [0.] for key in TAIL_FEATURES})
        self.assertEqual(tail_scores(model, below)[0], 0.)
        # Unlike a capped tree score, the learned envelope preserves the ordering
        # of different out-of-support magnitudes without seeing any attack label.
        outside = train.iloc[[2, 2]].copy()
        outside["request_count"] = [7., 11.]
        scores = tail_scores(model, outside)
        self.assertGreater(scores[1], scores[0])
        unchanged = copy.deepcopy(model)
        outside["attack"] = 1
        outside["scenario"] = "arbitrary"
        np.testing.assert_array_equal(scores, tail_scores(model, outside))
        self.assertEqual(model, unchanged)
        threshold = conformal_threshold(np.repeat(tail_scores(model, train), 400))
        self.assertFalse((tail_scores(model, train) > threshold).any())

    def test_complete_pipeline_preserves_empty_and_positive_episode_tables(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_fixture(root / "train", "train", 100)
            write_fixture(root / "cal", "cal")
            fit(root / "train", root / "cal", root / "frozen")
            bundle = load_frozen(root / "frozen")
            for has_attack in (False, True):
                name = "attack" if has_attack else "normal"
                test_dir = root / name
                test, labels = write_fixture(test_dir, name, 12)
                # Every held-out observation and online decision uses the real clock
                # after fit. Data values remain explicitly synthetic unit fixtures.
                test["timestamp"] = [time.time() for _ in range(len(test))]
                if has_attack:
                    test.loc[0, "report_mismatch"] = 1
                    labels.loc[0, ["attack", "scenario", "onset", "episode_id"]] = [
                        1, "S1", test.timestamp.iloc[0], "controlled-fixture"]
                test.to_csv(test_dir / "features.csv", index=False)
                labels.to_csv(test_dir / "labels.csv", index=False)
                online = score_frame(bundle, test)
                online["decision_timestamp"] = time.time()
                online.to_csv(test_dir / "online_predictions.csv", index=False)
                result = score(root / "frozen", test_dir, root / "reports" / name)
                self.assertEqual(result["delay_timing_source"], "online_decision_timestamp")
                summary = aggregate(root / "reports", root / "summary", expected_runs=1 + int(has_attack))
                hybrid = next(x for x in summary["metrics"] if x["method"] == "hybrid")
                self.assertEqual(hybrid["episodes"], int(has_attack))
                self.assertEqual(hybrid["detected_episodes"], int(has_attack))
                if has_attack:
                    self.assertGreaterEqual(hybrid["median_delay_detected_seconds"], 0.)
                else:
                    # Also support the previous platform-dependent blank CSV export.
                    (root / "reports" / name / "episodes.csv").write_bytes(b"\r\n")
                    aggregate(root / "reports", root / "summary", expected_runs=1)
            with self.assertRaisesRegex(ValueError, "Expected 9"):
                aggregate(root / "reports", root / "summary", expected_runs=9)
            with self.assertRaisesRegex(ValueError, "missing positive scenario"):
                aggregate(root / "reports", root / "summary", required_scenarios=["S1", "S2", "S3", "S4"])
            aggregate(root / "reports" / "attack", root / "one-summary", expected_runs=1, required_scenarios=["S1"])

    def test_online_decisions_require_complete_causal_frozen_replay(self):
        predictions = pd.DataFrame({"run_id": ["test"], "window_id": [1], "nf_id": ["n"],
            "timestamp": [10.], "method": ["hybrid"], "score": [1.], "alarm": [1]})
        online = predictions.copy()
        online["decision_timestamp"] = 12.
        checked = attach_online_decisions(predictions, online)
        self.assertEqual(checked.decision_timestamp.iloc[0], 12.)
        wrong = online.copy()
        wrong["decision_timestamp"] = 9.
        with self.assertRaisesRegex(ValueError, "precedes"):
            attach_online_decisions(predictions, wrong)
        wrong = online.copy()
        wrong["alarm"] = 0
        with self.assertRaisesRegex(ValueError, "disagree"):
            attach_online_decisions(predictions, wrong)
        with self.assertRaisesRegex(ValueError, "cover exactly"):
            attach_online_decisions(predictions, online.iloc[0:0])

    def test_episode_pooling_retains_misses(self):
        totals = [{"method": "hybrid"}]
        episodes = pd.DataFrame({"method": ["hybrid", "hybrid"], "detected": [True, False],
                                 "delay_seconds": [2., np.nan]})
        add_episode_metrics(totals, episodes)
        self.assertEqual(totals[0]["episodes"], 2)
        self.assertEqual(totals[0]["missed_episodes"], 1)
        self.assertEqual(totals[0]["episode_recall"], .5)
        self.assertEqual(totals[0]["median_delay_detected_seconds"], 2.)

    def test_conservative_rank_and_strict_ties(self):
        self.assertTrue(math.isinf(conformal_threshold([0.] * 998)))
        threshold = conformal_threshold([.1] * 999)
        self.assertEqual(threshold, .1)
        self.assertFalse(.1 > threshold)

    def test_reject_run_overlap(self):
        a, _ = fixture("same", 5)
        b, _ = fixture("same", 5)
        with self.assertRaisesRegex(ValueError, "Overlapping"):
            require_disjoint(a, b)

    def test_zero_denominators(self):
        frame = pd.DataFrame({"attack": [0, 0], "alarm": [0, 0], "score": [-1., -1.]})
        result = metrics(frame)
        self.assertIsNone(result["recall"])
        self.assertIsNone(result["precision"])
        self.assertIsNone(result["average_precision"])
        self.assertEqual(result["fpr"], 0)
        self.assertGreater(result["fpr_wilson_descriptive"][1], 0)

    def test_frozen_features_and_prospective_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_fixture(root / "train", "train", 100)
            write_fixture(root / "cal", "cal")
            manifest = fit(root / "train", root / "cal", root / "frozen")
            self.assertTrue(manifest["threshold_sufficient"])
            bundle = load_frozen(root / "frozen")
            legacy = copy.deepcopy(bundle)
            legacy["protocol_version"] = "live-5g-v1"
            legacy.pop("tail_model")
            legacy["thresholds"].pop("robust_tail")
            original, _ = fixture("test", 12)
            changed = original.copy()
            changed["attack"] = 1
            changed["scenario"] = "S4"
            changed["onset"] = 999
            a = score_frame(bundle, original)
            b = score_frame(bundle, changed)
            pd.testing.assert_series_equal(a.score, b.score)
            legacy_predictions = score_frame(legacy, original)
            self.assertEqual(set(legacy_predictions.method), set(METHODS))
            pd.testing.assert_frame_equal(legacy_predictions.reset_index(drop=True),
                a.loc[a.method.isin(METHODS)].reset_index(drop=True))
            poisoned = original.copy()
            poisoned.loc[0, "report_mismatch"] = 1
            c = score_frame(bundle, poisoned)
            self.assertEqual(c.loc[(c.method == "hybrid") & (c.window_id == 0), "alarm"].iloc[0], 1)
            self.assertEqual(c.loc[(c.method == "hybrid_without_KIE") & (c.window_id == 0), "alarm"].iloc[0],
                             a.loc[(a.method == "hybrid_without_KIE") & (a.window_id == 0), "alarm"].iloc[0])
            test, _ = write_fixture(root / "test", "test", 12)
            test["timestamp"] = 1.
            test.to_csv(root / "test" / "features.csv", index=False)
            with self.assertRaisesRegex(ValueError, "after model freezing"):
                score(root / "frozen", root / "test", root / "result")
            (root / "frozen" / "model.joblib").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                load_frozen(root / "frozen")

    def test_alternative_detectors_write_json_safe_frozen_manifests(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_fixture(root / "train", "train", 120)
            write_fixture(root / "cal", "cal", 1000)
            for detector in ("ocsvm_rbf", "lof", "elliptic"):
                with self.subTest(detector=detector):
                    frozen = root / detector
                    manifest = fit(root / "train", root / "cal", frozen, detector=detector)
                    self.assertEqual(manifest["detector"], detector)
                    self.assertTrue(manifest["threshold_sufficient"])
                    recorded = __import__("json").loads((frozen / "manifest.json").read_text())
                    self.assertEqual(recorded["detector"], detector)
                    self.assertIsInstance(recorded["detector_parameters"], dict)
                    bundle = load_frozen(frozen)
                    self.assertEqual(bundle["manifest"]["detector"], detector)

    def test_training_rejects_attack(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, labels = write_fixture(root / "train", "train", 10)
            labels.loc[0, ["attack", "onset", "episode_id"]] = [1, 1., "attack"]
            labels.to_csv(root / "train" / "labels.csv", index=False)
            write_fixture(root / "cal", "cal", 10)
            with self.assertRaisesRegex(ValueError, "only normal"):
                fit(root / "train", root / "cal", root / "frozen")


if __name__ == "__main__":
    unittest.main()
