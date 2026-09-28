"""Regression checks for SS7 CLI exits, null reports and block allocation."""
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "ss7")]
from gui.ml import harness, cli_args
from fix import SimConfig, blocked_split, null_test
import fix
import ss7_training_v7 as training


class RegressionTests(unittest.TestCase):
    def test_cli_exit_codes(self):
        for code in (None, 0, 2, "failure"):
            with self.subTest(code=code), tempfile.TemporaryDirectory() as directory:
                script = Path(directory) / "trainer.py"
                script.write_text(f"raise SystemExit({code!r})", encoding="utf-8")
                module = SimpleNamespace(__file__=str(script), main=lambda argv=None: None)
                job = SimpleNamespace(sim_dir="", train_module="dummy", protocol="ss7",
                    run_dir=directory, train_script=str(script), seed=42,
                    train_options={}, train_overrides={}, train_extra_args="",
                    cli_args=[], capture_models=False)
                capture = Mock()
                old_argv = sys.argv
                with patch.object(harness.importlib, "import_module", return_value=module), \
                     patch.object(cli_args, "build_argv", return_value=["--seed", "42"]):
                    if code in (None, 0):
                        harness.stage_train(job, "data.csv", capture)
                        capture.collect_torch.assert_called_once()
                    else:
                        with self.assertRaises(SystemExit):
                            harness.stage_train(job, "data.csv", capture)
                self.assertIs(sys.argv, old_argv)
                capture.restore.assert_called_once()

    def test_null_report_all_roles_even_when_diagnostics_fail(self):
        df = pd.DataFrame({"nid": [0]*4+[1]*4, "label": [0,0,1,1]*2,
                           **{f"f{i}": range(8) for i in range(5)}})
        sim = SimpleNamespace(topo=SimpleNamespace(nodes={
            0: SimpleNamespace(ntype=SimpleNamespace(name="SP")),
            1: SimpleNamespace(ntype=SimpleNamespace(name="IGW"))}))
        with patch.object(fix, "_counterfactual_null", return_value={"passed": True}), \
             patch.object(fix, "build_dataset", return_value=(df, sim, None)), \
             patch.object(fix, "_usable_columns", return_value=[f"f{i}" for i in range(5)]), \
             patch.object(fix, "_null_test_one", side_effect=AssertionError("diagnostic")):
            report = null_test(SimConfig(), [f"f{i}" for i in range(5)], min_pos=1, log=lambda s: None)
        self.assertEqual(set(report["by_role"]), {"SP", "IGW"})
        self.assertFalse(report["role_diagnostic_passed"])

    def test_gate_rejects_missing_or_failed_reports(self):
        for report in (None, {"passed": False}, {"passed": True, "skipped": True},
                       {"passed": True, "role_diagnostic_passed": False}):
            with patch.object(training, "_load_sidecar", return_value=(None, None)), \
                 patch.object(training, "null_test", return_value=report):
                with self.assertRaises(AssertionError):
                    training.gate_null_test(training.TrainingConfig(), [], False)
                self.assertFalse(training.gate_null_test(training.TrainingConfig(), [], True)["passed"])

    def test_split_classes_disjoint_episodes_and_time_gap(self):
        df = pd.DataFrame({"t": np.arange(2880), "label": 0, "episode_id": -1})
        for e, (lo, hi) in enumerate([(2561,2879),(2236,2422),(2044,2250),(125,508)]):
            df.loc[lo:hi, ["label", "episode_id"]] = [1,e]
        cfg = SimConfig(seed=42)
        parts = blocked_split(df, cfg, np.random.default_rng(42))
        again = blocked_split(df, cfg, np.random.default_rng(42))
        for a,b in zip(parts, again):
            pd.testing.assert_frame_equal(a,b)
            self.assertEqual(set(a.label), {0,1})
        for i,a in enumerate(parts):
            for b in parts[i+1:]:
                self.assertFalse(set(a[a.episode_id>=0].episode_id) & set(b[b.episode_id>=0].episode_id))
                self.assertGreater(np.abs(a.t.to_numpy()[:,None]-b.t.to_numpy()).min(), cfg.feat_window+cfg.z_window)
        with self.assertRaises(ValueError):
            blocked_split(df.assign(label=0), cfg, np.random.default_rng(42))

if __name__ == "__main__":
    unittest.main()
