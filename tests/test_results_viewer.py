from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from gui.results.charts import build_figure, suggested_spec
from gui.results.loader import load_experiment
from gui.results.model import ChartSpec
from gui.results.tables import aggregate_table, descriptive_table


class ResultsViewerTests(unittest.TestCase):
    def _make_result_dir(self, root: Path):
        (root / "ss7" / "results").mkdir(parents=True)
        manifest = {"protocol": "ss7", "status": "ok", "git": {"commit": "abc123"},
                    "experiment": {"seeds": [42, 43]}, "config_hash": "cfg001"}
        (root / "ss7" / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        pd.DataFrame([
            {"seed": 42, "model": "rule", "episode_recall": 0.70, "auc": 0.80},
            {"seed": 43, "model": "rule", "episode_recall": 0.72, "auc": 0.82},
            {"seed": 42, "model": "rf", "episode_recall": 0.91, "auc": 0.96},
            {"seed": 43, "model": "rf", "episode_recall": 0.93, "auc": 0.97},
        ]).to_csv(root / "ss7" / "results" / "multiseed_raw.csv", index=False)
        (root / "ss7" / "results" / "null_test.json").write_text(
            json.dumps({"passed": True, "worst_auc": 0.5}), encoding="utf-8")

    def test_loader(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); self._make_result_dir(root); result = load_experiment(root)
            self.assertEqual(result.protocol, "ss7")
            self.assertTrue(result.tables)
            self.assertTrue(result.null_test().get("passed"))

    def test_aggregate(self):
        df = pd.DataFrame({"model": ["a", "a", "b", "b"], "seed": [1, 2, 1, 2], "recall": [0.8, 0.9, 0.6, 0.7]})
        agg = aggregate_table(df, ["model"], ["recall"])
        self.assertIn("recall_mean", agg.columns)
        self.assertIn("recall_ci95", agg.columns)
        self.assertEqual(len(agg), 2)

    def test_descriptive(self):
        df = pd.DataFrame({"x": [1, 2, 3], "name": ["a", "b", "c"]})
        d = descriptive_table(df)
        self.assertEqual(d.iloc[0]["metric"], "x")

    def test_chart(self):
        df = pd.DataFrame({"seed": [1, 2, 1, 2], "model": ["a", "a", "b", "b"], "recall": [0.8, 0.9, 0.6, 0.7]})
        fig = build_figure(df, ChartSpec(chart_type="bar", x="model", y="recall", aggregation="mean", error="ci95"))
        self.assertIsNotNone(fig)

    def test_suggested_spec(self):
        df = pd.DataFrame({"seed": [1, 2], "model": ["a", "b"], "episode_recall": [0.8, 0.9]})
        spec = suggested_spec(df)
        self.assertEqual(spec.x, "model")
        self.assertEqual(spec.y, "episode_recall")


if __name__ == "__main__":
    unittest.main()
