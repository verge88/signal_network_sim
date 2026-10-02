import tempfile
import unittest
from pathlib import Path

from experiments.baseline import (
    canonical_hash,
    compare_metric_rows,
    read_metrics_csv,
    sha256_file,
)


class Stage0BaselineTests(unittest.TestCase):

    def test_canonical_hash_ignores_dict_order(self):
        a = {"b": 2, "a": 1}
        b = {"a": 1, "b": 2}
        self.assertEqual(
            canonical_hash(a),
            canonical_hash(b),
        )

    def test_file_hash_changes_with_content(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "x.txt"
            path.write_text("a", encoding="utf-8")
            h1 = sha256_file(path)

            path.write_text("b", encoding="utf-8")
            h2 = sha256_file(path)

            self.assertNotEqual(h1, h2)

    def test_metrics_csv_conversion(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.csv"
            path.write_text(
                "seed,name,f1,fpr\n"
                "42,Model A,0.91,0.01\n"
                "43,Model A,0.92,0.02\n",
                encoding="utf-8",
            )

            result = read_metrics_csv(path)

            self.assertEqual(
                result["row_count"],
                2,
            )
            self.assertEqual(
                result["rows"][0]["seed"],
                42,
            )
            self.assertAlmostEqual(
                result["rows"][0]["f1"],
                0.91,
            )

    def test_metric_comparison(self):
        a = [
            {
                "seed": 42,
                "name": "A",
                "family": "unsup",
                "f1": 0.9,
            }
        ]
        b = [
            {
                "seed": 42,
                "name": "A",
                "family": "unsup",
                "f1": 0.9000001,
            }
        ]

        strict = compare_metric_rows(
            a,
            b,
            atol=1e-9,
        )
        relaxed = compare_metric_rows(
            a,
            b,
            atol=1e-3,
        )

        self.assertFalse(
            strict["equivalent"]
        )
        self.assertTrue(
            relaxed["equivalent"]
        )


if __name__ == "__main__":
    unittest.main()
