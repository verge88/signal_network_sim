import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ss7"))

from ss7.osmocom_calibration import (  # noqa: E402
    calibrated_template,
    fit_profile,
    frame_kind,
    load_trace,
    split_trace,
    summarize_trace,
)
from ss7.fix import SimConfig  # noqa: E402


class OsmocomCalibrationTests(unittest.TestCase):
    def _trace(self, n=80):
        t = np.arange(n, dtype=float) * 0.05
        info = ["DATA (Payload Data)" if i % 3 else "ASPAC ACK"
                for i in range(n)]
        return pd.DataFrame({
            "t": t,
            "frame_len": 120 + (np.arange(n) % 7),
            "src_port": np.where(np.arange(n) % 2, 2905, 40000),
            "dst_port": np.where(np.arange(n) % 2, 40000, 2905),
            "info": info,
        })

    def test_frame_classification(self):
        self.assertEqual(frame_kind("DATA (Payload Data)"), "data")
        self.assertEqual(frame_kind("ASPUP ACK"), "control")
        self.assertEqual(frame_kind("unknown"), "other")

    def test_chronological_split_is_disjoint(self):
        train, holdout = split_trace(self._trace(), 0.60)
        self.assertLessEqual(train["t"].max(), holdout["t"].min())
        self.assertGreaterEqual(len(train), 8)
        self.assertGreaterEqual(len(holdout), 8)

    def test_summary_counts_data_and_control(self):
        stats = summarize_trace(self._trace())
        self.assertGreater(stats.frames_per_s, 0)
        self.assertGreater(stats.data_frames, 0)
        self.assertGreater(stats.control_frames, 0)
        self.assertGreater(stats.interarrival_p50_ms, 0)

    def test_fit_scales_canonical_reference(self):
        train, _ = split_trace(self._trace(120), 0.60)
        cfg = SimConfig(seed=7)
        profile = fit_profile(train, cfg, seed=7)
        self.assertGreater(profile["flow_scale"], 0)
        self.assertIn(profile["rate_basis"],
                      {"m3ua_data_frames", "all_m3ua_sctp_frames"})
        self.assertGreater(profile["recommended_capacity_bps"], 0)
        base = calibrated_template(cfg, {"flow_scale": 1.0}, seed=7)
        scaled = calibrated_template(cfg, profile, seed=7)
        base_rate = sum(flow.rate for flow in base.flows)
        scaled_rate = sum(flow.rate for flow in scaled.flows)
        self.assertAlmostEqual(
            scaled_rate / base_rate, profile["flow_scale"], places=9
        )

    def test_csv_loader_rejects_wrong_schema(self):
        with tempfile.TemporaryDirectory() as td:
            bad = Path(td) / "bad.csv"
            pd.DataFrame({"t": [1.0]}).to_csv(bad, index=False)
            with self.assertRaises(ValueError):
                load_trace(bad)


if __name__ == "__main__":
    unittest.main()
