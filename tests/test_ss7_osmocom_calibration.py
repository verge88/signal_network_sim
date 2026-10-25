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
    DEFAULT_HOLDOUT_SIZES,
    DEFAULT_PAYLOAD_SIZES,
    calibrated_template,
    fit_transport_profile,
    infer_payload_sizes,
    load_trace,
    run_calibration,
    summarize_trace,
)
from ss7.fix import RngBook, SimConfig, Ss7Sim  # noqa: E402


class OsmocomCalibrationTests(unittest.TestCase):
    def _trace(self, per_size=20, m3ua_overhead=32, sctp_ip_overhead=40):
        rows = []
        t = 0.0
        for payload in DEFAULT_PAYLOAD_SIZES:
            for i in range(per_size):
                m3ua_len = payload + m3ua_overhead
                ip_len = m3ua_len + sctp_ip_overhead
                rows.append({
                    "t": t,
                    "frame_len": ip_len + 20,
                    "ip_len": ip_len,
                    "src_port": 3333,
                    "dst_port": 2905,
                    "m3ua_len": m3ua_len,
                    "opc": 23,
                    "dpc": 1,
                    "si": 15,
                })
                t += 0.02 + (0.001 if i % 7 == 0 else 0.0)
        return pd.DataFrame(rows)

    def test_infers_constant_m3ua_overhead(self):
        annotated, overhead = infer_payload_sizes(self._trace())
        self.assertEqual(overhead, 32.0)
        self.assertEqual(
            set(annotated["payload_bytes"].unique()),
            set(DEFAULT_PAYLOAD_SIZES),
        )

    def test_transport_fit_uses_payload_size_holdout(self):
        annotated, _ = infer_payload_sizes(self._trace())
        fit, holdout = fit_transport_profile(
            annotated, DEFAULT_HOLDOUT_SIZES, min_frames_per_size=10
        )
        self.assertEqual(fit["m3ua_overhead_bytes"], 32.0)
        self.assertEqual(fit["sctp_ipv4_overhead_bytes"], 40.0)
        self.assertEqual(fit["transport_overhead_bytes"], 72.0)
        self.assertEqual(fit["validation_mae_bytes"], 0.0)
        self.assertEqual(fit["validation_max_abs_error_bytes"], 0.0)
        self.assertTrue((holdout["abs_error_bytes"] == 0).all())

    def test_calibrated_template_changes_only_wire_load_not_flow_rates(self):
        cfg = SimConfig(seed=7)
        base = Ss7Sim(cfg, RngBook(7), size_capacities=False)
        profile = {"transport_overhead_bytes": 72.0}
        calibrated = calibrated_template(cfg, profile, seed=7)

        self.assertEqual(calibrated.transport_overhead_bytes, 72.0)
        self.assertEqual(
            [f.rate for f in calibrated.flows],
            [f.rate for f in base.flows],
        )
        self.assertGreater(
            max(calibrated._nominal_offered().values()),
            max(base._nominal_offered().values()),
        )
        self.assertGreater(calibrated.mean_msg_bits, base.mean_msg_bits)

    def test_default_simulator_preserves_zero_overhead_baseline(self):
        cfg = SimConfig(seed=11)
        sim = Ss7Sim(cfg, RngBook(11), size_capacities=False)
        self.assertEqual(sim.transport_overhead_bytes, 0.0)
        self.assertAlmostEqual(
            sim.mean_msg_bits,
            8.0 * float(np.mean([
                60, 25, 25, 28, 22, 55, 190, 190, 58,
                62, 70, 160, 60, 150, 90, 40, 38,
            ])),
        )

    def test_run_calibration_writes_holdout_evidence(self):
        with tempfile.TemporaryDirectory() as td:
            trace = Path(td) / "trace.csv"
            out = Path(td) / "out"
            self._trace().to_csv(trace, index=False)
            result = run_calibration(trace, out)
            self.assertEqual(
                result["fit"]["transport_overhead_bytes"], 72.0
            )
            self.assertTrue((out / "calibrated_profile.json").exists())
            self.assertTrue((out / "payload_size_summary.csv").exists())
            self.assertTrue((out / "holdout_validation.csv").exists())
            self.assertTrue((out / "summary.md").exists())

    def test_trace_summary_and_loader(self):
        with tempfile.TemporaryDirectory() as td:
            trace = Path(td) / "trace.csv"
            self._trace().to_csv(trace, index=False)
            loaded = load_trace(trace)
            stats = summarize_trace(loaded)
            self.assertGreater(stats.frames_per_s, 0)
            self.assertGreater(stats.ip_len_p95, stats.ip_len_p50)

    def test_csv_loader_rejects_wrong_schema(self):
        with tempfile.TemporaryDirectory() as td:
            bad = Path(td) / "bad.csv"
            pd.DataFrame({"t": [1.0]}).to_csv(bad, index=False)
            with self.assertRaises(ValueError):
                load_trace(bad)


if __name__ == "__main__":
    unittest.main()
