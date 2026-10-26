import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "diameter")]

from diameter.diameter_sim_v1 import (  # noqa: E402
    DiameterNetworkTopology,
    DiameterTrafficGenerator,
    NodeType,
    SimulationConfig,
)
from diameter.open5gs_s6a_calibration import (  # noqa: E402
    calibrated_config,
    chronological_split,
    fit_profile,
    load_trace,
    pair_transactions,
    run_calibration,
)


class Open5GSS6aCalibrationTests(unittest.TestCase):
    def _frames(self, n_each=20):
        rows = []
        t = 0.0
        seq = 1000
        specs = {
            318: (300, 500),   # AIR/AIA => IP transaction 880 B
            316: (420, 700),   # ULR/ULA => IP transaction 1200 B
        }
        for i in range(n_each):
            for code, (req_d, ans_d) in specs.items():
                hop = seq
                end = seq + 100000
                rows.append({
                    "t": t,
                    "frame_len": req_d + 54,
                    "ip_len": req_d + 40,
                    "diameter_len": req_d,
                    "cmd_code": code,
                    "is_request": 1,
                    "app_id": 16777251,
                    "hop_by_hop": hop,
                    "end_to_end": end,
                })
                rows.append({
                    "t": t + 0.005,
                    "frame_len": ans_d + 54,
                    "ip_len": ans_d + 40,
                    "diameter_len": ans_d,
                    "cmd_code": code,
                    "is_request": 0,
                    "app_id": 16777251,
                    "hop_by_hop": hop,
                    "end_to_end": end,
                })
                seq += 1
                t += 0.020
        return pd.DataFrame(rows)

    def test_pair_transactions(self):
        tx = pair_transactions(self._frames())
        self.assertEqual(len(tx), 40)
        self.assertEqual(set(tx["command"]), {"AIR", "ULR"})
        air = tx[tx["command"] == "AIR"].iloc[0]
        ulr = tx[tx["command"] == "ULR"].iloc[0]
        self.assertEqual(air["transaction_ip_bytes"], 880.0)
        self.assertEqual(ulr["transaction_ip_bytes"], 1200.0)
        self.assertAlmostEqual(air["rtt_ms"], 5.0, places=6)

    def test_chronological_fit_holdout_is_disjoint(self):
        tx = pair_transactions(self._frames())
        fit, hold = chronological_split(tx, 0.60)
        self.assertLess(fit["request_t"].max(), hold["request_t"].min())
        self.assertEqual(len(fit) + len(hold), len(tx))

    def test_fit_command_sizes_and_holdout(self):
        tx = pair_transactions(self._frames())
        fit, hold = chronological_split(tx, 0.60)
        profile, pred, _ = fit_profile(fit, hold)
        self.assertEqual(
            profile["s6a_command_transaction_size_bytes"]["AIR"], 880.0
        )
        self.assertEqual(
            profile["s6a_command_transaction_size_bytes"]["ULR"], 1200.0
        )
        self.assertEqual(profile["s6a_transaction_size_bytes"], 1040.0)
        self.assertEqual(profile["holdout_mae_bytes"], 0.0)
        self.assertTrue((pred["abs_error_bytes"] == 0).all())

    def test_calibrated_config_changes_only_s6a_wire_size_fields(self):
        base = SimulationConfig(seed=7)
        profile = {
            "s6a_transaction_size_bytes": 1040.0,
            "s6a_command_transaction_size_bytes": {
                "AIR": 880.0,
                "ULR": 1200.0,
            },
        }
        cfg = calibrated_config(base, profile)
        self.assertEqual(cfg.s6a_transaction_size_bytes, 1040.0)
        self.assertEqual(
            cfg.s6a_command_transaction_size_bytes["AIR"], 880.0
        )
        self.assertEqual(cfg.seed, base.seed)
        self.assertEqual(cfg.ar_phi, base.ar_phi)
        self.assertEqual(cfg.avg_message_size_bytes, base.avg_message_size_bytes)

    def test_simulator_legacy_load_is_preserved_without_profile(self):
        cfg = SimulationConfig(seed=3)
        topo = DiameterNetworkTopology(cfg)
        gen = DiameterTrafficGenerator(topo, cfg)
        node = next(n for n in topo.nodes.values() if n.node_type == NodeType.MME)
        traffic = {
            "total_messages": 100,
            "s6a_count": 50,
            "s6a_air_count": 30,
            "s6a_ulr_count": 20,
        }
        gen.add_traffic_load_to_links(node, traffic)
        load = max(l.current_load_mbps for l in topo.links.values())
        expected = 100 * 500 * 8 / (cfg.interval_s * 1_000_000.0)
        self.assertAlmostEqual(load, expected)

    def test_simulator_applies_only_calibrated_s6a_bytes(self):
        cfg = SimulationConfig(
            seed=3,
            s6a_transaction_size_bytes=1040.0,
            s6a_command_transaction_size_bytes={
                "AIR": 880.0,
                "ULR": 1200.0,
            },
        )
        topo = DiameterNetworkTopology(cfg)
        gen = DiameterTrafficGenerator(topo, cfg)
        node = next(n for n in topo.nodes.values() if n.node_type == NodeType.MME)
        traffic = {
            "total_messages": 100,
            "s6a_count": 50,
            "s6a_air_count": 30,
            "s6a_ulr_count": 20,
        }
        gen.add_traffic_load_to_links(node, traffic)
        load = max(l.current_load_mbps for l in topo.links.values())
        expected_bytes = 30 * 880 + 20 * 1200 + 50 * 500
        expected = expected_bytes * 8 / (cfg.interval_s * 1_000_000.0)
        self.assertAlmostEqual(load, expected)

    def test_run_calibration_writes_evidence(self):
        with tempfile.TemporaryDirectory() as td:
            trace = Path(td) / "s6a.csv"
            out = Path(td) / "out"
            self._frames().to_csv(trace, index=False)
            result = run_calibration(trace, out)
            self.assertEqual(
                result["fit"]["s6a_transaction_size_bytes"], 1040.0
            )
            for name in (
                "calibrated_profile.json",
                "s6a_frames.csv",
                "s6a_transactions.csv",
                "fit_transactions.csv",
                "holdout_validation.csv",
                "command_fit_summary.csv",
                "summary.md",
            ):
                self.assertTrue((out / name).exists(), name)

    def test_loader_rejects_wrong_schema(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "bad.csv"
            pd.DataFrame({"t": [1.0]}).to_csv(path, index=False)
            with self.assertRaises(ValueError):
                load_trace(path)


if __name__ == "__main__":
    unittest.main()
