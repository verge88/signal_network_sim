from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_open5gs_hierarchical_analysis import run


def _make_global() -> pd.DataFrame:
    rows = []
    for wid in range(260):
        phase = "baseline" if wid < 220 else ("nrf_burst" if wid < 240 else "stable_recovery")
        label = 1 if phase == "nrf_burst" else 0
        burst = 15.0 if phase == "nrf_burst" else 2.0
        rows.append(
            {
                "window_id": wid,
                "window_start": float(wid * 2),
                "phase": phase,
                "cycle": 1 if phase != "baseline" else 0,
                "server_nf": "ALL",
                "request_count": int(burst * 2),
                "response_count": int(burst * 2),
                "observed_rps": burst + 0.05 * np.sin(wid),
                "response_rps": burst,
                "error_rate": 0.0,
                "authz_403_rate": 0.0,
                "http2_rst_rate": 0.0,
                "unique_streams": int(max(1, burst)),
                "latency_ms": 0.5,
                "latency_p95_ms": 0.8,
                "request_content_length_mean": 10.0,
                "tcp_payload_bytes": 1000.0 * burst,
                "nrf_register_rate": 0.0,
                "nrf_delete_rate": 0.0,
                "discovery_rate": 0.0,
                "nrf_query_rate": 12.0 if phase == "nrf_burst" else 0.0,
                "label": label,
                "req_norm": burst / 2.0,
                "streams_norm": burst / 2.0,
                "payload_norm": burst / 2.0,
            }
        )
    return pd.DataFrame(rows)


def _make_per_nf(global_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for nf in ("nrf", "scp"):
        for _, row in global_df.iterrows():
            x = row.copy()
            x["server_nf"] = nf
            if nf == "scp":
                x["observed_rps"] = 1.0
                x["unique_streams"] = 1
                x["tcp_payload_bytes"] = 500.0
                x["nrf_query_rate"] = 0.0
            rows.append(x)
    return pd.DataFrame(rows)


def test_hierarchical_analysis_uses_untouched_holdout(tmp_path: Path):
    global_df = _make_global()
    per_nf = _make_per_nf(global_df)
    gp = tmp_path / "global.csv"
    npf = tmp_path / "per_nf.csv"
    global_df.to_csv(gp, index=False)
    per_nf.to_csv(npf, index=False)

    summary = run(gp, npf, tmp_path / "out", target_fpr=0.01, seed=7)

    assert summary["baseline_split"]["holdout"] > 0
    assert summary["fusion_calibration_resolution"] < 0.05
    assert "GLOBAL" in summary["detectors"]
    assert any(x["phase"] == "nrf_burst" for x in summary["hierarchical_phase_metrics"])
    assert (tmp_path / "out" / "hierarchical_scored_windows.csv").exists()
