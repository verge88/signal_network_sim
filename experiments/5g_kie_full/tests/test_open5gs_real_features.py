from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from open5gs_real_features import build_features, load_endpoint_map, load_intervals, load_tshark_events


def test_real_open5gs_feature_bridge(tmp_path: Path):
    endpoints = tmp_path / "endpoints.json"
    endpoints.write_text(
        json.dumps(
            {
                "endpoints": [
                    {"nf": "nrf", "address": "127.0.0.10"},
                    {"nf": "udm", "address": "127.0.0.12"},
                ]
            }
        ),
        encoding="utf-8",
    )

    intervals = tmp_path / "intervals.csv"
    pd.DataFrame(
        [
            {"start_ts": 1000.0, "end_ts": 1004.0, "phase": "baseline", "cycle": 0},
            {"start_ts": 1004.0, "end_ts": 1008.0, "phase": "udm_restart", "cycle": 1},
        ]
    ).to_csv(intervals, index=False)

    events = tmp_path / "events.tsv"
    header = [
        "frame.time_epoch",
        "ip.src",
        "ip.dst",
        "tcp.stream",
        "tcp.flags.reset",
        "tcp.len",
        "http2.streamid",
        "http2.type",
        "http2.header.name",
        "http2.header.value",
    ]
    rows = [
        [
            "1000.5",
            "127.0.0.1",
            "127.0.0.10",
            "1",
            "0",
            "50",
            "1",
            "1",
            ":method,:path",
            "GET,/nnrf-nfm/v1/nf-instances",
        ],
        [
            "1000.6",
            "127.0.0.10",
            "127.0.0.1",
            "1",
            "0",
            "40",
            "1",
            "1",
            ":status,content-length",
            "200,100",
        ],
        [
            "1004.5",
            "127.0.0.1",
            "127.0.0.10",
            "2",
            "0",
            "60",
            "3",
            "1",
            ":method,:path",
            "PUT,/nnrf-nfm/v1/nf-instances/x",
        ],
        [
            "1004.7",
            "127.0.0.10",
            "127.0.0.1",
            "2",
            "0",
            "40",
            "3",
            "1",
            ":status,content-length",
            "201,0",
        ],
    ]
    with events.open("w", encoding="utf-8") as fh:
        fh.write("\t".join(header) + "\n")
        for row in rows:
            fh.write("\t".join(row) + "\n")

    parsed = load_tshark_events(events)
    endpoint_map = load_endpoint_map(endpoints)
    phase_intervals = load_intervals(intervals)
    global_df, per_nf_df = build_features(
        parsed,
        endpoint_map,
        phase_intervals,
        window_s=2.0,
    )

    baseline = global_df[global_df["phase"] == "baseline"].iloc[0]
    restart = global_df[global_df["phase"] == "udm_restart"].iloc[0]

    assert baseline["nrf_query_rate"] > 0
    assert baseline["latency_ms"] > 0
    assert restart["nrf_register_rate"] > 0
    assert restart["label"] == 1
    assert not per_nf_df.empty
