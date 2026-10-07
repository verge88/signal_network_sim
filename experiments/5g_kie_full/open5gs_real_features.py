from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd


RAW_COLUMNS = [
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


def _split_multi(value: object) -> list[str]:
    if value is None:
        return []
    text = str(value)
    if text == "" or text.lower() == "nan":
        return []
    return [x.strip() for x in text.split(",")]


def _headers(names: object, values: object) -> dict[str, str]:
    ns = _split_multi(names)
    vs = _split_multi(values)
    out: dict[str, str] = {}
    for idx, name in enumerate(ns):
        if not name:
            continue
        value = vs[idx] if idx < len(vs) else ""
        out[name.lower()] = value
    return out


def load_tshark_events(path: Path) -> pd.DataFrame:
    """Load tshark field export and recover HTTP/2 pseudo-headers."""
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    for col in RAW_COLUMNS:
        if col not in df.columns:
            df[col] = ""

    parsed = []
    for _, row in df.iterrows():
        h = _headers(row["http2.header.name"], row["http2.header.value"])
        parsed.append(
            {
                "ts": pd.to_numeric(row["frame.time_epoch"], errors="coerce"),
                "ip_src": row["ip.src"],
                "ip_dst": row["ip.dst"],
                "tcp_stream": row["tcp.stream"],
                "tcp_reset": row["tcp.flags.reset"] == "1",
                "tcp_len": pd.to_numeric(row["tcp.len"], errors="coerce"),
                "h2_stream": _split_multi(row["http2.streamid"])[0]
                if _split_multi(row["http2.streamid"])
                else "",
                "h2_type": _split_multi(row["http2.type"])[0]
                if _split_multi(row["http2.type"])
                else "",
                "method": h.get(":method", ""),
                "path": h.get(":path", ""),
                "status": h.get(":status", ""),
                "content_length": h.get("content-length", ""),
            }
        )

    out = pd.DataFrame(parsed).dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    out["tcp_len"] = pd.to_numeric(out["tcp_len"], errors="coerce").fillna(0.0)
    out["status_num"] = pd.to_numeric(out["status"], errors="coerce")
    out["content_length_num"] = pd.to_numeric(out["content_length"], errors="coerce")
    out["is_request"] = out["method"].str.len().gt(0)
    out["is_response"] = out["status"].str.len().gt(0)
    out["is_h2_reset"] = out["h2_type"].eq("3")
    return out


def load_endpoint_map(path: Path) -> Dict[str, str]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(row["address"]): str(row.get("nf", "unknown")).lower()
        for row in doc.get("endpoints", [])
        if row.get("address")
    }


def load_intervals(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"start_ts", "end_ts", "phase", "cycle"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"phase interval file missing columns: {sorted(missing)}")
    df["start_ts"] = pd.to_numeric(df["start_ts"], errors="raise")
    df["end_ts"] = pd.to_numeric(df["end_ts"], errors="raise")
    return df.sort_values("start_ts").reset_index(drop=True)


def _phase_at(ts: float, intervals: pd.DataFrame) -> tuple[str, int]:
    hit = intervals[(intervals["start_ts"] <= ts) & (ts < intervals["end_ts"])]
    if hit.empty:
        return "unlabeled", -1
    row = hit.iloc[0]
    return str(row["phase"]), int(row["cycle"])


def _request_response_latency(events: pd.DataFrame, endpoint_map: Dict[str, str]) -> pd.DataFrame:
    req = events[events["is_request"]].copy()
    resp = events[events["is_response"]].copy()
    if req.empty or resp.empty:
        return pd.DataFrame(columns=["ts", "server_nf", "latency_ms"])

    req["key"] = req["tcp_stream"].astype(str) + "|" + req["h2_stream"].astype(str)
    resp["key"] = resp["tcp_stream"].astype(str) + "|" + resp["h2_stream"].astype(str)
    req = req[req["h2_stream"].astype(str).str.len().gt(0)].sort_values("ts")
    resp = resp[resp["h2_stream"].astype(str).str.len().gt(0)].sort_values("ts")
    req = req.drop_duplicates("key", keep="first")
    resp = resp.drop_duplicates("key", keep="first")

    merged = req[["key", "ts", "ip_dst"]].merge(
        resp[["key", "ts"]].rename(columns={"ts": "resp_ts"}),
        on="key",
        how="inner",
    )
    merged = merged[merged["resp_ts"] >= merged["ts"]].copy()
    merged["latency_ms"] = (merged["resp_ts"] - merged["ts"]) * 1000.0
    merged["server_nf"] = merged["ip_dst"].map(endpoint_map).fillna("unknown")
    return merged[["ts", "server_nf", "latency_ms"]]


def _normalize(frame: pd.DataFrame, groupwise: bool) -> pd.DataFrame:
    frame = frame.copy()
    baseline = frame[frame["phase"] == "baseline"].copy()

    mapping = {
        "observed_rps": "req_norm",
        "unique_streams": "streams_norm",
        "tcp_payload_bytes": "payload_norm",
    }
    for source, target in mapping.items():
        if groupwise:
            medians = baseline.groupby("server_nf")[source].median().to_dict()
            vals = []
            for nf, value in zip(frame["server_nf"], frame[source]):
                base = float(medians.get(nf, 0.0))
                vals.append(float(value) / max(base, 1e-9))
            frame[target] = vals
        else:
            base = float(baseline[source].median())
            frame[target] = frame[source] / max(base, 1e-9)
    return frame


def build_features(
    events: pd.DataFrame,
    endpoint_map: Dict[str, str],
    intervals: pd.DataFrame,
    window_s: float = 2.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if events.empty:
        raise ValueError("no HTTP/2/TCP events supplied")

    origin = float(intervals["start_ts"].min())
    end = float(intervals["end_ts"].max())
    work = events[(events["ts"] >= origin) & (events["ts"] < end)].copy()
    if work.empty:
        raise ValueError("no events fall inside experiment phase intervals")

    work["window_id"] = np.floor((work["ts"] - origin) / window_s).astype(int)
    work["server_nf"] = np.where(
        work["is_request"],
        work["ip_dst"].map(endpoint_map),
        work["ip_src"].map(endpoint_map),
    )
    work["server_nf"] = pd.Series(work["server_nf"]).fillna("unknown").astype(str)

    latency = _request_response_latency(work, endpoint_map)
    if not latency.empty:
        latency["window_id"] = np.floor((latency["ts"] - origin) / window_s).astype(int)

    max_window = int(np.ceil((end - origin) / window_s)) - 1
    nfs = sorted(set(endpoint_map.values()))
    global_rows: list[dict] = []
    per_nf_rows: list[dict] = []

    def row_for(wid: int, server_nf: str | None) -> dict:
        start = origin + wid * window_s
        phase, cycle = _phase_at(start + window_s / 2.0, intervals)

        window = work[work["window_id"] == wid]
        if server_nf is not None:
            window = window[window["server_nf"] == server_nf]

        req = window[window["is_request"]]
        resp = window[window["is_response"]]
        wl = latency[latency["window_id"] == wid] if not latency.empty else latency
        if server_nf is not None and not wl.empty:
            wl = wl[wl["server_nf"] == server_nf]

        status = resp["status_num"].dropna()
        paths = req["path"].fillna("")
        methods = req["method"].str.upper()
        nfm = paths.str.contains("/nnrf-nfm/", regex=False)
        disc = paths.str.contains("/nnrf-disc/", regex=False)
        streams = req["tcp_stream"].astype(str) + "|" + req["h2_stream"].astype(str)
        lats = wl["latency_ms"].dropna() if not wl.empty else pd.Series(dtype=float)
        content = req["content_length_num"].dropna()
        resets = int(window["tcp_reset"].sum() + window["is_h2_reset"].sum())

        return {
            "window_id": wid,
            "window_start": start,
            "phase": phase,
            "cycle": cycle,
            "server_nf": server_nf or "ALL",
            "request_count": int(len(req)),
            "response_count": int(len(resp)),
            "observed_rps": float(len(req) / window_s),
            "response_rps": float(len(resp) / window_s),
            "error_rate": float((status >= 400).sum() / max(len(resp), 1)),
            "authz_403_rate": float((status == 403).sum() / max(len(resp), 1)),
            "http2_rst_rate": float(resets / max(len(window), 1)),
            "unique_streams": int(streams.nunique()),
            "latency_ms": float(lats.median()) if len(lats) else np.nan,
            "latency_p95_ms": float(lats.quantile(0.95)) if len(lats) else np.nan,
            "request_content_length_mean": float(content.mean()) if len(content) else 0.0,
            "tcp_payload_bytes": float(window["tcp_len"].sum()),
            "nrf_register_rate": float((nfm & methods.eq("PUT")).sum() / window_s),
            "nrf_delete_rate": float((nfm & methods.eq("DELETE")).sum() / window_s),
            "discovery_rate": float(disc.sum() / window_s),
            "nrf_query_rate": float(
                (paths.str.contains("/nnrf-", regex=False) & methods.eq("GET")).sum()
                / window_s
            ),
            "label": 1 if phase in {"udm_restart", "nrf_burst"} else 0,
        }

    for wid in range(max_window + 1):
        global_rows.append(row_for(wid, None))
        for nf in nfs:
            per_nf_rows.append(row_for(wid, nf))

    return _normalize(pd.DataFrame(global_rows), False), _normalize(
        pd.DataFrame(per_nf_rows), True
    )


def feature_availability() -> dict:
    return {
        "directly_observable_from_real_sbi_capture": [
            "observed_rps",
            "error_rate",
            "http2_rst_rate",
            "latency_ms",
            "latency_p95_ms",
            "streams_norm",
            "authz_403_rate",
            "discovery_rate",
            "nrf_register_rate",
            "nrf_delete_rate",
            "nrf_query_rate",
        ],
        "not_available_without_additional_instrumentation": [
            "report_rps",
            "report_observed_gap",
            "kie_rtt_ms",
            "kie_loss",
            "kie_auth_fail",
            "heartbeat_age_s",
            "cross_slice_rate",
            "scope_mismatch_rate",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build real Open5GS SBI feature windows")
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--endpoints", type=Path, required=True)
    parser.add_argument("--intervals", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--window-s", type=float, default=2.0)
    args = parser.parse_args()

    events = load_tshark_events(args.events)
    endpoint_map = load_endpoint_map(args.endpoints)
    intervals = load_intervals(args.intervals)
    global_df, per_nf_df = build_features(
        events,
        endpoint_map,
        intervals,
        window_s=args.window_s,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    global_df.to_csv(args.output_dir / "real_sbi_global_features.csv", index=False)
    per_nf_df.to_csv(args.output_dir / "real_sbi_per_nf_features.csv", index=False)
    (args.output_dir / "real_feature_availability.json").write_text(
        json.dumps(feature_availability(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"events={len(events)}")
    print(f"global_windows={len(global_df)}")
    print(global_df.groupby("phase").size().to_string())
    print(f"per_nf_rows={len(per_nf_df)}")


if __name__ == "__main__":
    main()
