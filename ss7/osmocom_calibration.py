"""Calibrate SS7/SIGTRAN transport overhead from Osmocom M3UA DATA traffic.

The calibration is deliberately narrow and physically interpretable:
a controlled set of M3UA DATA payload sizes is sent through an independent
Osmocom stack.  Calibration payload sizes and holdout payload sizes are
separate.  The fitted parameter is per-message M3UA/SCTP/IP overhead used by
the canonical SS7 simulator's link-load and queueing model.

This does *not* fit traffic intensity, attack priors, MAP/TCAP/ISUP operation
mixes or operator workload distributions to a conformance test.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from .fix import MSG_BYTES, RngBook, SimConfig, Ss7Sim
except ImportError:
    from fix import MSG_BYTES, RngBook, SimConfig, Ss7Sim


SCHEMA = "ss7-osmocom-calibration-v2"
TRACE_COLUMNS = (
    "t", "frame_len", "ip_len", "src_port", "dst_port",
    "m3ua_len", "opc", "dpc", "si",
)
DEFAULT_PAYLOAD_SIZES = (24, 40, 56, 72, 92, 152, 192, 256)
DEFAULT_HOLDOUT_SIZES = (40, 72, 192, 256)


@dataclass(frozen=True)
class TraceStats:
    duration_s: float
    n_frames: int
    frames_per_s: float
    frame_len_p50: float
    frame_len_p95: float
    ip_len_p50: float
    ip_len_p95: float
    m3ua_len_p50: float
    m3ua_len_p95: float
    interarrival_p50_ms: float
    interarrival_p95_ms: float
    interarrival_cv: float


def _parse_int_list(value: str | Sequence[int]) -> Tuple[int, ...]:
    if isinstance(value, str):
        parts = [p.strip() for p in value.split(",") if p.strip()]
        vals = tuple(int(p) for p in parts)
    else:
        vals = tuple(int(v) for v in value)
    if not vals or any(v <= 0 for v in vals):
        raise ValueError("payload sizes must be positive integers")
    if len(set(vals)) != len(vals):
        raise ValueError("payload sizes must be unique")
    return vals


def _pad4(n: float) -> float:
    return float(4 * int(np.ceil(float(n) / 4.0)))


def load_trace(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in TRACE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"trace is missing columns: {missing}")
    out = df.loc[:, TRACE_COLUMNS].copy()
    for col in TRACE_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna().sort_values("t").reset_index(drop=True)
    out = out[(out["frame_len"] > 0) & (out["ip_len"] > 0)
              & (out["m3ua_len"] > 0)]
    if out.empty:
        raise ValueError("trace contains no usable controlled M3UA DATA frames")
    return out.reset_index(drop=True)


def summarize_trace(df: pd.DataFrame) -> TraceStats:
    t = df["t"].to_numpy(float)
    duration = max(float(t.max() - t.min()), 1e-9)
    diffs = np.diff(t)
    diffs = diffs[diffs > 0]
    if diffs.size:
        mean_dt = float(diffs.mean())
        cv = float(diffs.std(ddof=0) / mean_dt) if mean_dt > 0 else 0.0
        p50_dt = float(np.quantile(diffs, 0.50) * 1000.0)
        p95_dt = float(np.quantile(diffs, 0.95) * 1000.0)
    else:
        cv = p50_dt = p95_dt = 0.0

    def q(col: str, p: float) -> float:
        return float(np.quantile(df[col].to_numpy(float), p))

    return TraceStats(
        duration_s=duration,
        n_frames=int(len(df)),
        frames_per_s=float(len(df) / duration),
        frame_len_p50=q("frame_len", 0.50),
        frame_len_p95=q("frame_len", 0.95),
        ip_len_p50=q("ip_len", 0.50),
        ip_len_p95=q("ip_len", 0.95),
        m3ua_len_p50=q("m3ua_len", 0.50),
        m3ua_len_p95=q("m3ua_len", 0.95),
        interarrival_p50_ms=p50_dt,
        interarrival_p95_ms=p95_dt,
        interarrival_cv=cv,
    )


def infer_payload_sizes(
    df: pd.DataFrame,
    payload_sizes: Sequence[int] = DEFAULT_PAYLOAD_SIZES,
) -> Tuple[pd.DataFrame, float]:
    """Map controlled M3UA message lengths back to known generated payloads.

    The GitHub Action generates payload sizes that are all 32-bit aligned.
    Hence every controlled DATA message has:
        m3ua_len = payload_bytes + constant_m3ua_overhead.
    Pairing sorted unique lengths with sorted generated sizes makes the
    inference independent of send order and retransmissions.
    """
    expected = tuple(sorted(_parse_int_list(payload_sizes)))
    observed = tuple(sorted(int(v) for v in df["m3ua_len"].unique()))
    if len(observed) != len(expected):
        raise ValueError(
            "controlled trace has unexpected M3UA length cardinality: "
            f"observed={observed}, expected {len(expected)} sizes={expected}"
        )

    offsets = np.asarray(observed, dtype=float) - np.asarray(expected, dtype=float)
    if float(offsets.max() - offsets.min()) > 1e-9:
        raise ValueError(
            "M3UA length is not a constant-overhead mapping for controlled "
            f"payloads: offsets={offsets.tolist()}"
        )
    m3ua_overhead = float(np.median(offsets))
    if m3ua_overhead <= 0:
        raise ValueError("inferred M3UA overhead must be positive")

    mapping = dict(zip(observed, expected))
    out = df.copy()
    out["payload_bytes"] = out["m3ua_len"].round().astype(int).map(mapping)
    if out["payload_bytes"].isna().any():
        raise ValueError("failed to map all controlled M3UA lengths to payloads")
    out["payload_bytes"] = out["payload_bytes"].astype(int)
    return out, m3ua_overhead


def size_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("payload_bytes", as_index=False)
        .agg(
            n_frames=("t", "size"),
            m3ua_len_median=("m3ua_len", "median"),
            ip_len_median=("ip_len", "median"),
            frame_len_median=("frame_len", "median"),
            ip_len_p95=("ip_len", lambda s: float(np.quantile(s, 0.95))),
        )
        .sort_values("payload_bytes")
        .reset_index(drop=True)
    )


def fit_transport_profile(
    annotated: pd.DataFrame,
    holdout_sizes: Sequence[int] = DEFAULT_HOLDOUT_SIZES,
    *,
    min_frames_per_size: int = 10,
) -> Tuple[Dict, pd.DataFrame]:
    holdout_set = set(_parse_int_list(holdout_sizes))
    summary = size_summary(annotated)
    all_sizes = set(int(v) for v in summary["payload_bytes"])
    if not holdout_set < all_sizes:
        raise ValueError(
            "holdout sizes must be a non-empty proper subset of observed sizes"
        )

    if int(summary["n_frames"].min()) < int(min_frames_per_size):
        raise ValueError(
            f"need >= {min_frames_per_size} frames for every payload size; "
            f"minimum is {int(summary['n_frames'].min())}"
        )

    train = summary[~summary["payload_bytes"].isin(holdout_set)].copy()
    hold = summary[summary["payload_bytes"].isin(holdout_set)].copy()
    if len(train) < 3 or len(hold) < 2:
        raise ValueError("need at least 3 calibration and 2 holdout payload sizes")

    # Physically constrained fit: IP packet bytes = aligned MTP3-user payload
    # + fixed M3UA/SCTP/IPv4 encapsulation overhead.
    train_overheads = (
        train["ip_len_median"].to_numpy(float)
        - train["payload_bytes"].map(_pad4).to_numpy(float)
    )
    transport_overhead = float(np.median(train_overheads))

    m3ua_overheads = (
        train["m3ua_len_median"].to_numpy(float)
        - train["payload_bytes"].map(_pad4).to_numpy(float)
    )
    m3ua_overhead = float(np.median(m3ua_overheads))
    sctp_ip_overhead = float(transport_overhead - m3ua_overhead)

    hold["predicted_ip_len"] = (
        hold["payload_bytes"].map(_pad4) + transport_overhead
    )
    hold["abs_error_bytes"] = (
        hold["predicted_ip_len"] - hold["ip_len_median"]
    ).abs()
    hold["relative_error"] = (
        hold["abs_error_bytes"] / hold["ip_len_median"].clip(lower=1.0)
    )

    app_values = np.asarray(list(MSG_BYTES.values()), dtype=float)
    app_mean = float(app_values.mean())
    effective_mean_wire = float(
        np.mean([_pad4(v) + transport_overhead for v in app_values])
    )

    cfg = SimConfig(seed=42)
    base = Ss7Sim(cfg, RngBook(cfg.seed), size_capacities=False)
    calibrated = Ss7Sim(cfg, RngBook(cfg.seed), size_capacities=False)
    calibrated.transport_overhead_bytes = transport_overhead
    calibrated.mean_msg_bits = calibrated._mean_wire_message_bits()
    base_off = max(base._nominal_offered().values())
    cal_off = max(calibrated._nominal_offered().values())

    profile = {
        "schema": SCHEMA,
        "scope": "controlled M3UA DATA encapsulation calibration",
        "source": "Osmocom OsmoSTP + nplab/m3ua-testtool",
        "transport_overhead_bytes": transport_overhead,
        "m3ua_overhead_bytes": m3ua_overhead,
        "sctp_ipv4_overhead_bytes": sctp_ip_overhead,
        "canonical_application_mean_bytes": app_mean,
        "effective_mean_wire_bytes": effective_mean_wire,
        "effective_mean_message_bits": effective_mean_wire * 8.0,
        "wire_bytes_multiplier": effective_mean_wire / max(app_mean, 1e-12),
        "canonical_busiest_link_bps_before": float(base_off),
        "canonical_busiest_link_bps_after": float(cal_off),
        "canonical_busiest_link_load_multiplier": float(
            cal_off / max(base_off, 1e-12)
        ),
        "calibration_payload_sizes": [
            int(v) for v in train["payload_bytes"].tolist()
        ],
        "holdout_payload_sizes": [
            int(v) for v in hold["payload_bytes"].tolist()
        ],
        "validation_mae_bytes": float(hold["abs_error_bytes"].mean()),
        "validation_max_abs_error_bytes": float(hold["abs_error_bytes"].max()),
        "validation_median_relative_error": float(
            hold["relative_error"].median()
        ),
        "untouched_model_families": [
            "traffic-rate priors",
            "MAP/TCAP/ISUP operation mix",
            "attack priors and attack semantics",
            "measurement/reporting noise",
            "diurnal amplitude",
            "AR traffic dynamics",
            "integrity/key-custody model",
        ],
    }
    return profile, hold.reset_index(drop=True)


def calibrated_template(
    cfg: SimConfig,
    profile: Dict,
    *,
    seed: int | None = None,
) -> Ss7Sim:
    fit = profile.get("fit", profile)
    overhead = float(fit["transport_overhead_bytes"])
    if not np.isfinite(overhead) or overhead <= 0:
        raise ValueError("transport_overhead_bytes must be finite and positive")
    run_seed = cfg.seed if seed is None else int(seed)
    sim = Ss7Sim(cfg, RngBook(run_seed), size_capacities=False)
    sim.transport_overhead_bytes = overhead
    sim.mean_msg_bits = sim._mean_wire_message_bits()
    return sim


def run_calibration(
    trace_path: str | Path,
    out_dir: str | Path,
    *,
    payload_sizes: Sequence[int] = DEFAULT_PAYLOAD_SIZES,
    holdout_sizes: Sequence[int] = DEFAULT_HOLDOUT_SIZES,
    min_frames_per_size: int = 10,
) -> Dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    raw = load_trace(trace_path)
    annotated, inferred_m3ua_overhead = infer_payload_sizes(raw, payload_sizes)
    fit, holdout = fit_transport_profile(
        annotated,
        holdout_sizes,
        min_frames_per_size=min_frames_per_size,
    )
    # Cross-check the direct message-length inference against the grouped fit.
    if abs(float(fit["m3ua_overhead_bytes"]) - inferred_m3ua_overhead) > 1e-9:
        raise RuntimeError("inconsistent inferred M3UA overhead")

    summary = size_summary(annotated)
    holdout_set = set(_parse_int_list(holdout_sizes))
    result = {
        "schema": SCHEMA,
        "method": {
            "design": "controlled payload-size calibration with size holdout",
            "payload_sizes": [int(v) for v in _parse_int_list(payload_sizes)],
            "holdout_sizes": [int(v) for v in _parse_int_list(holdout_sizes)],
            "min_frames_per_size": int(min_frames_per_size),
        },
        "fit": fit,
        "trace": asdict(summarize_trace(annotated)),
        "claim_boundary": (
            "This calibrates M3UA/SCTP/IPv4 encapsulation overhead using "
            "controlled DATA messages handled by an independent Osmocom "
            "implementation. It is not operator traffic and does not calibrate "
            "traffic-rate or MAP/TCAP/ISUP workload distributions."
        ),
    }

    (out / "calibrated_profile.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    annotated.to_csv(out / "controlled_data_trace.csv", index=False)
    summary["split"] = np.where(
        summary["payload_bytes"].isin(holdout_set), "holdout", "calibration"
    )
    summary.to_csv(out / "payload_size_summary.csv", index=False)
    holdout.to_csv(out / "holdout_validation.csv", index=False)

    md = f"""# SS7 / Osmocom controlled transport calibration

- Source: Osmocom OsmoSTP + nplab M3UA test tool
- Controlled DATA frames: {len(annotated)}
- Payload sizes: {', '.join(map(str, result['method']['payload_sizes']))} bytes
- Calibration sizes: {', '.join(map(str, fit['calibration_payload_sizes']))}
- Holdout sizes: {', '.join(map(str, fit['holdout_payload_sizes']))}
- Fitted M3UA overhead: {fit['m3ua_overhead_bytes']:.1f} bytes/message
- Fitted SCTP+IPv4 overhead: {fit['sctp_ipv4_overhead_bytes']:.1f} bytes/message
- Total transport overhead applied to simulator: {fit['transport_overhead_bytes']:.1f} bytes/message
- Holdout MAE: {fit['validation_mae_bytes']:.3f} bytes
- Holdout max absolute error: {fit['validation_max_abs_error_bytes']:.3f} bytes
- Holdout median relative error: {fit['validation_median_relative_error']:.6f}
- Canonical busiest-link load multiplier after calibration: {fit['canonical_busiest_link_load_multiplier']:.3f}x

**Claim boundary:** {result['claim_boundary']}
"""
    (out / "summary.md").write_text(md, encoding="utf-8")
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Calibrate canonical SS7 transport overhead from Osmocom M3UA DATA"
    )
    ap.add_argument("--trace", required=True)
    ap.add_argument("--out", default="ss7_calibration_osmocom")
    ap.add_argument(
        "--payload-sizes",
        default=",".join(map(str, DEFAULT_PAYLOAD_SIZES)),
    )
    ap.add_argument(
        "--holdout-sizes",
        default=",".join(map(str, DEFAULT_HOLDOUT_SIZES)),
    )
    ap.add_argument("--min-frames-per-size", type=int, default=10)
    args = ap.parse_args(argv)

    result = run_calibration(
        args.trace,
        args.out,
        payload_sizes=_parse_int_list(args.payload_sizes),
        holdout_sizes=_parse_int_list(args.holdout_sizes),
        min_frames_per_size=args.min_frames_per_size,
    )
    fit = result["fit"]
    print(json.dumps({
        "profile": str(Path(args.out) / "calibrated_profile.json"),
        "transport_overhead_bytes": fit["transport_overhead_bytes"],
        "validation_mae_bytes": fit["validation_mae_bytes"],
        "validation_max_abs_error_bytes": fit["validation_max_abs_error_bytes"],
        "validation_median_relative_error": fit[
            "validation_median_relative_error"
        ],
        "canonical_busiest_link_load_multiplier": fit[
            "canonical_busiest_link_load_multiplier"
        ],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
