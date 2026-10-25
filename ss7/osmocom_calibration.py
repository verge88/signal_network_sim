"""SS7 calibration from an independent Osmocom M3UA/SCTP trace.

Only transport/control quantities observable in the Osmocom software-in-the-loop
run are calibrated. MAP/TCAP/ISUP workload semantics are intentionally not
inferred from an M3UA conformance trace. The trace is split chronologically:
the first part is used for fitting and the last part is a holdout.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd

try:
    from .fix import RngBook, SimConfig, Ss7Sim
except ImportError:
    from fix import RngBook, SimConfig, Ss7Sim


TRACE_COLUMNS = ("t", "frame_len", "src_port", "dst_port", "info")
SCHEMA = "ss7-osmocom-calibration-v1"


@dataclass(frozen=True)
class TraceStats:
    duration_s: float
    n_frames: int
    frames_per_s: float
    bytes_per_s: float
    frame_len_p50: float
    frame_len_p95: float
    interarrival_p50_ms: float
    interarrival_p95_ms: float
    interarrival_cv: float
    data_frames: int
    data_frames_per_s: float
    control_frames: int
    control_share: float


def load_trace(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in TRACE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"trace is missing columns: {missing}")
    out = df.loc[:, TRACE_COLUMNS].copy()
    for col in ("t", "frame_len", "src_port", "dst_port"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["info"] = out["info"].fillna("").astype(str)
    out = out.dropna(subset=["t", "frame_len"]).sort_values("t")
    out = out[out["frame_len"] > 0].reset_index(drop=True)
    if out.empty:
        raise ValueError("trace contains no usable M3UA/SCTP frames")
    return out


def frame_kind(info: str) -> str:
    text = str(info).upper()
    if "DATA" in text or "PAYLOAD" in text:
        return "data"
    if any(token in text for token in (
        "ASPUP", "ASPAC", "ASPDN", "ASPIA", "BEAT", "NOTIFY",
        "ERROR", "DAUD", "DAVA", "DUNA", "SCON", "DUPU", "DRST",
    )):
        return "control"
    return "other"


def split_trace(df: pd.DataFrame, train_fraction: float = 0.60
                ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not 0.25 <= train_fraction <= 0.80:
        raise ValueError("train_fraction must be in [0.25, 0.80]")
    if len(df) < 20:
        raise ValueError("at least 20 M3UA/SCTP frames are required")
    t0, t1 = float(df["t"].min()), float(df["t"].max())
    cut = t0 + (t1 - t0) * train_fraction
    train = df[df["t"] <= cut].copy()
    holdout = df[df["t"] > cut].copy()
    if len(train) < 8 or len(holdout) < 8:
        idx = max(8, min(len(df) - 8, int(round(len(df) * train_fraction))))
        train, holdout = df.iloc[:idx].copy(), df.iloc[idx:].copy()
    if train.empty or holdout.empty:
        raise ValueError("trace cannot be split into calibration and holdout")
    return train.reset_index(drop=True), holdout.reset_index(drop=True)


def summarize_trace(df: pd.DataFrame) -> TraceStats:
    t = df["t"].to_numpy(float)
    lengths = df["frame_len"].to_numpy(float)
    duration = max(float(t.max() - t.min()), 1e-9)
    diffs = np.diff(t)
    diffs = diffs[diffs > 0]
    kinds = df["info"].map(frame_kind)
    n_data = int((kinds == "data").sum())
    n_control = int((kinds == "control").sum())
    if diffs.size:
        mean_dt = float(diffs.mean())
        cv = float(diffs.std(ddof=0) / mean_dt) if mean_dt > 0 else 0.0
        p50_dt = float(np.quantile(diffs, 0.50) * 1000.0)
        p95_dt = float(np.quantile(diffs, 0.95) * 1000.0)
    else:
        cv = p50_dt = p95_dt = 0.0
    return TraceStats(
        duration_s=duration,
        n_frames=int(len(df)),
        frames_per_s=float(len(df) / duration),
        bytes_per_s=float(lengths.sum() / duration),
        frame_len_p50=float(np.quantile(lengths, 0.50)),
        frame_len_p95=float(np.quantile(lengths, 0.95)),
        interarrival_p50_ms=p50_dt,
        interarrival_p95_ms=p95_dt,
        interarrival_cv=cv,
        data_frames=n_data,
        data_frames_per_s=float(n_data / duration),
        control_frames=n_control,
        control_share=float(n_control / max(1, len(df))),
    )


def _rate_basis(df: pd.DataFrame) -> Tuple[str, pd.DataFrame]:
    kinds = df["info"].map(frame_kind)
    data = df[kinds == "data"]
    if len(data) >= max(5, int(math.ceil(0.05 * len(df)))):
        return "m3ua_data_frames", data
    return "all_m3ua_sctp_frames", df


def trace_rate_series(df: pd.DataFrame, window_s: float = 1.0,
                      basis: str = "all_m3ua_sctp_frames") -> np.ndarray:
    if window_s <= 0:
        raise ValueError("window_s must be positive")
    selected = df
    if basis == "m3ua_data_frames":
        selected = df[df["info"].map(frame_kind) == "data"]
    if selected.empty:
        return np.zeros(0, dtype=float)
    t = selected["t"].to_numpy(float)
    start = float(df["t"].min())
    stop = float(df["t"].max())
    n = max(1, int(math.floor((stop - start) / window_s)) + 1)
    idx = np.floor((t - start) / window_s).astype(int)
    idx = np.clip(idx, 0, n - 1)
    return np.bincount(idx, minlength=n).astype(float) / window_s


def _base_busiest_link_rate(cfg: SimConfig, seed: int) -> Tuple[float, Ss7Sim]:
    sim = Ss7Sim(cfg, RngBook(seed), size_capacities=False)
    offered = sim._nominal_offered()
    if not offered or sim.mean_msg_bits <= 0:
        raise RuntimeError("canonical SS7 model has no offered load")
    return float(max(offered.values()) / sim.mean_msg_bits), sim


def _reference_capacity(rate_hz: float, mean_msg_bits: float,
                        target_rho: float) -> Dict[str, float | int]:
    if not 0.05 < target_rho < 0.95:
        raise ValueError("target_rho must be in (0.05, 0.95)")
    payload_bps = max(0.0, rate_hz * mean_msg_bits)
    required_bps = payload_bps / target_rho
    n64 = max(1, int(math.ceil(required_bps / 64_000.0)))
    if n64 <= 16:
        link_kbps, n_links = 64.0, n64
    else:
        link_kbps = 2048.0
        n_links = max(1, int(math.ceil(required_bps / 2_048_000.0)))
    return {
        "reference_payload_bps": float(payload_bps),
        "target_rho": float(target_rho),
        "recommended_link_kbps": float(link_kbps),
        "recommended_n_links": int(n_links),
        "recommended_capacity_bps": float(link_kbps * 1000.0 * n_links),
    }


def fit_profile(train: pd.DataFrame, cfg: SimConfig, *, seed: int = 42,
                target_rho: float = 0.50) -> Dict:
    stats = summarize_trace(train)
    basis, selected = _rate_basis(train)
    selected_duration = max(
        float(selected["t"].max() - selected["t"].min()), 1e-9
    )
    observed_rate = float(len(selected) / selected_duration)
    base_rate, sim = _base_busiest_link_rate(cfg, seed)
    flow_scale = observed_rate / max(base_rate, 1e-12)
    capacity = _reference_capacity(observed_rate, sim.mean_msg_bits, target_rho)
    return {
        "schema": SCHEMA,
        "scope": "M3UA/SCTP software-in-the-loop transport/control calibration",
        "source": "Osmocom OsmoSTP + nplab/m3ua-testtool",
        "rate_basis": basis,
        "observed_train_rate_hz": observed_rate,
        "canonical_reference_rate_hz": base_rate,
        "flow_scale": float(flow_scale),
        "canonical_mean_message_bits": float(sim.mean_msg_bits),
        **capacity,
        "train_trace": asdict(stats),
        "untouched_model_families": [
            "MAP/TCAP/ISUP operation mix",
            "attack priors and attack semantics",
            "measurement/reporting noise",
            "diurnal amplitude",
            "AR traffic dynamics",
            "integrity/key-custody model",
        ],
    }


def calibrated_template(cfg: SimConfig, profile: Dict, *,
                        seed: int | None = None) -> Ss7Sim:
    """Return canonical Ss7Sim with only the fitted traffic scale applied."""
    fit = profile.get("fit", profile)
    scale = float(fit["flow_scale"])
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("flow_scale must be finite and positive")
    run_seed = cfg.seed if seed is None else int(seed)
    sim = Ss7Sim(cfg, RngBook(run_seed), size_capacities=False)
    sim.flows = [replace(flow, rate=float(flow.rate * scale))
                 for flow in sim.flows]
    return sim


def synthetic_rate_series(cfg: SimConfig, profile: Dict, *, seed: int = 42,
                          n_windows: int = 40) -> np.ndarray:
    sim = calibrated_template(cfg, profile, seed=seed)
    out = []
    for t in range(max(1, int(n_windows))):
        _, _, meta = sim.step_true(t, [])
        offered = meta.get("offered", {})
        rate = (max(offered.values()) / sim.mean_msg_bits
                if offered and sim.mean_msg_bits > 0 else 0.0)
        out.append(float(rate))
    return np.asarray(out, dtype=float)


def _ks_statistic(a: Iterable[float], b: Iterable[float]) -> float:
    aa = np.sort(np.asarray(list(a), dtype=float))
    bb = np.sort(np.asarray(list(b), dtype=float))
    aa = aa[np.isfinite(aa)]
    bb = bb[np.isfinite(bb)]
    if aa.size == 0 or bb.size == 0:
        return float("nan")
    grid = np.union1d(aa, bb)
    fa = np.searchsorted(aa, grid, side="right") / aa.size
    fb = np.searchsorted(bb, grid, side="right") / bb.size
    return float(np.max(np.abs(fa - fb)))


def comparison_table(real_rate: np.ndarray,
                     synthetic_rate: np.ndarray) -> pd.DataFrame:
    def q(x: np.ndarray, p: float) -> float:
        return float(np.quantile(x, p)) if x.size else float("nan")
    real_med = q(real_rate, 0.50)
    syn_med = q(synthetic_rate, 0.50)
    return pd.DataFrame([{
        "quantity": "reference_link_frames_per_s",
        "real_p10": q(real_rate, 0.10),
        "real_p50": real_med,
        "real_p90": q(real_rate, 0.90),
        "synthetic_p10": q(synthetic_rate, 0.10),
        "synthetic_p50": syn_med,
        "synthetic_p90": q(synthetic_rate, 0.90),
        "ks_statistic": _ks_statistic(real_rate, synthetic_rate),
        "relative_median_error": (
            abs(syn_med - real_med) / max(abs(real_med), 1e-12)
        ),
    }])


def run_calibration(trace_path: str | Path, out_dir: str | Path, *,
                    seed: int = 42, train_fraction: float = 0.60,
                    window_s: float = 1.0, target_rho: float = 0.50,
                    max_validation_windows: int = 80,
                    min_frames: int = 20) -> Dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    trace = load_trace(trace_path)
    if len(trace) < min_frames:
        raise ValueError(
            f"need at least {min_frames} usable frames, got {len(trace)}"
        )
    train, holdout = split_trace(trace, train_fraction)
    cfg = SimConfig(seed=seed)
    fit = fit_profile(train, cfg, seed=seed, target_rho=target_rho)
    basis = fit["rate_basis"]
    real_rate = trace_rate_series(holdout, window_s=window_s, basis=basis)
    n_syn = min(max_validation_windows, max(10, int(real_rate.size)))
    syn_rate = synthetic_rate_series(cfg, fit, seed=seed + 1, n_windows=n_syn)
    comparison = comparison_table(real_rate, syn_rate)

    result = {
        "schema": SCHEMA,
        "method": {
            "split": "chronological",
            "train_fraction": float(train_fraction),
            "holdout_fraction": float(1.0 - train_fraction),
            "window_s": float(window_s),
            "seed": int(seed),
        },
        "fit": fit,
        "holdout_trace": asdict(summarize_trace(holdout)),
        "validation": comparison.iloc[0].to_dict(),
        "claim_boundary": (
            "This is software-in-the-loop validation against an independent "
            "Osmocom M3UA/SCTP implementation, not operator traffic and not "
            "a calibration of MAP/TCAP/ISUP workload semantics."
        ),
    }
    (out / "calibrated_profile.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    comparison.to_csv(out / "holdout_comparison.csv", index=False)
    train.to_csv(out / "calibration_trace.csv", index=False)
    holdout.to_csv(out / "holdout_trace.csv", index=False)

    v = result["validation"]
    summary = f"""# SS7 / Osmocom calibration

- Source: Osmocom OsmoSTP + nplab M3UA test tool
- Scope: M3UA/SCTP software-in-the-loop transport/control calibration
- Split: first {train_fraction:.0%} fit, final {1-train_fraction:.0%} holdout
- Rate basis: {fit['rate_basis']}
- Observed fit rate: {fit['observed_train_rate_hz']:.3f} frames/s
- Canonical reference rate: {fit['canonical_reference_rate_hz']:.3f} messages/s
- Fitted traffic scale: {fit['flow_scale']:.6f}
- Holdout KS statistic: {v['ks_statistic']:.4f}
- Holdout median relative error: {v['relative_median_error']:.4f}
- Reference link recommendation: {fit['recommended_n_links']} x {fit['recommended_link_kbps']:.0f} kbit/s at rho={fit['target_rho']:.2f}

**Claim boundary:** {result['claim_boundary']}
"""
    (out / "summary.md").write_text(summary, encoding="utf-8")
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Calibrate canonical SS7 model from an Osmocom M3UA trace"
    )
    ap.add_argument("--trace", required=True,
                    help="normalized CSV exported by tshark")
    ap.add_argument("--out", default="ss7_calibration_osmocom")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-fraction", type=float, default=0.60)
    ap.add_argument("--window-s", type=float, default=1.0)
    ap.add_argument("--target-rho", type=float, default=0.50)
    ap.add_argument("--max-validation-windows", type=int, default=80)
    ap.add_argument("--min-frames", type=int, default=20)
    args = ap.parse_args(argv)
    result = run_calibration(
        args.trace, args.out, seed=args.seed,
        train_fraction=args.train_fraction, window_s=args.window_s,
        target_rho=args.target_rho,
        max_validation_windows=args.max_validation_windows,
        min_frames=args.min_frames,
    )
    print(json.dumps({
        "profile": str(Path(args.out) / "calibrated_profile.json"),
        "flow_scale": result["fit"]["flow_scale"],
        "ks_statistic": result["validation"]["ks_statistic"],
        "relative_median_error": result["validation"]["relative_median_error"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
