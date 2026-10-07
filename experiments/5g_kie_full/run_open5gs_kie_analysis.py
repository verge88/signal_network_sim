from __future__ import annotations

import argparse
import hashlib
import hmac
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from open5gs_kie_sidecar import verify_report


SBI_FEATURES = [
    "observed_rps",
    "error_rate",
    "authz_403_rate",
    "http2_rst_rate",
    "unique_streams",
    "latency_ms",
    "latency_p95_ms",
    "tcp_payload_bytes",
    "nrf_register_rate",
    "nrf_delete_rate",
    "discovery_rate",
    "nrf_query_rate",
]

NF_IDS = ("udm", "nrf")
PERTURBATION_PHASES = {
    "truthful_udm_restart",
    "hidden_udm_restart",
    "truthful_nrf_burst",
    "hidden_nrf_burst",
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_reports(path: Path, secret: bytes) -> pd.DataFrame:
    rows = []
    for report in _read_jsonl(path):
        rep = report.get("reported", {})
        rows.append(
            {
                "ts": float(report["ts"]),
                "sequence": int(report["sequence"]),
                "nf_id": str(report["nf_id"]).lower(),
                "signature_valid": bool(verify_report(report, secret)),
                "fault_injection": str(report.get("fault_injection", "none")),
                "reported_pid": int(rep.get("pid", 0) or 0),
                "reported_invocation_id": str(rep.get("invocation_id", "")),
                "reported_generation": int(rep.get("restart_generation", 0) or 0),
                "reported_uptime_s": float(rep.get("service_uptime_s", 0.0) or 0.0),
                "reported_cpu_delta": float(rep.get("cpu_ticks_delta", 0.0) or 0.0),
                "reported_io_delta": float(rep.get("io_chars_delta", 0.0) or 0.0),
                "reported_ctx_delta": float(rep.get("ctx_switch_delta", 0.0) or 0.0),
                "reported_fd_count": float(rep.get("fd_count", 0.0) or 0.0),
            }
        )
    frame = pd.DataFrame(rows).sort_values(["nf_id", "ts"]).reset_index(drop=True)
    if frame.empty:
        raise ValueError("no KIE reports")
    for nf, idx in frame.groupby("nf_id").groups.items():
        g = frame.loc[idx].sort_values("ts")
        frame.loc[g.index, "reported_pid_change"] = (
            g["reported_pid"].ne(g["reported_pid"].shift()).astype(float)
        )
        frame.loc[g.index, "reported_generation_delta"] = (
            g["reported_generation"].diff().fillna(0.0).clip(lower=0.0)
        )
        frame.loc[g.index, "reported_uptime_drop"] = (
            (g["reported_uptime_s"].shift() - g["reported_uptime_s"])
            .fillna(0.0)
            .clip(lower=0.0)
        )
    return frame


def load_ground_truth(path: Path) -> pd.DataFrame:
    rows = []
    for row in _read_jsonl(path):
        actual = row.get("actual", {})
        truthful = row.get("truthful_report", {})
        rows.append(
            {
                "ts": float(row["ts"]),
                "sequence": int(row["sequence"]),
                "nf_id": str(row["nf_id"]).lower(),
                "masking_active": bool(row.get("masking_active", False)),
                "actual_pid": int(actual.get("pid", 0) or 0),
                "actual_invocation_id": str(actual.get("invocation_id", "")),
                "actual_cpu_ticks": float(actual.get("cpu_ticks", 0.0) or 0.0),
                "actual_io_chars": float(actual.get("io_chars", 0.0) or 0.0),
                "truth_generation": int(truthful.get("restart_generation", 0) or 0),
                "truth_cpu_delta": float(truthful.get("cpu_ticks_delta", 0.0) or 0.0),
                "truth_io_delta": float(truthful.get("io_chars_delta", 0.0) or 0.0),
                "truth_ctx_delta": float(truthful.get("ctx_switch_delta", 0.0) or 0.0),
            }
        )
    return pd.DataFrame(rows).sort_values(["nf_id", "ts"]).reset_index(drop=True)


def _nearest_reports(
    windows: pd.DataFrame,
    reports: pd.DataFrame,
    nf_id: str,
    tolerance_s: float = 2.5,
) -> pd.DataFrame:
    left = windows[["window_id", "window_start"]].copy()
    left["center_ts"] = left["window_start"] + 1.0
    right = reports[reports["nf_id"] == nf_id].sort_values("ts").copy()
    merged = pd.merge_asof(
        left.sort_values("center_ts"),
        right,
        left_on="center_ts",
        right_on="ts",
        direction="nearest",
        tolerance=tolerance_s,
    )
    return merged.sort_values("window_id").reset_index(drop=True)


def _robust_params(series: pd.Series) -> tuple[float, float]:
    x = pd.to_numeric(series, errors="coerce")
    med = float(x.median()) if x.notna().any() else 0.0
    arr = x.fillna(med).to_numpy(float)
    mad = float(np.median(np.abs(arr - med)))
    std = float(np.std(arr))
    return med, max(1.4826 * mad, 0.10 * std, 1e-6)


def _robust_z(series: pd.Series, med: float, scale: float) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce").fillna(med)
    return (x - med).abs() / scale


def build_joined_features(
    global_df: pd.DataFrame,
    per_nf_df: pd.DataFrame,
    reports: pd.DataFrame,
) -> pd.DataFrame:
    out = global_df.sort_values("window_id").reset_index(drop=True).copy()
    out["label"] = out["phase"].isin(PERTURBATION_PHASES).astype(int)

    for nf in NF_IDS:
        nf_ext = (
            per_nf_df[per_nf_df["server_nf"] == nf][
                [
                    "window_id",
                    "observed_rps",
                    "nrf_register_rate",
                    "nrf_delete_rate",
                    "discovery_rate",
                    "nrf_query_rate",
                ]
            ]
            .copy()
            .rename(
                columns={
                    c: f"ext_{nf}_{c}"
                    for c in [
                        "observed_rps",
                        "nrf_register_rate",
                        "nrf_delete_rate",
                        "discovery_rate",
                        "nrf_query_rate",
                    ]
                }
            )
        )
        out = out.merge(nf_ext, on="window_id", how="left")

        aligned = _nearest_reports(out, reports, nf)
        keep = [
            "window_id",
            "signature_valid",
            "fault_injection",
            "reported_pid",
            "reported_generation",
            "reported_uptime_s",
            "reported_cpu_delta",
            "reported_io_delta",
            "reported_ctx_delta",
            "reported_fd_count",
            "reported_pid_change",
            "reported_generation_delta",
            "reported_uptime_drop",
        ]
        aligned = aligned[keep].rename(
            columns={c: f"kie_{nf}_{c}" for c in keep if c != "window_id"}
        )
        out = out.merge(aligned, on="window_id", how="left")

        out[f"kie_{nf}_report_activity_raw"] = (
            np.log1p(out[f"kie_{nf}_reported_cpu_delta"].fillna(0.0).clip(lower=0.0))
            + 0.25
            * np.log1p(out[f"kie_{nf}_reported_ctx_delta"].fillna(0.0).clip(lower=0.0))
            + 0.05
            * np.log1p(out[f"kie_{nf}_reported_io_delta"].fillna(0.0).clip(lower=0.0))
        )

        external = np.log1p(
            out[f"ext_{nf}_observed_rps"].fillna(0.0).clip(lower=0.0)
        )
        if nf == "nrf":
            external = (
                external
                + np.log1p(
                    out[f"ext_{nf}_nrf_query_rate"].fillna(0.0).clip(lower=0.0)
                    + out[f"ext_{nf}_nrf_register_rate"].fillna(0.0).clip(lower=0.0)
                    + out[f"ext_{nf}_nrf_delete_rate"].fillna(0.0).clip(lower=0.0)
                )
            )
        else:
            external = (
                external
                + np.log1p(
                    out["nrf_register_rate"].fillna(0.0).clip(lower=0.0)
                    + out["nrf_delete_rate"].fillna(0.0).clip(lower=0.0)
                    + out["discovery_rate"].fillna(0.0).clip(lower=0.0)
                )
            )
        out[f"kie_{nf}_external_activity_raw"] = external

    return out


def add_consistency_features(
    frame: pd.DataFrame,
    baseline_train_ids: set[int],
) -> pd.DataFrame:
    out = frame.copy()
    train = out[out["window_id"].isin(baseline_train_ids)]
    for nf in NF_IDS:
        rcol = f"kie_{nf}_report_activity_raw"
        ecol = f"kie_{nf}_external_activity_raw"
        rmed, rscale = _robust_params(train[rcol])
        emed, escale = _robust_params(train[ecol])
        out[f"kie_{nf}_report_activity_z"] = _robust_z(out[rcol], rmed, rscale)
        out[f"kie_{nf}_external_activity_z"] = _robust_z(out[ecol], emed, escale)
        out[f"kie_{nf}_consistency_gap"] = (
            out[f"kie_{nf}_external_activity_z"]
            - out[f"kie_{nf}_report_activity_z"]
        ).abs()
        out[f"kie_{nf}_signature_invalid"] = (
            ~out[f"kie_{nf}_signature_valid"].fillna(False).astype(bool)
        ).astype(float)
    return out


KIE_EXTRA_FEATURES = [
    f"kie_{nf}_{feature}"
    for nf in NF_IDS
    for feature in [
        "reported_generation_delta",
        "reported_pid_change",
        "reported_uptime_drop",
        "reported_cpu_delta",
        "reported_io_delta",
        "reported_ctx_delta",
        "reported_fd_count",
        "report_activity_z",
        "external_activity_z",
        "consistency_gap",
        "signature_invalid",
    ]
]


def _split_baseline(frame: pd.DataFrame) -> dict[str, set[int]]:
    ids = sorted(frame.loc[frame["phase"] == "baseline", "window_id"].unique())
    if len(ids) < 300:
        raise ValueError(f"need >=300 baseline windows, got {len(ids)}")
    n = len(ids)
    train_end = int(round(n * 0.50))
    cal_end = int(round(n * 0.75))
    return {
        "train": set(ids[:train_end]),
        "calibration": set(ids[train_end:cal_end]),
        "holdout": set(ids[cal_end:]),
    }


def _fit_scale(train: pd.DataFrame, features: list[str]) -> dict[str, dict[str, float]]:
    params = {}
    for col in features:
        x = pd.to_numeric(train[col], errors="coerce")
        med = float(x.median()) if x.notna().any() else 0.0
        arr = x.fillna(med).to_numpy(float)
        mad = float(np.median(np.abs(arr - med)))
        std = float(np.std(arr))
        params[col] = {
            "fill": med,
            "center": med,
            "scale": max(1.4826 * mad, 0.10 * std, 1e-6),
        }
    return params


def _transform(
    frame: pd.DataFrame, features: list[str], params: dict[str, dict[str, float]]
) -> np.ndarray:
    cols = []
    for col in features:
        p = params[col]
        x = pd.to_numeric(frame[col], errors="coerce").fillna(p["fill"]).to_numpy(float)
        cols.append(((x - p["center"]) / p["scale"]).reshape(-1, 1))
    return np.hstack(cols)


def _conformal_p(score: np.ndarray, calibration_score: np.ndarray) -> np.ndarray:
    cal = np.sort(np.asarray(calibration_score, dtype=float))
    idx = np.searchsorted(cal, score, side="left")
    ge = len(cal) - idx
    return (1.0 + ge.astype(float)) / (len(cal) + 1.0)


def fit_score(
    frame: pd.DataFrame,
    split: dict[str, set[int]],
    features: list[str],
    seed: int,
    prefix: str,
) -> pd.DataFrame:
    train = frame[frame["window_id"].isin(split["train"])].copy()
    cal = frame[frame["window_id"].isin(split["calibration"])].copy()
    params = _fit_scale(train, features)
    model = IsolationForest(
        n_estimators=350,
        contamination="auto",
        max_samples="auto",
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(_transform(train, features, params))
    cal_score = -model.score_samples(_transform(cal, features, params))
    score = -model.score_samples(_transform(frame, features, params))
    p = _conformal_p(score, cal_score)

    out = frame[["window_id"]].copy()
    out[f"{prefix}_score"] = score
    out[f"{prefix}_p"] = p
    out[f"{prefix}_pred"] = (p <= 0.01).astype(int)
    return out


def _phase_metrics(scored: pd.DataFrame, prefix: str) -> list[dict]:
    rows = []
    for phase, group in scored.groupby("phase_eval"):
        if phase in {"washout", "baseline_nonholdout", "unlabeled"}:
            continue
        y = group["label"].to_numpy(int)
        pred = group[f"{prefix}_pred"].to_numpy(int)
        positive = int(y.sum())
        rows.append(
            {
                "method": prefix,
                "phase": phase,
                "windows": int(len(group)),
                "positive_windows": positive,
                "recall": (
                    float(((pred == 1) & (y == 1)).sum() / positive)
                    if positive
                    else None
                ),
                "false_alarm_rate": (
                    float((pred == 1).sum() / max(len(pred), 1))
                    if not positive
                    else None
                ),
                "mean_p": float(group[f"{prefix}_p"].mean()),
            }
        )
    return rows


def _cycle_metrics(scored: pd.DataFrame, prefix: str) -> list[dict]:
    rows = []
    for phase in sorted(PERTURBATION_PHASES):
        for cycle, group in scored[scored["phase"] == phase].groupby("cycle"):
            detected = bool(group[f"{prefix}_pred"].astype(bool).any())
            rows.append(
                {
                    "method": prefix,
                    "phase": phase,
                    "cycle": int(cycle),
                    "windows": int(len(group)),
                    "detected": detected,
                    "first_alarm_offset_s": (
                        float(
                            group.loc[
                                group[f"{prefix}_pred"] == 1, "window_start"
                            ].min()
                            - group["window_start"].min()
                        )
                        if detected
                        else None
                    ),
                }
            )
    return rows


def validate_reports(
    reports: pd.DataFrame, truth: pd.DataFrame
) -> dict[str, Any]:
    merged = reports.merge(
        truth,
        on=["sequence", "nf_id"],
        how="inner",
        suffixes=("_report", "_truth"),
    )
    hidden = merged[merged["masking_active"]].copy()
    if hidden.empty:
        divergence = 0.0
    else:
        report_activity = (
            hidden["reported_cpu_delta"].fillna(0.0)
            + hidden["reported_io_delta"].fillna(0.0)
            + hidden["reported_ctx_delta"].fillna(0.0)
        )
        truth_activity = (
            hidden["truth_cpu_delta"].fillna(0.0)
            + hidden["truth_io_delta"].fillna(0.0)
            + hidden["truth_ctx_delta"].fillna(0.0)
        )
        divergence = float((report_activity != truth_activity).mean())

    return {
        "report_count": int(len(reports)),
        "all_signatures_valid": bool(reports["signature_valid"].all()),
        "invalid_signature_count": int((~reports["signature_valid"]).sum()),
        "ground_truth_pairs": int(len(merged)),
        "masked_ground_truth_pairs": int(len(hidden)),
        "masked_activity_divergence_fraction": divergence,
        "controlled_fault_injection_only": True,
    }


def run(
    global_path: Path,
    per_nf_path: Path,
    reports_path: Path,
    ground_truth_path: Path,
    secret: bytes,
    output_dir: Path,
) -> dict[str, Any]:
    global_df = pd.read_csv(global_path).sort_values("window_id").reset_index(drop=True)
    per_nf_df = pd.read_csv(per_nf_path)
    reports = load_reports(reports_path, secret)
    truth = load_ground_truth(ground_truth_path)
    split = _split_baseline(global_df)

    joined = build_joined_features(global_df, per_nf_df, reports)
    joined = add_consistency_features(joined, split["train"])

    sbi = fit_score(joined, split, SBI_FEATURES, 401, "sbi_only")
    full = fit_score(
        joined,
        split,
        SBI_FEATURES + KIE_EXTRA_FEATURES,
        402,
        "sbi_kie",
    )
    scored = joined.merge(sbi, on="window_id").merge(full, on="window_id")

    partition = {}
    for name, ids in split.items():
        for wid in ids:
            partition[int(wid)] = name
    scored["partition"] = scored["window_id"].map(partition).fillna("evaluation")
    scored["phase_eval"] = scored["phase"]
    scored.loc[
        (scored["phase"] == "baseline") & (scored["partition"] == "holdout"),
        "phase_eval",
    ] = "baseline_holdout"
    scored.loc[
        (scored["phase"] == "baseline") & (scored["partition"] != "holdout"),
        "phase_eval",
    ] = "baseline_nonholdout"

    phase_metrics = _phase_metrics(scored, "sbi_only") + _phase_metrics(
        scored, "sbi_kie"
    )
    cycle_metrics = _cycle_metrics(scored, "sbi_only") + _cycle_metrics(
        scored, "sbi_kie"
    )
    validation = validate_reports(reports, truth)

    output_dir.mkdir(parents=True, exist_ok=True)
    scored.to_csv(output_dir / "kie_scored_windows.csv", index=False)
    pd.DataFrame(phase_metrics).to_csv(
        output_dir / "kie_phase_metrics.csv", index=False
    )
    pd.DataFrame(cycle_metrics).to_csv(
        output_dir / "kie_cycle_metrics.csv", index=False
    )
    (output_dir / "kie_report_validation.json").write_text(
        json.dumps(validation, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    summary = {
        "baseline_split": {k: len(v) for k, v in split.items()},
        "calibration_resolution": float(1.0 / (len(split["calibration"]) + 1)),
        "sbi_features": SBI_FEATURES,
        "kie_extra_features": KIE_EXTRA_FEATURES,
        "phase_metrics": phase_metrics,
        "cycle_metrics": cycle_metrics,
        "report_validation": validation,
        "interpretation_note": (
            "hidden-report phases use an explicitly controlled report-masking fault "
            "injection. Ground-truth local telemetry is stored separately and is not "
            "used as a detector feature."
        ),
    }
    (output_dir / "kie_analysis.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(
        scored["window_start"],
        -np.log10(np.clip(scored["sbi_only_p"], 1e-12, 1.0)),
        linewidth=1,
        label="SBI-only",
    )
    ax.plot(
        scored["window_start"],
        -np.log10(np.clip(scored["sbi_kie_p"], 1e-12, 1.0)),
        linewidth=1,
        label="SBI + KIE",
    )
    ax.axhline(2.0, linestyle="--", label="p=0.01")
    for phase, alpha in (
        ("truthful_udm_restart", 0.08),
        ("hidden_udm_restart", 0.12),
        ("truthful_nrf_burst", 0.08),
        ("hidden_nrf_burst", 0.12),
        ("washout", 0.03),
    ):
        for _, group in scored[scored["phase"] == phase].groupby("cycle"):
            if not group.empty:
                ax.axvspan(
                    group["window_start"].min(),
                    group["window_start"].max() + 2.0,
                    alpha=alpha,
                )
    ax.set_xlabel("Epoch time")
    ax.set_ylabel("-log10 conformal p")
    ax.set_title("Real Open5GS: passive SBI vs SBI + signed KIE")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "kie_comparison_timeline.png", dpi=170)
    plt.close(fig)
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="Compare real SBI-only and SBI+KIE")
    p.add_argument("--global-features", type=Path, required=True)
    p.add_argument("--per-nf-features", type=Path, required=True)
    p.add_argument("--reports", type=Path, required=True)
    p.add_argument("--ground-truth", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--secret-env", default="OPEN5GS_KIE_SECRET")
    args = p.parse_args()

    import os

    secret_text = os.environ.get(args.secret_env, "")
    if not secret_text:
        raise SystemExit(f"missing secret in {args.secret_env}")

    summary = run(
        args.global_features,
        args.per_nf_features,
        args.reports,
        args.ground_truth,
        secret_text.encode("utf-8"),
        args.output_dir,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
