from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


FEATURES = [
    "req_norm",
    "error_rate",
    "authz_403_rate",
    "http2_rst_rate",
    "streams_norm",
    "latency_ms",
    "latency_p95_ms",
    "payload_norm",
    "nrf_register_rate",
    "nrf_delete_rate",
    "discovery_rate",
    "nrf_query_rate",
]


def _fit_transform_params(train: pd.DataFrame) -> dict:
    params = {}
    for col in FEATURES:
        x = pd.to_numeric(train[col], errors="coerce")
        median = float(x.median()) if x.notna().any() else 0.0
        filled = x.fillna(median).to_numpy(float)
        mad = float(np.median(np.abs(filled - median)))
        params[col] = {
            "fill": median,
            "center": median,
            "scale": max(1.4826 * mad, 1e-6),
        }
    return params


def _transform(frame: pd.DataFrame, params: dict) -> np.ndarray:
    cols = []
    for col in FEATURES:
        p = params[col]
        x = pd.to_numeric(frame[col], errors="coerce").fillna(p["fill"]).to_numpy(float)
        cols.append(((x - p["center"]) / p["scale"]).reshape(-1, 1))
    return np.hstack(cols)


def _phase_metrics(scored: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    for phase, group in scored[scored["phase"] != "unlabeled"].groupby("phase"):
        y = group["label"].to_numpy(int)
        pred = group["pred"].to_numpy(int)
        positive = int(y.sum())
        rows.append(
            {
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
                "mean_score": float(group["score"].mean()),
                "max_score": float(group["score"].max()),
            }
        )
    return rows


def run(
    input_path: Path,
    output_dir: Path,
    target_fpr: float = 0.01,
    seed: int = 101,
) -> dict:
    frame = pd.read_csv(input_path).sort_values("window_start").reset_index(drop=True)
    baseline = frame[frame["phase"] == "baseline"].copy()
    if len(baseline) < 20:
        raise ValueError(f"need >=20 baseline windows, got {len(baseline)}")

    cut = max(10, int(round(len(baseline) * 0.65)))
    cut = min(cut, len(baseline) - 5)
    train = baseline.iloc[:cut].copy()
    calibration = baseline.iloc[cut:].copy()
    if len(calibration) < 5:
        raise ValueError("not enough baseline calibration windows")

    params = _fit_transform_params(train)
    model = IsolationForest(
        n_estimators=300,
        contamination="auto",
        max_samples="auto",
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(_transform(train, params))

    cal_score = -model.score_samples(_transform(calibration, params))
    threshold = float(np.quantile(cal_score, 1.0 - target_fpr))

    scored = frame.copy()
    scored["score"] = -model.score_samples(_transform(scored, params))
    scored["pred"] = (scored["score"] >= threshold).astype(int)

    metrics = _phase_metrics(scored)
    output_dir.mkdir(parents=True, exist_ok=True)
    scored.to_csv(output_dir / "real_sbi_scored_windows.csv", index=False)
    pd.DataFrame(metrics).to_csv(output_dir / "real_sbi_phase_metrics.csv", index=False)

    summary = {
        "model": "IsolationForest",
        "seed": seed,
        "target_fpr": target_fpr,
        "threshold": threshold,
        "features": FEATURES,
        "train_baseline_windows": int(len(train)),
        "calibration_baseline_windows": int(len(calibration)),
        "phase_metrics": metrics,
        "interpretation_note": (
            "udm_restart and nrf_burst are controlled operational perturbations, "
            "not asserted attack labels"
        ),
    }
    (output_dir / "real_sbi_analysis.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(
        scored["window_start"],
        scored["score"],
        marker="o",
        markersize=3,
        linewidth=1,
        label="real SBI score",
    )
    ax.axhline(threshold, linestyle="--", label="99th-percentile calibration threshold")
    for phase, alpha in (("udm_restart", 0.12), ("nrf_burst", 0.08)):
        for _, group in scored[scored["phase"] == phase].groupby("cycle"):
            if group.empty:
                continue
            ax.axvspan(
                group["window_start"].min(),
                group["window_start"].max() + 2.0,
                alpha=alpha,
            )
    ax.set_xlabel("Epoch time")
    ax.set_ylabel("Isolation-Forest anomaly score")
    ax.set_title("Real Open5GS SBI anomaly score")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "real_sbi_anomaly_timeline.png", dpi=170)
    plt.close(fig)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze real Open5GS SBI features")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-fpr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=101)
    args = parser.parse_args()

    summary = run(
        args.input,
        args.output_dir,
        target_fpr=args.target_fpr,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
