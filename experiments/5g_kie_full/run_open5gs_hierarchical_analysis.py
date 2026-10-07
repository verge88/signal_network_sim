from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


FEATURES = [
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


def _fit_scale(train: pd.DataFrame) -> dict:
    params = {}
    for col in FEATURES:
        x = pd.to_numeric(train[col], errors="coerce")
        median = float(x.median()) if x.notna().any() else 0.0
        filled = x.fillna(median).to_numpy(float)
        mad = float(np.median(np.abs(filled - median)))
        std = float(np.std(filled))
        scale = max(1.4826 * mad, 0.10 * std, 1e-6)
        params[col] = {"fill": median, "center": median, "scale": scale}
    return params


def _transform(frame: pd.DataFrame, params: dict) -> np.ndarray:
    cols = []
    for col in FEATURES:
        p = params[col]
        x = pd.to_numeric(frame[col], errors="coerce").fillna(p["fill"]).to_numpy(float)
        cols.append(((x - p["center"]) / p["scale"]).reshape(-1, 1))
    return np.hstack(cols)


@dataclass
class ConformalIF:
    name: str
    model: IsolationForest
    params: dict
    calibration_scores: np.ndarray

    def score(self, frame: pd.DataFrame) -> np.ndarray:
        return -self.model.score_samples(_transform(frame, self.params))

    def pvalue(self, frame: pd.DataFrame) -> np.ndarray:
        score = self.score(frame)
        cal = self.calibration_scores
        idx = np.searchsorted(cal, score, side="left")
        ge = len(cal) - idx
        return (1.0 + ge.astype(float)) / (len(cal) + 1.0)


def fit_detector(name: str, train: pd.DataFrame, calibration: pd.DataFrame, seed: int) -> ConformalIF:
    params = _fit_scale(train)
    model = IsolationForest(
        n_estimators=350,
        contamination="auto",
        max_samples="auto",
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(_transform(train, params))
    cal = -model.score_samples(_transform(calibration, params))
    return ConformalIF(name, model, params, np.sort(np.asarray(cal, dtype=float)))


def _split_baseline_ids(global_df: pd.DataFrame) -> dict[str, set[int]]:
    ids = sorted(global_df.loc[global_df["phase"] == "baseline", "window_id"].unique())
    if len(ids) < 200:
        raise ValueError(f"need at least 200 baseline windows, got {len(ids)}")

    n = len(ids)
    a = int(round(n * 0.40))
    b = int(round(n * 0.60))
    c = int(round(n * 0.80))
    return {
        "train": set(ids[:a]),
        "channel_cal": set(ids[a:b]),
        "fusion_cal": set(ids[b:c]),
        "holdout": set(ids[c:]),
    }


def _subset(frame: pd.DataFrame, ids: set[int]) -> pd.DataFrame:
    return frame[frame["window_id"].isin(ids)].copy()


def _eligible_nf(frame: pd.DataFrame, train_ids: set[int]) -> bool:
    train = _subset(frame, train_ids)
    if len(train) < 50:
        return False
    nonzero = int((train["request_count"] > 0).sum())
    return nonzero >= 10


def _channel_pvalues(
    global_df: pd.DataFrame,
    per_nf_df: pd.DataFrame,
    detectors: dict[str, ConformalIF],
) -> pd.DataFrame:
    out = global_df[["window_id", "window_start", "phase", "cycle", "label"]].copy()
    global_det = detectors["GLOBAL"]
    out["p_GLOBAL"] = global_det.pvalue(global_df)

    for channel, detector in detectors.items():
        if channel == "GLOBAL":
            continue
        nf = channel.removeprefix("NF:")
        nf_frame = per_nf_df[per_nf_df["server_nf"] == nf].sort_values("window_id")
        nf_frame = out[["window_id"]].merge(nf_frame, on="window_id", how="left")
        for col in FEATURES:
            if col not in nf_frame:
                nf_frame[col] = np.nan
        out[f"p_{channel}"] = detector.pvalue(nf_frame)

    pcols = [c for c in out.columns if c.startswith("p_")]
    out["min_channel_p"] = out[pcols].min(axis=1)
    out["raw_fusion_score"] = -np.log10(np.clip(out["min_channel_p"], 1e-12, 1.0))
    out["dominant_channel"] = out[pcols].idxmin(axis=1).str.removeprefix("p_")
    return out


def _conformal_from_reference(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    ref = np.sort(np.asarray(reference, dtype=float))
    idx = np.searchsorted(ref, values, side="left")
    ge = len(ref) - idx
    return (1.0 + ge.astype(float)) / (len(ref) + 1.0)


def _phase_metrics(scored: pd.DataFrame, pred_col: str, baseline_partition: str) -> list[dict]:
    rows = []
    eval_df = scored.copy()
    eval_df.loc[
        (eval_df["phase"] == "baseline") & (eval_df["partition"] != baseline_partition),
        "phase_eval",
    ] = "baseline_nonholdout"
    eval_df.loc[
        (eval_df["phase"] == "baseline") & (eval_df["partition"] == baseline_partition),
        "phase_eval",
    ] = "baseline_holdout"
    eval_df["phase_eval"] = eval_df["phase_eval"].fillna(eval_df["phase"])

    for phase, group in eval_df.groupby("phase_eval"):
        if phase == "baseline_nonholdout" or phase == "unlabeled" or phase == "washout":
            continue
        y = group["label"].to_numpy(int)
        pred = group[pred_col].to_numpy(int)
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
            }
        )
    return rows


def _cycle_metrics(scored: pd.DataFrame, pred_col: str) -> list[dict]:
    rows = []
    for phase in ("udm_restart", "nrf_burst"):
        for cycle, group in scored[scored["phase"] == phase].groupby("cycle"):
            if int(cycle) < 0:
                continue
            rows.append(
                {
                    "phase": phase,
                    "cycle": int(cycle),
                    "windows": int(len(group)),
                    "detected": bool(group[pred_col].astype(bool).any()),
                    "first_alarm_offset_s": (
                        float(
                            group.loc[group[pred_col] == 1, "window_start"].min()
                            - group["window_start"].min()
                        )
                        if group[pred_col].astype(bool).any()
                        else None
                    ),
                }
            )
    return rows


def run(
    global_path: Path,
    per_nf_path: Path,
    output_dir: Path,
    target_fpr: float = 0.01,
    seed: int = 202,
) -> dict:
    global_df = pd.read_csv(global_path).sort_values("window_id").reset_index(drop=True)
    per_nf_df = pd.read_csv(per_nf_path).sort_values(["server_nf", "window_id"]).reset_index(drop=True)
    split = _split_baseline_ids(global_df)

    partition = {}
    for name, ids in split.items():
        for wid in ids:
            partition[int(wid)] = name

    detectors: dict[str, ConformalIF] = {}
    detectors["GLOBAL"] = fit_detector(
        "GLOBAL",
        _subset(global_df, split["train"]),
        _subset(global_df, split["channel_cal"]),
        seed,
    )

    for idx, nf in enumerate(sorted(per_nf_df["server_nf"].dropna().unique())):
        frame = per_nf_df[per_nf_df["server_nf"] == nf].copy()
        if not _eligible_nf(frame, split["train"]):
            continue
        detectors[f"NF:{nf}"] = fit_detector(
            f"NF:{nf}",
            _subset(frame, split["train"]),
            _subset(frame, split["channel_cal"]),
            seed + 101 * (idx + 1),
        )

    scored = _channel_pvalues(global_df, per_nf_df, detectors)
    scored["partition"] = scored["window_id"].map(partition).fillna("evaluation")

    fusion_ref = scored.loc[
        scored["window_id"].isin(split["fusion_cal"]), "raw_fusion_score"
    ].to_numpy(float)
    scored["hier_p"] = _conformal_from_reference(
        scored["raw_fusion_score"].to_numpy(float), fusion_ref
    )
    scored["hier_pred"] = (scored["hier_p"] <= target_fpr).astype(int)

    global_p = scored["p_GLOBAL"].to_numpy(float)
    scored["global_pred"] = (global_p <= target_fpr).astype(int)

    global_metrics = _phase_metrics(scored, "global_pred", "holdout")
    hier_metrics = _phase_metrics(scored, "hier_pred", "holdout")
    global_cycles = _cycle_metrics(scored, "global_pred")
    hier_cycles = _cycle_metrics(scored, "hier_pred")

    output_dir.mkdir(parents=True, exist_ok=True)
    scored.to_csv(output_dir / "hierarchical_scored_windows.csv", index=False)
    pd.DataFrame(global_metrics).to_csv(output_dir / "global_phase_metrics.csv", index=False)
    pd.DataFrame(hier_metrics).to_csv(output_dir / "hierarchical_phase_metrics.csv", index=False)
    pd.DataFrame(hier_cycles).to_csv(output_dir / "hierarchical_cycle_metrics.csv", index=False)

    alarm_attr = (
        scored[scored["hier_pred"] == 1]
        .groupby(["phase", "dominant_channel"])
        .size()
        .rename("alarm_windows")
        .reset_index()
    )
    alarm_attr.to_csv(output_dir / "hierarchical_alarm_attribution.csv", index=False)

    summary = {
        "target_fpr": target_fpr,
        "baseline_windows": int(sum(len(x) for x in split.values())),
        "baseline_split": {k: len(v) for k, v in split.items()},
        "channel_calibration_resolution": float(1.0 / (len(split["channel_cal"]) + 1)),
        "fusion_calibration_resolution": float(1.0 / (len(split["fusion_cal"]) + 1)),
        "detectors": sorted(detectors),
        "global_phase_metrics": global_metrics,
        "hierarchical_phase_metrics": hier_metrics,
        "global_cycle_metrics": global_cycles,
        "hierarchical_cycle_metrics": hier_cycles,
        "note": (
            "Baseline FPR is reported only on the untouched final 20% holdout. "
            "Washout windows are excluded from benign FPR. Controlled restarts and "
            "NRF bursts are perturbation labels, not asserted attacks."
        ),
    }
    (output_dir / "hierarchical_analysis.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(
        scored["window_start"],
        -np.log10(np.clip(scored["hier_p"], 1e-12, 1.0)),
        linewidth=1,
        marker="o",
        markersize=2,
        label="hierarchical conformal score",
    )
    ax.axhline(-np.log10(target_fpr), linestyle="--", label=f"alpha={target_fpr:g}")
    for phase, alpha in (("udm_restart", 0.12), ("nrf_burst", 0.08), ("washout", 0.04)):
        for _, group in scored[scored["phase"] == phase].groupby("cycle"):
            if group.empty:
                continue
            ax.axvspan(
                group["window_start"].min(),
                group["window_start"].max() + 2.0,
                alpha=alpha,
            )
    ax.set_xlabel("Epoch time")
    ax.set_ylabel("-log10 conformal p")
    ax.set_title("Real Open5GS hierarchical SBI detector")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "hierarchical_sbi_timeline.png", dpi=170)
    plt.close(fig)
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="Hierarchical real Open5GS SBI analysis")
    p.add_argument("--global-features", type=Path, required=True)
    p.add_argument("--per-nf-features", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--target-fpr", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=202)
    args = p.parse_args()
    summary = run(
        args.global_features,
        args.per_nf_features,
        args.output_dir,
        target_fpr=args.target_fpr,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
