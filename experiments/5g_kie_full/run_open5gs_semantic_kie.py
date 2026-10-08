"""Independent semantic KIE cascade on *real* Open5GS SBI + signed reports.

No scenario/phase hints, fault-injection flags or ground-truth fields enter scores.
Training and all thresholds use only chronological baseline partitions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_open5gs_kie_analysis import (
    SBI_FEATURES,
    build_joined_features,
    fit_score,
    load_ground_truth,
    load_reports,
    validate_reports,
)

NF_IDS = ("nrf", "udm")
ACTIVE_ALPHA = 0.01
SBI_ALPHA = 0.01
CASCADE_ALPHA = 0.01


def baseline_split(frame: pd.DataFrame) -> dict[str, set[int]]:
    ids = np.array(sorted(frame.loc[frame["phase"] == "baseline", "window_id"].unique()))
    if len(ids) < 350:
        raise ValueError(f"At least 350 baseline windows needed; got {len(ids)}")
    n = len(ids)
    bounds = [0, int(.35*n), int(.60*n), int(.85*n), n]
    names = ("train", "channel_cal", "fusion_cal", "holdout")
    return {
        name: set(map(int, ids[bounds[j]:bounds[j+1]]))
        for j, name in enumerate(names)
    }


def _scale(series: pd.Series) -> tuple[float, float]:
    x = pd.to_numeric(series, errors="coerce")
    values = x.dropna().to_numpy(dtype=float)
    if not len(values):
        return 0.0, 1.0
    med = float(np.median(values))
    mad = float(np.median(np.abs(values-med)))
    std = float(np.std(values))
    # Absolute 0.2 floor prevents a sparse baseline from magnifying noise.
    return med, max(1.4826*mad, .2*std, .2)


def _positive_z(series: pd.Series, center: float, scale: float) -> np.ndarray:
    value = pd.to_numeric(series, errors="coerce").to_numpy(float)
    return np.maximum(0.0, (value-center)/scale)


def _upper_p(scores: np.ndarray, ref_scores: np.ndarray) -> np.ndarray:
    reference = np.sort(np.asarray(ref_scores, dtype=float))
    if not len(reference):
        raise ValueError("Empty benign calibration distribution")
    idx = np.searchsorted(reference, np.asarray(scores, dtype=float), side="left")
    return (1.0+len(reference)-idx)/(len(reference)+1.0)


def semantic_signals(joined: pd.DataFrame, train_ids: set[int]) -> pd.DataFrame:
    """Compute direction-specific activity and network-to-report lifecycle gaps.

    The NRF churn indicator for UDM is explicitly only a *proxy*: the current
    packet representation cannot reliably attribute client-side NF identity.
    """
    out = joined[["window_id", "window_start"]].copy()
    activity_channels = []
    lifecycle_channels = []
    invalid_channels = []
    missing_channels = []
    train = joined["window_id"].isin(train_ids)

    proxy = (
        pd.to_numeric(joined["nrf_register_rate"], errors="coerce").fillna(0.0)
        + pd.to_numeric(joined["nrf_delete_rate"], errors="coerce").fillna(0.0)
    )
    nrf_center, nrf_scale = _scale(np.log1p(proxy.loc[train]))

    for nf in NF_IDS:
        ext_col = f"kie_{nf}_external_activity_raw"
        rep_col = f"kie_{nf}_report_activity_raw"
        ext = pd.to_numeric(joined[ext_col], errors="coerce")
        rep = pd.to_numeric(joined[rep_col], errors="coerce")
        valid = joined[f"kie_{nf}_signature_valid"]
        present = valid.notna()
        verified = valid.fillna(False).astype(bool)
        missing = ~present
        invalid = present & (~verified)
        out[f"{nf}_missing_report"] = missing.to_numpy(bool)
        out[f"{nf}_invalid_signature"] = invalid.to_numpy(bool)

        ext_center, ext_scale = _scale(ext.loc[train])
        rep_center, rep_scale = _scale(rep.loc[train])
        e_z = _positive_z(ext, ext_center, ext_scale)
        r_z = _positive_z(rep, rep_center, rep_scale)

        # Temporal smoothing is trailing/causal and does not consult labels.
        e_smooth = pd.Series(e_z).rolling(3, min_periods=1).mean().to_numpy()
        r_smooth = pd.Series(r_z).rolling(3, min_periods=1).mean().to_numpy()
        directional = np.maximum(0.0, e_smooth-r_smooth)
        directional[missing.to_numpy(bool)] = 0.0
        out[f"{nf}_directional_gap"] = directional
        activity_channels.append(directional)

        generation = pd.to_numeric(
            joined[f"kie_{nf}_reported_generation_delta"], errors="coerce"
        ).fillna(0.0)
        recent_report_restart = (
            generation.rolling(5, min_periods=1).max().to_numpy(float) > 0
        )
        if nf == "udm":
            # An un-attributed NRF registration burst is NOT proof that UDM
            # restarted. Treat it as a weak, calibrated corroborating proxy.
            ext_life = _positive_z(np.log1p(proxy), nrf_center, nrf_scale)
        else:
            local_churn = (
                pd.to_numeric(
                    joined["ext_nrf_nrf_register_rate"], errors="coerce"
                ).fillna(0.0)
                + pd.to_numeric(
                    joined["ext_nrf_nrf_delete_rate"], errors="coerce"
                ).fillna(0.0)
            )
            c, s = _scale(np.log1p(local_churn.loc[train]))
            ext_life = _positive_z(np.log1p(local_churn), c, s)

        ext_life = pd.Series(ext_life).rolling(5, min_periods=1).max().to_numpy()
        life_gap = np.where(recent_report_restart, 0.0, ext_life)
        life_gap[missing.to_numpy(bool)] = 0.0
        out[f"{nf}_lifecycle_proxy_gap"] = life_gap
        lifecycle_channels.append(life_gap)
        invalid_channels.append(invalid.to_numpy(bool))
        missing_channels.append(missing.to_numpy(bool))

    out["activity_gap"] = np.maximum.reduce(activity_channels)
    # Lifecycle is independent auxiliary evidence but deliberately capped
    # relative to a strong activity under-reporting signal.
    out["lifecycle_proxy_gap"] = np.maximum.reduce(lifecycle_channels)
    out["semantic_raw"] = np.maximum(
        out["activity_gap"].to_numpy(float),
        .75*out["lifecycle_proxy_gap"].to_numpy(float),
    )
    out["signature_invalid"] = np.logical_or.reduce(invalid_channels)
    out["any_missing_report"] = np.logical_or.reduce(missing_channels)
    return out


def _phase_metrics(scored: pd.DataFrame, method: str) -> list[dict]:
    rows = []
    for phase, group in scored.groupby("phase_eval"):
        if phase in {"baseline_nonholdout", "washout", "unlabeled"}:
            continue
        pred = group[f"{method}_pred"].astype(bool)
        target = group["phase"].str.startswith(("truthful_", "hidden_"))
        rows.append({
            "method":method, "phase":phase, "windows":int(len(group)),
            "recall":float(pred.mean()) if target.all() else None,
            "false_alarm_rate":float(pred.mean()) if (~target).all() else None,
        })
    return rows


def _cycle_metrics(scored: pd.DataFrame, method: str) -> list[dict]:
    rows = []
    for (phase, cycle), group in scored.loc[
        scored["phase"].str.startswith(("truthful_", "hidden_"))
    ].groupby(["phase", "cycle"]):
        hits = group[group[f"{method}_pred"] == 1]
        rows.append({
            "method":method, "phase":phase, "cycle":int(cycle),
            "detected":bool(len(hits)),
            "first_alarm_offset_s":(
                float(hits["window_start"].min()-group["window_start"].min())
                if len(hits) else None
            ),
        })
    return rows


def analyze(
    global_path: Path,
    per_nf_path: Path,
    reports_path: Path,
    truth_path: Path,
    secret: bytes,
    output_dir: Path,
) -> dict:
    global_df = pd.read_csv(global_path).sort_values("window_id").reset_index(drop=True)
    per_nf_df = pd.read_csv(per_nf_path)
    reports = load_reports(reports_path, secret)
    truth = load_ground_truth(truth_path)
    joined = build_joined_features(global_df, per_nf_df, reports)
    split = baseline_split(joined)

    baseline_sbi = fit_score(
        joined,
        {"train":split["train"], "calibration":split["channel_cal"]},
        SBI_FEATURES,
        401,
        "sbi_only",
    )
    scored = joined[["window_id","window_start","phase","cycle"]].merge(
        baseline_sbi,on="window_id",how="left"
    )
    semantic = semantic_signals(joined,split["train"])
    scored = scored.merge(
        semantic.drop(columns=["window_start"]),on="window_id",how="left"
    )

    channel_ref = scored.loc[
        scored["window_id"].isin(split["channel_cal"]), "semantic_raw"
    ].to_numpy(float)
    scored["semantic_p"] = _upper_p(scored["semantic_raw"].to_numpy(float),channel_ref)
    scored["semantic_pred"] = (
        (scored["semantic_p"]<=ACTIVE_ALPHA) | scored["signature_invalid"]
    ).astype(int)

    scored["cascade_preserve_pred"] = (
        (scored["sbi_only_pred"] == 1) |
        (scored["semantic_pred"] == 1)
    ).astype(int)

    # Independent null calibration of the final fused ranking.
    scored["fusion_raw"] = np.maximum(
        -np.log10(np.clip(scored["sbi_only_p"],1e-12,1.0)),
        -np.log10(np.clip(scored["semantic_p"],1e-12,1.0)),
    )
    fusion_ref = scored.loc[
        scored["window_id"].isin(split["fusion_cal"]),"fusion_raw"
    ].to_numpy(float)
    scored["fusion_p"] = _upper_p(scored["fusion_raw"].to_numpy(float),fusion_ref)
    scored["fusion_calibrated_pred"] = (
        (scored["fusion_p"]<=CASCADE_ALPHA) |
        scored["signature_invalid"]
    ).astype(int)

    partition = {int(wid):part for part,ids in split.items() for wid in ids}
    scored["partition"] = scored["window_id"].map(partition).fillna("evaluation")
    scored["phase_eval"] = scored["phase"]
    baseline = scored["phase"]=="baseline"
    scored.loc[baseline & (scored["partition"]=="holdout"),"phase_eval"]="baseline_holdout"
    scored.loc[baseline & (scored["partition"]!="holdout"),"phase_eval"]="baseline_nonholdout"

    methods=("sbi_only","semantic","cascade_preserve","fusion_calibrated")
    phase_metrics=[m for method in methods for m in _phase_metrics(scored,method)]
    cycle_metrics=[m for method in methods for m in _cycle_metrics(scored,method)]
    attribution = scored.loc[
        (scored["phase"].str.startswith(("truthful_", "hidden_"))) &
        (scored["cascade_preserve_pred"]==1)
    ].copy()
    attribution["source"] = np.select(
        [
            (attribution["sbi_only_pred"]==1) & (attribution["semantic_pred"]==1),
            attribution["sbi_only_pred"]==1,
            attribution["semantic_pred"]==1,
        ],
        ["both","sbi_only","semantic_only"],default="none"
    )
    grouped_attr = attribution.groupby(["phase","source"]).size().rename(
        "alarm_windows"
    ).reset_index()

    output_dir.mkdir(parents=True,exist_ok=True)
    scored.to_csv(output_dir/"semantic_scored_windows.csv",index=False)
    pd.DataFrame(phase_metrics).to_csv(output_dir/"semantic_phase_metrics.csv",index=False)
    pd.DataFrame(cycle_metrics).to_csv(output_dir/"semantic_cycle_metrics.csv",index=False)
    grouped_attr.to_csv(output_dir/"semantic_stage_attribution.csv",index=False)

    validation=validate_reports(reports,truth)
    summary={
        "baseline_split":{k:len(v) for k,v in split.items()},
        "channel_p_min":1/(len(split["channel_cal"])+1),
        "fusion_p_min":1/(len(split["fusion_cal"])+1),
        "target_fpr":CASCADE_ALPHA,
        "methods":methods,
        "phase_metrics":phase_metrics,
        "cycle_metrics":cycle_metrics,
        "report_validation":validation,
        "scientific_caveats":[
            "No ground-truth local telemetry or phase/fault markers used for detector scoring.",
            "Under-report gap uses separately normalized resource/activity proxies; not a matched request count.",
            "UDM lifecycle cue is un-attributed NRF churn, not independent proof of a UDM restart.",
            "HMAC authenticates report bytes against the sidecar key; it does not establish trusted hardware attestation.",
            "Temporal smoothing is trailing/causal; correlated 2-second windows weaken formal exchangeability assumptions.",
            "0 false alarms on a small holdout does not establish 1% operational FPR.",
        ]
    }
    (output_dir/"semantic_analysis.json").write_text(
        json.dumps(summary,indent=2,ensure_ascii=False),encoding="utf-8"
    )
    fig,ax=plt.subplots(figsize=(14,5))
    ax.plot(scored["window_start"],-np.log10(np.clip(scored["sbi_only_p"],1e-12,1.0)),
            label="passive SBI",linewidth=1)
    ax.plot(scored["window_start"],-np.log10(np.clip(scored["semantic_p"],1e-12,1.0)),
            label="semantic KIE",linewidth=1)
    ax.axhline(2.,linestyle="--",label="p=0.01")
    for phase,color in [("hidden_udm_restart","red"),("hidden_nrf_burst","orange")]:
        for _, g in scored[scored["phase"]==phase].groupby("cycle"):
            ax.axvspan(g["window_start"].min(),g["window_start"].max()+2,
                       alpha=.10,color=color)
    ax.legend(loc="best");ax.set_xlabel("Unix time (s)")
    ax.set_ylabel("-log10 empirical p")
    ax.set_title("Open5GS: semantic KIE vs passive SBI")
    fig.tight_layout()
    fig.savefig(output_dir/"semantic_timeline.png",dpi=160)
    plt.close(fig)
    return summary


def main() -> None:
    p=argparse.ArgumentParser(description="Semantic KIE / SBI real Open5GS analysis")
    p.add_argument("--global-features",type=Path,required=True)
    p.add_argument("--per-nf-features",type=Path,required=True)
    p.add_argument("--reports",type=Path,required=True)
    p.add_argument("--ground-truth",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--secret-env",default="OPEN5GS_KIE_SECRET")
    args=p.parse_args()
    import os
    secret=os.environ.get(args.secret_env)
    if not secret: raise SystemExit("HMAC key missing from environment")
    result=analyze(args.global_features,args.per_nf_features,args.reports,
                   args.ground_truth,secret.encode("utf-8"),args.output_dir)
    print(json.dumps(result,indent=2,ensure_ascii=False))


if __name__=="__main__":
    main()
