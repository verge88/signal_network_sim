"""Independent per-HTTP/2-request NRF capture audit for fresh and archived trials.

This is a measurement/coverage audit, NOT a revised KIE detector. Its
experimental trial labels are joined only after extracting *all* raw passive
packet timestamps and after frozen v1/v2 decisions have been archived.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from open5gs_real_features import load_endpoint_map, load_tshark_events

INTERVENTIONS = ("truthful_nrf_burst", "hidden_nrf_burst")
# New independent replicates plus six prospectively randomized original runs.
RUNNERS = {
    21011:1., 21013:1., 31011:1., 31013:1.,
    22011:2., 22013:2., 32011:2., 32013:2.,
    24011:4., 24013:4., 34011:4., 34013:4.,
}
DEFAULT_THRESHOLD_COUNT = 20  # Frozen old 2-second NRF 10 GET/s threshold.
RAW_PRESENCE_COUNT = 5  # Coverage audit only, NOT an intrusion detector.


def _collected_requests(events_path: Path, endpoints_path: Path) -> pd.DataFrame:
    """Return independent NRF GET packet evidence; no trial labels or KIE fields."""
    events = load_tshark_events(events_path)
    mapped = load_endpoint_map(endpoints_path)
    nrf_addresses = {ip for ip, nf in mapped.items() if nf == "nrf"}
    selected = events[
        events.method.str.upper().eq("GET")
        & events.path.str.contains("/nnrf-", regex=False)
        & events.ip_dst.isin(nrf_addresses)
    ].copy()
    selected["request_key"] = (
        selected.tcp_stream.astype(str) + "|" + selected.h2_stream.astype(str)
    )
    if (selected.tcp_stream.astype(str).eq("").any() or
        selected.h2_stream.astype(str).eq("").any()):
        raise ValueError("cannot uniquely identify raw HTTP/2 request/stream")
    # Two tshark rows could decode the same HEADERS packet; count a stream once.
    selected = selected.sort_values("ts").drop_duplicates("request_key", keep="first")
    if selected.empty:
        raise ValueError("no independent NRF HTTP/2 GET evidence")
    return selected[["ts", "ip_dst", "tcp_stream", "h2_stream", "request_key"]].sort_values(
        "ts"
    ).reset_index(drop=True)


def max_count_in_sliding_window(ts: np.ndarray, seconds: float) -> int:
    """Max requests in any right-open continuous window of given duration."""
    if seconds <= 0:
        raise ValueError("positive sliding window required")
    values = np.sort(np.asarray(ts, dtype=float))
    lo, best = 0, 0
    for hi, value in enumerate(values):
        while value - values[lo] >= seconds:
            lo += 1
        best = max(best, hi - lo + 1)
    return best


def max_count_in_fixed_bins(ts: np.ndarray, origin: float, seconds: float) -> int:
    """Exact old floor-bin partition from experiment baseline epoch."""
    values = np.asarray(ts, dtype=float)
    if len(values) == 0:
        return 0
    bucket = np.floor((values - origin) / seconds).astype(int)
    return int(np.unique(bucket, return_counts=True)[1].max())


def count_trial(ts: np.ndarray, origin: float, start: float, end: float) -> dict:
    """Per-trial raw witness and bin-local thresholds, never detector decisions."""
    if not (end > start >= origin):
        raise ValueError("invalid or overlapping intervention time range")
    local = np.asarray(ts, dtype=float)
    local = local[(local >= start) & (local < end)]
    peak_fixed = max_count_in_fixed_bins(local, origin, 2.0)
    peak_sliding = max_count_in_sliding_window(local, 2.0)
    peak_one = max_count_in_sliding_window(local, 1.0)
    return {
        "raw_request_count": len(local),
        "raw_5_get_coverage": len(local) >= RAW_PRESENCE_COUNT,
        "peak_fixed_2s_count": peak_fixed,
        "peak_sliding_2s_count": peak_sliding,
        "peak_sliding_1s_count": peak_one,
        "fixed_2s_threshold_20": peak_fixed >= DEFAULT_THRESHOLD_COUNT,
        "sliding_2s_threshold_20": peak_sliding >= DEFAULT_THRESHOLD_COUNT,
        "sliding_1s_threshold_10": peak_one >= 10,
        "first_get_ts": float(local.min()) if len(local) else None,
        "last_get_ts": float(local.max()) if len(local) else None,
    }


def audit_one_artifact(directory: Path, seed: int, period: float) -> tuple[pd.DataFrame,pd.DataFrame,dict]:
    manifest = json.loads((directory / "phase_trial_manifest.json").read_text())
    verify = json.loads((directory / "kie_report_validation.json").read_text())
    if (int(manifest["seed"]) != seed or float(manifest["kie_period_s"]) != period or
        manifest["design"] != "prospective-nrf-sampler-phase-v1" or
        len(manifest["ordered_interventions"]) != 12):
        raise ValueError(f"unrecognized randomized protocol for {seed}")
    if (verify.get("all_signatures_valid") is not True or
        verify.get("all_reports_truth_paired") is not True or
        int(verify.get("invalid_signature_count", -1)) != 0 or
        int(verify["report_count"]) != int(verify["paired_collector_samples"])):
        raise ValueError(f"missing original live HMAC verification: {seed}")

    requests = _collected_requests(
        directory / "sbi_http2_events.tsv",
        directory / "open5gs_endpoints.json"
    )
    intervals = pd.read_csv(directory / "phase_intervals.csv")
    stim = intervals[intervals.phase.isin(INTERVENTIONS)].copy()
    if (len(stim) != 12 or stim.duplicated(["cycle","phase"]).any() or
        stim.start_ts.isna().any() or stim.end_ts.isna().any()):
        raise ValueError(f"incomplete stimulus timestamps for {seed}")
    delivery = pd.read_csv(directory / "burst_profiles.csv")
    if (len(delivery) != 12 or delivery.duplicated(["cycle","phase"]).any() or
        not delivery.requested.eq(delivery.successful).all()):
        raise ValueError(f"incomplete HTTP/2 stimulus delivery: {seed}")
    plan = pd.DataFrame(manifest["ordered_interventions"])
    stim = stim.merge(
        plan, on=["cycle","phase"], how="outer", validate="one_to_one", indicator=True
    )
    if not stim._merge.eq("both").all():
        raise ValueError(f"incomplete manifest join for {seed}")
    coverage = pd.read_csv(directory / "boundary_intervention_coverage.csv")
    if len(coverage) != 12 or coverage.duplicated(["cycle","phase"]).any():
        raise ValueError(f"original 2s witness coverage missing: {seed}")
    coverage = coverage[["cycle","phase","witnessed_events","fully_evaluable_events"]].rename(
        columns={"witnessed_events":"original_2s_witness_events",
                 "fully_evaluable_events":"original_2s_evaluable_events"}
    )
    stim = stim.merge(
        coverage,on=["cycle","phase"],how="outer",validate="one_to_one",indicator="_cov"
    )
    if not stim._cov.eq("both").all():
        raise ValueError(f"incomplete original independent witness: {seed}")
    align = pd.read_csv(directory / "randomized_alignment.csv")
    if len(align)!=12 or align.duplicated(["cycle","phase"]).any():
        raise ValueError(f"missing realized sampling alignment: {seed}")
    stim = stim.merge(
        align[["cycle","phase","actual_phase_fraction","phase_alignment_overrun"]],
        on=["cycle","phase"],how="outer",validate="one_to_one",indicator="_align"
    )
    if not stim._align.eq("both").all():
        raise ValueError(f"missing actual sampler phase: {seed}")
    origin = float(intervals.start_ts.min())
    output = []
    for row in stim.to_dict("records"):
        result = count_trial(
            requests.ts.to_numpy(float), origin,
            float(row["start_ts"]), float(row["end_ts"])
        )
        result.update({
            "seed": seed, "period_s": period,
            "cycle": int(row["cycle"]), "phase": row["phase"],
            "profile": row["profile"],
            "assigned_phase_fraction":float(row["assigned_phase_fraction"]),
            "actual_phase_fraction":float(row["actual_phase_fraction"]),
            "alignment_overrun":bool(row["phase_alignment_overrun"]),
            "original_2s_witness_events":int(row["original_2s_witness_events"]),
            "original_2s_evaluable_events":int(row["original_2s_evaluable_events"]),
            "captured_window_start_ts":float(row["start_ts"]),
            "captured_window_end_ts":float(row["end_ts"]),
        })
        output.append(result)
    # A raw timestamp inventory is deliberately *unlabeled*: no truth/cycle.
    raw = requests.copy()
    raw.insert(0,"seed",seed)
    raw.insert(1,"period_s",period)
    provenance = {
        "seed":seed, "period_s":period,
        "raw_independent_gets":len(raw), "signed_reports_checked_live":int(verify["report_count"]),
        "offline_hmac_reverified":False,"runner_is_independent":True
    }
    return pd.DataFrame(output), raw, provenance


def summarize(input_root: Path, output_dir: Path, *, runners: dict[int,float]=RUNNERS) -> dict:
    events, raws, provenance = [],[],[]
    for seed,period in sorted(runners.items()):
        candidates=list(input_root.rglob(f"5g-open5gs-phase-{seed}"))
        if len(candidates)!=1:
            raise ValueError(f"missing or duplicate runner {seed}: {len(candidates)}")
        result,raw,prov=audit_one_artifact(candidates[0],seed,period)
        events.append(result); raws.append(raw); provenance.append(prov)
    events_df=pd.concat(events,ignore_index=True)
    raw_df=pd.concat(raws,ignore_index=True)
    if len(events_df) != 12*len(runners):
        raise AssertionError("incomplete intervention inventory")
    output_dir.mkdir(parents=True,exist_ok=True)
    events_df.to_csv(output_dir/"raw_packet_intervention_audit.csv",index=False)
    raw_df.to_csv(output_dir/"raw_packet_timestamp_inventory.csv",index=False)
    pd.DataFrame(provenance).to_csv(output_dir/"raw_packet_provenance.csv",index=False)

    metrics=[]
    for (period,profile,phase),g in events_df.groupby(["period_s","profile","phase"]):
        metrics.append({
            "period_s":period,"profile":profile,"phase":phase,
            "planned":len(g),
            "original_confirmed":int((g.original_2s_witness_events>0).sum()),
            "raw_ge5_confirmed":int(g.raw_5_get_coverage.sum()),
            "same_bin_ge20":int(g.fixed_2s_threshold_20.sum()),
            "sliding_2s_ge20":int(g.sliding_2s_threshold_20.sum()),
            "sliding_1s_ge10":int(g.sliding_1s_threshold_10.sum()),
            "raw_without_original":int((
                g.raw_5_get_coverage & g.original_2s_witness_events.eq(0)
            ).sum()),
            "raw_absent":int((~g.raw_5_get_coverage).sum()),
            "independent_runners":int(g.seed.nunique()),
        })
    pd.DataFrame(metrics).to_csv(output_dir/"raw_packet_mechanism_by_condition.csv",index=False)
    missing=events_df[events_df.original_2s_witness_events.eq(0)]
    recovered=missing[missing.raw_5_get_coverage]
    summary={
        "classification":"prespecified external-witness measurement audit on 12 independent runners",
        "runners":len(runners),"planned_interventions":len(events_df),
        "previously_unwitnessed_2s":len(missing),
        "raw_ge5_among_previous_unwitnessed":len(recovered),
        "raw_absent_among_previous_unwitnessed":len(missing)-len(recovered),
        "raw_ge5_coverage":int(events_df.raw_5_get_coverage.sum()),
        "fixed_2s_ge20_coverage":int(events_df.fixed_2s_threshold_20.sum()),
        "sliding_2s_ge20_coverage":int(events_df.sliding_2s_threshold_20.sum()),
        "sliding_1s_ge10_coverage":int(events_df.sliding_1s_threshold_10.sum()),
        "all_stimuli_short_where_legacy_missing":bool(missing.profile.eq("short").all()),
        "original_detector_scores_changed":False,
        "all_runner_hmac_provenance_live":True,
        "limitations":[
            "Raw HTTP/2 packet GET count is independent witness coverage, NOT KIE detection.",
            "Threshold comparisons use only packet times inside exact stimulus intervals, not original full-window feature counts.",
            "Floor-bin and sliding-window statistics are predeclared explanatory diagnostics, NOT retuned detector scores.",
            "Manual caller stimulus success does not imply that every request was captured in PCAP.",
            "Only pre-existing and newly randomized laboratory environments; within-run events correlated.",
        ],
        "provenance":provenance,
    }
    (output_dir/"raw_packet_mechanism_summary.json").write_text(
        json.dumps(summary,indent=2,ensure_ascii=False),encoding="utf-8"
    )
    print(json.dumps(summary,indent=2,ensure_ascii=False))
    return summary


def main()->None:
    p=argparse.ArgumentParser()
    p.add_argument("--input-root",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--historical-only",action="store_true")
    args=p.parse_args()
    runners={k:v for k,v in RUNNERS.items() if k<30000} if args.historical_only else RUNNERS
    summarize(args.input_root,args.output_dir,runners=runners)


if __name__=="__main__":
    main()
