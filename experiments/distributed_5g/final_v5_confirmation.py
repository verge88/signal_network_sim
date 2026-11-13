"""Aggregate the preregistered ten-seed final V5 confirmation.

The unit of statistical replication is the independent prospective run (seed),
not the NF-window. Window counts are pooled only for descriptive confusion
matrices/FPR/recall. Paired inference is performed across seeds.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

EXPECTED_SEEDS=tuple(range(90401,90411))
METHODS=("WITHOUT_MINING","BUDGETED_WITHOUT_MINING","TG_V5","FULL_HYBRID")
FPR_TARGET=.001
BOOTSTRAP_REPS=20000
BOOTSTRAP_SEED=20261113


def metric(report,name):
    rows=[row for row in report["metrics"] if row["method"]==name]
    if len(rows)!=1:
        raise ValueError(f"Expected one metric row for {name}, got {len(rows)}")
    return rows[0]


def pooled(rows):
    result={k:int(sum(int(row[k]) for row in rows)) for k in ("tp","fp","fn","tn")}
    result["recall"]=result["tp"]/(result["tp"]+result["fn"])
    result["fpr"]=result["fp"]/(result["fp"]+result["tn"])
    result["precision"]=result["tp"]/(result["tp"]+result["fp"]) if result["tp"]+result["fp"] else None
    return result


def bootstrap_mean_ci(values,reps=BOOTSTRAP_REPS,seed=BOOTSTRAP_SEED):
    values=np.asarray(values,dtype=float)
    rng=np.random.default_rng(seed)
    draws=rng.integers(0,len(values),size=(reps,len(values)))
    means=values[draws].mean(axis=1)
    lo,hi=np.quantile(means,[.025,.975])
    return {"mean":float(values.mean()),"ci95":[float(lo),float(hi)]}


def signed_wilcoxon_greater(values):
    values=np.asarray(values,dtype=float)
    if np.allclose(values,0):
        return {"statistic":0.0,"pvalue_one_sided":1.0}
    result=wilcoxon(values,alternative="greater",zero_method="wilcox",method="auto")
    return {"statistic":float(result.statistic),"pvalue_one_sided":float(result.pvalue)}


def find_seed_from_report(report):
    seed=int(report["seed"])
    if seed not in EXPECTED_SEEDS:
        raise ValueError(f"Unexpected final-confirmation seed {seed}")
    return seed


def episode_summary(frame,method):
    part=frame[(frame.method==method)&(frame.scenario=="S5")].copy()
    if len(part)!=4:
        raise ValueError(f"Expected four S5 episodes for {method}, got {len(part)}")
    detected=int(part.alarm_present.astype(bool).sum())
    cyclic=part[part.episode.astype(str).str.endswith("-1820")]
    if len(cyclic)!=1:
        raise ValueError(f"Expected one cyclic S5 episode, got {len(cyclic)}")
    return {
        "s5_detected":detected,
        "s5_total":4,
        "cyclic_s5_detected":int(bool(cyclic.alarm_present.iloc[0])),
        "cyclic_s5_new_alarm":int(bool(cyclic.new_alarm.iloc[0])),
        "cyclic_s5_delay_windows":(
            None if pd.isna(cyclic.new_alarm_delay_windows.iloc[0])
            else int(cyclic.new_alarm_delay_windows.iloc[0])
        ),
    }


def run(root,output):
    root=Path(root); output=Path(output); output.mkdir(parents=True,exist_ok=True)
    report_paths=list(root.rglob("report.json"))
    if len(report_paths)!=10:
        raise ValueError(f"Expected 10 report.json files, got {len(report_paths)}")

    reports={}
    episodes={}
    for path in report_paths:
        report=json.loads(path.read_text())
        seed=find_seed_from_report(report)
        if seed in reports:
            raise ValueError(f"Duplicate seed {seed}")
        if not report.get("audit_pass"):
            raise ValueError(f"Audit failed for {seed}")
        if report.get("detector")!="temporal_graph_logistic_v5":
            raise ValueError(f"Unexpected detector for {seed}: {report.get('detector')}")
        reports[seed]=report
        episode_path=path.with_name("episodes.csv")
        if not episode_path.exists():
            raise ValueError(f"Missing episodes.csv next to {path}")
        episodes[seed]=pd.read_csv(episode_path)

    if tuple(sorted(reports))!=EXPECTED_SEEDS:
        raise ValueError(f"Seed set mismatch: {sorted(reports)}")

    per_seed=[]
    method_rows={name:[] for name in METHODS}
    for seed in EXPECTED_SEEDS:
        report=reports[seed]
        row={"seed":seed,"audit_pass":True}
        for name in METHODS:
            m=metric(report,name)
            method_rows[name].append(m)
            prefix=name.lower()
            for key in ("tp","fp","fn","tn","recall","fpr","precision","episodes","episodes_with_alarm"):
                if key in m:
                    row[f"{prefix}_{key}"]=m[key]
        for name in METHODS:
            ep=episode_summary(episodes[seed],name)
            for key,value in ep.items():
                row[f"{name.lower()}_{key}"]=value
        row["full_fpr_target_met"]=bool(metric(report,"FULL_HYBRID")["fpr"]<=FPR_TARGET)
        row["recall_gain_vs_without"]=float(metric(report,"FULL_HYBRID")["recall"]-metric(report,"WITHOUT_MINING")["recall"])
        row["fpr_delta_vs_without"]=float(metric(report,"FULL_HYBRID")["fpr"]-metric(report,"WITHOUT_MINING")["fpr"])
        row["s5_episode_gain_vs_without"]=int(
            episode_summary(episodes[seed],"FULL_HYBRID")["s5_detected"]-
            episode_summary(episodes[seed],"WITHOUT_MINING")["s5_detected"])
        per_seed.append(row)

    per_seed_frame=pd.DataFrame(per_seed)
    per_seed_frame.to_csv(output/"per_seed.csv",index=False)

    pooled_metrics={name:pooled(method_rows[name]) for name in METHODS}
    recall_gains=per_seed_frame.recall_gain_vs_without.to_numpy(float)
    fpr_deltas=per_seed_frame.fpr_delta_vs_without.to_numpy(float)
    s5_gains=per_seed_frame.s5_episode_gain_vs_without.to_numpy(float)

    s5_totals={}
    for method in METHODS:
        detected=sum(int(per_seed_frame.loc[per_seed_frame.seed==seed,f"{method.lower()}_s5_detected"].iloc[0])
                     for seed in EXPECTED_SEEDS)
        cyclic=sum(int(per_seed_frame.loc[per_seed_frame.seed==seed,f"{method.lower()}_cyclic_s5_detected"].iloc[0])
                   for seed in EXPECTED_SEEDS)
        s5_totals[method]={
            "detected":detected,"total":40,"episode_recall":detected/40,
            "cyclic_detected":cyclic,"cyclic_total":10,"cyclic_episode_recall":cyclic/10,
        }

    primary={
        "audit_10_of_10":bool(per_seed_frame.audit_pass.all()),
        "fpr_target_10_of_10":bool(per_seed_frame.full_fpr_target_met.all()),
        "pooled_recall_gain_positive":bool(
            pooled_metrics["FULL_HYBRID"]["recall"]>pooled_metrics["WITHOUT_MINING"]["recall"]),
        "paired_recall_gain_wilcoxon_p_lt_0_05":False,
        "s5_episode_recall_gain_positive":bool(
            s5_totals["FULL_HYBRID"]["episode_recall"]>
            s5_totals["WITHOUT_MINING"]["episode_recall"]),
    }
    recall_test=signed_wilcoxon_greater(recall_gains)
    primary["paired_recall_gain_wilcoxon_p_lt_0_05"]=bool(recall_test["pvalue_one_sided"]<.05)
    primary["all_primary_criteria_met"]=bool(all(primary.values()))

    summary={
        "preregistration":{
            "seeds":list(EXPECTED_SEEDS),
            "fpr_target":FPR_TARGET,
            "statistical_unit":"prospective seed/run",
            "primary_criteria":[
                "audit pass on all 10 runs",
                "FULL_HYBRID FPR <= 0.001 on each run",
                "pooled FULL_HYBRID recall > pooled WITHOUT_MINING recall",
                "one-sided paired Wilcoxon across seeds for recall gain p < 0.05",
                "FULL_HYBRID S5 episode recall > WITHOUT_MINING S5 episode recall",
            ],
            "cyclic_s5":"secondary descriptive endpoint; no 100% requirement",
        },
        "pooled_metrics":pooled_metrics,
        "s5_episode_metrics":s5_totals,
        "paired_statistics":{
            "recall_gain":bootstrap_mean_ci(recall_gains),
            "recall_gain_wilcoxon":recall_test,
            "fpr_delta":bootstrap_mean_ci(fpr_deltas),
            "s5_episode_gain_per_seed":bootstrap_mean_ci(s5_gains),
        },
        "primary_acceptance":primary,
        "per_seed_records":per_seed,
        "limitations":[
            "NF-window observations within a run are temporally and topologically dependent.",
            "Paired inference therefore uses the ten independent prospective runs as replicates.",
            "The laboratory Open5GS setting does not establish performance on operator traffic.",
            "Cyclic-S5 is reported separately as a secondary endpoint to avoid post-hoc tuning toward 100% detection.",
        ],
    }
    (output/"final_summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")

    lines=[
        "# Final V5 prospective confirmation",
        "",
        f"Runs: {len(EXPECTED_SEEDS)} ({EXPECTED_SEEDS[0]}–{EXPECTED_SEEDS[-1]})",
        f"Primary acceptance: {'PASS' if primary['all_primary_criteria_met'] else 'FAIL'}",
        f"FPR target runs: {int(per_seed_frame.full_fpr_target_met.sum())}/10",
        "",
        "## Pooled metrics",
        "",
        "| Method | TP | FP | Recall | Precision | FPR |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in METHODS:
        m=pooled_metrics[name]
        lines.append(f"| {name} | {m['tp']} | {m['fp']} | {m['recall']:.6f} | {m['precision']:.6f} | {m['fpr']:.6f} |")
    lines += [
        "",
        "## S5 episodes",
        "",
        "| Method | S5 detected | S5 recall | Cyclic-S5 detected | Cyclic-S5 recall |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in METHODS:
        s=s5_totals[name]
        lines.append(f"| {name} | {s['detected']}/40 | {s['episode_recall']:.3f} | {s['cyclic_detected']}/10 | {s['cyclic_episode_recall']:.3f} |")
    lines += [
        "",
        "## Paired seed-level inference",
        "",
        f"Recall gain mean: {summary['paired_statistics']['recall_gain']['mean']:.6f}",
        f"Recall gain bootstrap 95% CI: {summary['paired_statistics']['recall_gain']['ci95']}",
        f"Recall gain Wilcoxon one-sided p: {recall_test['pvalue_one_sided']:.6g}",
        f"FPR delta mean: {summary['paired_statistics']['fpr_delta']['mean']:.6f}",
        f"FPR delta bootstrap 95% CI: {summary['paired_statistics']['fpr_delta']['ci95']}",
    ]
    (output/"final_summary.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({
        "primary_acceptance":primary,
        "pooled_full":pooled_metrics["FULL_HYBRID"],
        "pooled_without":pooled_metrics["WITHOUT_MINING"],
        "s5_full":s5_totals["FULL_HYBRID"],
        "s5_without":s5_totals["WITHOUT_MINING"],
        "paired_statistics":summary["paired_statistics"],
    },allow_nan=False),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--root",required=True)
    p.add_argument("--output",required=True)
    a=p.parse_args()
    run(a.root,a.output)
