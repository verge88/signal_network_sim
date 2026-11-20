"""Exact same-window, same-label comparison of all archived V5 online methods + V11/V12.

Archived online alarms are evaluated as-is, not retrained or reconstructed.
V11/V12 are from a separate, unchanged, feature-only chronological replay
verified in v12_ood_audit.py. Results are historical posthoc comparisons, not
prospectively validated cross-scenario detector generalization.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

SEEDS = (90401, 90402, 90405)
KEYS = ["run_id", "window_id", "nf_id"]
SCENARIOS = ("S1", "S2", "S3", "S4", "S5", "C", "D", "F", "G", "R")
ARCHIVED_METHODS = (
    "IF_CONTEXT", "IF_FULL", "IF_SBI", "TG_RESCUE", "TG_V4", "TG_V5",
    "hybrid", "hybrid_tail", "hybrid_tail_without_KIE", "hybrid_without_KIE",
    "robust_tail", "rules",
)
REPLAYED_METHODS = ("TG_V11", "TG_V12")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metric_counts(truth, pred):
    tr = truth.astype(bool)
    pr = pred.astype(bool)
    return {
        "tp": int((tr & pr).sum()),
        "fn": int((tr & ~pr).sum()),
        "fp": int((~tr & pr).sum()),
        "tn": int((~tr & ~pr).sum()),
    }


def finalize(d):
    a = dict(d)
    a["recall"] = a["tp"] / (a["tp"] + a["fn"]) if a["tp"] + a["fn"] else None
    a["fpr"] = a["fp"] / (a["fp"] + a["tn"]) if a["fp"] + a["tn"] else None
    a["precision"] = a["tp"] / (a["tp"] + a["fp"]) if a["tp"] + a["fp"] else None
    return a


def archival_seed(root, audit):
    """Analyze retained online predictions against their exact historical truth."""
    root = Path(root)
    manifest = json.loads((root / "test" / "manifest.json").read_text())
    seed = int(manifest["seed"])
    if seed not in SEEDS or manifest["windows"] != 3600 or manifest["actors"] != 32:
        raise ValueError("Unexpected archive seed or capture dimensions")
    if audit["seed"] != seed or audit["source_run_id"] != manifest["run_id"]:
        raise ValueError("OOD replay report refers to a different source acquisition")
    labels_path = root / "test" / "labels.csv"
    if sha(labels_path) != audit["source_labels_sha256"]:
        raise ValueError("Labels differ from frozen V12 OOD replay")
    labels = pd.read_csv(labels_path, usecols=KEYS + ["attack", "scenario", "episode_id"],
                         dtype={"run_id":str, "nf_id":str, "episode_id":str}).fillna({"episode_id":""})
    if (len(labels) != 115200 or labels.duplicated(KEYS).any()
            or labels.run_id.nunique() != 1
            or labels.window_id.nunique() != 3600
            or labels.nf_id.nunique() != 32
            or not set(SCENARIOS).issubset(set(labels.loc[labels.attack.astype(bool), "scenario"]))):
        raise ValueError("Incomplete or wrong archive label keys")
    online_path = root / "test" / "online_predictions.csv"
    online = pd.read_csv(online_path, usecols=KEYS + ["method","alarm"],
                         dtype={"run_id":str, "nf_id":str, "method":str, "alarm":"int8"})
    if (len(online) != 115200*len(ARCHIVED_METHODS)
            or set(online.method) != set(ARCHIVED_METHODS)
            or online.duplicated(KEYS + ["method"]).any()
            or not set(online.alarm).issubset({0,1})):
        raise ValueError("Missing, unexpected, duplicated, or invalid historical alarms")
    joined = online.merge(labels, on=KEYS, validate="many_to_one")
    if len(joined) != len(online):
        raise ValueError("Online and labeled keys have different coverage")

    methods = {}
    scenarios = {}
    for method, chunk in joined.groupby("method", sort=True):
        if len(chunk) != len(labels):
            raise ValueError("Incomplete source method "+method)
        methods[method] = finalize(metric_counts(chunk.attack, chunk.alarm))
        scenarios[method] = {}
        for scenario, subset in chunk[chunk.attack.astype(bool)].groupby("scenario"):
            scenarios[method][scenario] = {
                "positive": len(subset),
                "tp": int(subset.alarm.sum()),
                "episodes": int(subset.episode_id.nunique()),
                "episodes_detected": int(subset.groupby("episode_id").alarm.max().sum()),
            }
        if set(scenarios[method]) != set(SCENARIOS):
            raise ValueError("Missing scenario for "+method)

    for method in REPLAYED_METHODS:
        methods[method] = audit["confusion"][method]
        scenarios[method] = {
            scenario: {
                "positive": data["attacked_nf_windows"],
                "tp": data[method]["detected_nf_windows"],
                "episodes": data["episodes"],
                "episodes_detected": data[method]["episodes_detected"],
            }
            for scenario, data in audit["scenarios"].items()
        }
    # Precisely the same labeled positive-window totals for all 14 methods.
    expected = {scenario: int((labels.attack.astype(bool)
                  & labels.scenario.eq(scenario)).sum()) for scenario in SCENARIOS}
    if any({scenario: vals[scenario]["positive"] for scenario in SCENARIOS} != expected
           for vals in scenarios.values()):
        raise ValueError("Scenario denominator inconsistent across methods")
    for method in REPLAYED_METHODS:
        if sum(methods[method][k] for k in ("tp","fn","fp","tn")) != len(labels):
            raise ValueError("Incorrect V11/V12 OOD replay dimensions")
    return {
        "seed":seed, "source_run_id":manifest["run_id"],
        "online_predictions_sha256":sha(online_path),
        "labels_sha256":sha(labels_path), "methods":methods, "scenarios":scenarios,
    }


def compare(inputs, reports, output):
    inputs, reports, output = Path(inputs), Path(reports), Path(output)
    seed_reports = {}
    for seed in SEEDS:
        audit = json.loads((reports / str(seed) / "report.json").read_text())
        seed_reports[str(seed)] = archival_seed(inputs / str(seed), audit)
    names = list(ARCHIVED_METHODS) + list(REPLAYED_METHODS)
    pooled = {}
    by_scenario = {}
    for method in names:
        counts = {k:sum(report["methods"][method][k] for report in seed_reports.values())
                  for k in ("tp", "fn", "fp", "tn")}
        pooled[method] = finalize(counts)
        by_scenario[method] = {}
        for scenario in SCENARIOS:
            pos = sum(report["scenarios"][method][scenario]["positive"]
                      for report in seed_reports.values())
            tp = sum(report["scenarios"][method][scenario]["tp"]
                     for report in seed_reports.values())
            episodes = sum(report["scenarios"][method][scenario]["episodes"]
                           for report in seed_reports.values())
            episodes_found = sum(report["scenarios"][method][scenario]["episodes_detected"]
                                 for report in seed_reports.values())
            by_scenario[method][scenario] = {
                "positive":pos, "tp":tp, "recall":tp/pos if pos else None,
                "episodes":episodes, "episodes_detected":episodes_found,
            }
    baseline = pooled[names[0]]
    if any((v["tp"]+v["fn"] != baseline["tp"]+baseline["fn"]
            or v["fp"]+v["tn"] != baseline["fp"]+baseline["tn"])
           for v in pooled.values()):
        raise ValueError("Metrics were not computed over the same NF-windows")
    if any({s:d["positive"] for s,d in by_scenario[m].items()} !=
           {s:d["positive"] for s,d in by_scenario[names[0]].items()}
           for m in names):
        raise ValueError("Scenario denominators differ between methods")
    result = {
        "phase":"posthoc_archive_same_window_comparison", "source_seeds":list(SEEDS),
        "methods_online_archived":list(ARCHIVED_METHODS),
        "methods_feature_only_replay":list(REPLAYED_METHODS),
        "total_nf_windows":sum(v["tp"]+v["fn"]+v["fp"]+v["tn"]
                                for v in [pooled[names[0]]]),
        "replayed_versions_not_available":"TG_V1, TG_V2, TG_V3, TG_V6, TG_V7, TG_V8, TG_V9, TG_V10: no original decisions in these source archives",
        "pooled":pooled, "by_scenario":by_scenario, "per_seed":seed_reports,
        "caveats":[
            "Historical online predictions for 12 original methods, causal offline replay for V11/V12; not an online latency comparison",
            "No estimates for V1–V3 and V6–V10 on these exact captures",
            "All methods compared on exactly 345600 same NF-windows (2490 positive; 343110 negative)",
            "Posthoc analysis, not independent generalization confirmation or valid model selection",
        ],
    }
    output.mkdir(parents=True,exist_ok=True)
    (output / "all-methods.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    ranking = sorted(names, key=lambda n:pooled[n]["recall"], reverse=True)
    lines = ["method,tp,fn,fp,tn,recall,fpr,precision"]
    for method in ranking:
        m = pooled[method]
        lines.append(",".join([method]+[str(m[k]) for k in
                                   ("tp","fn","fp","tn","recall","fpr","precision")]))
    (output / "all-methods.csv").write_text("\n".join(lines)+"\n")
    with (output / "README.md").open("w") as f:
        f.write("# 90401 / 90402 / 90405: exact same-window all-method comparison\n\n")
        f.write("Historical online method scores and frozen V11/V12 causal replays.\n")
        f.write("This is **posthoc OOD**, not V12 prospective S9 confirmation.\n\n")
        f.write("| Method | NF-window recall | FPR | Precision | TP | FP |\n")
        f.write("|---|---:|---:|---:|---:|---:|\n")
        for method in ranking:
            m = pooled[method]
            f.write(f"| {method} | {m['recall']:.2%} | {m['fpr']:.3%} | "
                    f"{m['precision']:.2%} | {m['tp']} | {m['fp']} |\n")
        f.write("\nNo original TG_V1–V3 / TG_V6–V10 online predictions exist in these source runs; "
                "their results cannot be fabricated or substituted from different seeds.\n")
    print(json.dumps({"phase":result["phase"],"source_seeds":list(SEEDS),
                     "pooled":pooled,"number_of_methods":len(names)},allow_nan=False),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("inputs",type=Path)
    p.add_argument("reports",type=Path)
    p.add_argument("output",type=Path)
    a=p.parse_args()
    compare(a.inputs,a.reports,a.output)

if __name__=="__main__":
    main()
