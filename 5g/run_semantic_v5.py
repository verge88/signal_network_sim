"""Targeted Byzantine-hardening ablation for SBA semantic consistency v5.

Compares q2of3, q3of5 and q4of7 under:
1) benign traffic;
2) adaptive attacks;
3) one-origin SUPPRESS;
4) one-origin INJECT;
5) soft-persistence on/off.

Outputs CSV/JSON suitable for Chapter 4 tables.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from sba_semantic_v5 import (
    AttackFamily,
    CompromiseMode,
    EvidenceOrigin,
    QUORUM_PROFILES,
    SemanticQuorumDetector,
    SemanticSbaSimulator,
    Sophistication,
    empirical_auc,
)

LOG = logging.getLogger("sba-semantic-v5")


def setup_logging() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure:
            reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )


def progress(label: str, i: int, n: int, started: float) -> None:
    step = max(1, n // 5)
    if i not in (1, n) and i % step:
        return
    elapsed = max(time.perf_counter() - started, 1e-9)
    rate = i / elapsed
    eta = (n - i) / rate if rate > 0 else float("nan")
    LOG.info("%-36s %4d/%-4d %5.1f%% %.1f win/s ETA %.1fs", label, i, n, 100*i/n, rate, eta)


def attack_ids(window):
    return set(window.attack_event_ids)


def cell_metrics(windows, results) -> Dict[str, float]:
    n = len(windows)
    window_alerts = sum(int(r.alert) for r in results)
    semantic_alerts = sum(int(r.semantic_alert) for r in results)
    transport_alerts = sum(int(r.transport_alert) for r in results)
    hidden = 0
    detected = 0
    fired = 0
    false_ids = 0
    event_latencies: List[float] = []
    campaign_delays: List[float] = []

    for w, r in zip(windows, results):
        aids = attack_ids(w)
        fids = set(r.fired_event_ids)
        hidden += len(aids)
        detected += len(aids & fids)
        fired += len(fids)
        false_ids += len(fids - aids)

        by_id = {e.event_id: e for e in w.events}
        for eid in aids & fids:
            if eid in r.event_alert_times and eid in by_id:
                event_latencies.append(max(0.0, r.event_alert_times[eid] - by_id[eid].timestamp))
        if aids and r.alert and np.isfinite(r.first_alert_time):
            first_attack = min(by_id[eid].timestamp for eid in aids if eid in by_id)
            campaign_delays.append(max(0.0, r.first_alert_time - first_attack))

    return {
        "n_windows": n,
        "window_recall": window_alerts / n if n else float("nan"),
        "semantic_window_rate": semantic_alerts / n if n else float("nan"),
        "transport_window_rate": transport_alerts / n if n else float("nan"),
        "event_recall": detected / hidden if hidden else float("nan"),
        "event_precision": detected / fired if fired else float("nan"),
        "false_attributed_events": false_ids,
        "event_latency_median_s": float(np.median(event_latencies)) if event_latencies else float("nan"),
        "event_latency_p95_s": float(np.percentile(event_latencies, 95)) if event_latencies else float("nan"),
        "campaign_ttd_median_s": float(np.median(campaign_delays)) if campaign_delays else float("nan"),
        "campaign_ttd_p95_s": float(np.percentile(campaign_delays, 95)) if campaign_delays else float("nan"),
        "unknown_fraction": float(np.nanmean([r.unknown_fraction for r in results])) if results else float("nan"),
    }


def make_calibration(seed: int, n: int):
    sim = SemanticSbaSimulator(seed=seed)
    contexts = list(sim.CONTROL_CONTEXTS)
    windows = []
    started = time.perf_counter()
    for i in range(n):
        windows.append(sim.generate_window(i, context=contexts[i % len(contexts)]))
        progress("calibration", i + 1, n, started)
    return windows


def fitted_detectors(calibration, target_fpr: float, persistence: bool):
    out = {}
    for name in QUORUM_PROFILES:
        det = SemanticQuorumDetector(
            name,
            target_fpr,
            soft_persistence_events=2,
            soft_persistence_window_s=90.0,
            enable_soft_persistence=persistence,
        ).fit(calibration)
        out[name] = det
    return out


def evaluate_benign(seed: int, n: int, detectors, persistence_label: str):
    sim = SemanticSbaSimulator(seed=seed + 1000)
    contexts = list(sim.CONTROL_CONTEXTS)
    windows = [sim.generate_window(10_000 + i, context=contexts[i % len(contexts)]) for i in range(n)]
    rows = []
    for profile, det in detectors.items():
        results = [det.score(w) for w in windows]
        m = cell_metrics(windows, results)
        rows.append({
            "mode": "benign",
            "persistence": persistence_label,
            "profile": profile,
            "fpr": m["window_recall"],
            "semantic_fpr": m["semantic_window_rate"],
            "transport_fpr": m["transport_window_rate"],
            "false_attributed_events": m["false_attributed_events"],
            "unknown_fraction": m["unknown_fraction"],
        })
        LOG.info("BENIGN %-5s persistence=%-3s FPR=%.4f semantic=%.4f transport=%.4f", profile, persistence_label, m["window_recall"], m["semantic_window_rate"], m["transport_window_rate"])
    labels = np.arange(n) % 2
    scores = [detectors["q3of5"].score(w).score for w in windows]
    return rows, empirical_auc(labels, scores)


def evaluate_attacks(seed: int, n: int, hidden: int, detectors, persistence_label: str):
    rows = []
    families = [f for f in AttackFamily if f != AttackFamily.NONE]
    for profile, det in detectors.items():
        for j, family in enumerate(families):
            sim = SemanticSbaSimulator(seed=seed + 20_000 + 100*j)
            windows = [
                sim.generate_window(
                    20_000 + i,
                    attack_family=family,
                    hidden_calls=hidden,
                    sophistication=Sophistication.ADAPTIVE,
                )
                for i in range(n)
            ]
            results = [det.score(w) for w in windows]
            m = cell_metrics(windows, results)
            rows.append({"mode": "attack", "persistence": persistence_label, "profile": profile, "family": family.value, "hidden_calls": hidden, **m})
            LOG.info("ATTACK %-5s %-20s winR=%.3f eventR=%.3f eventLat=%.3fs campaignTTD=%.3fs", profile, family.value, m["window_recall"], m["event_recall"], m["event_latency_median_s"], m["campaign_ttd_median_s"])
    return rows


def evaluate_compromise(seed: int, n: int, hidden: int, detectors, persistence_label: str):
    suppress_rows = []
    inject_rows = []
    families = [f for f in AttackFamily if f != AttackFamily.NONE]

    for profile, det in detectors.items():
        for origin in EvidenceOrigin:
            family_metrics = []
            for j, family in enumerate(families):
                sim = SemanticSbaSimulator(seed=seed + 30_000 + 1000*j + list(EvidenceOrigin).index(origin))
                windows = [
                    sim.generate_window(
                        30_000 + i,
                        attack_family=family,
                        hidden_calls=hidden,
                        sophistication=Sophistication.ADAPTIVE,
                        compromised_origins=[origin],
                        compromise_mode=CompromiseMode.SUPPRESS,
                    )
                    for i in range(n)
                ]
                results = [det.score(w) for w in windows]
                family_metrics.append(cell_metrics(windows, results))
            suppress_rows.append({
                "mode": "suppress",
                "persistence": persistence_label,
                "profile": profile,
                "origin": origin.value,
                "mean_window_recall": float(np.mean([m["window_recall"] for m in family_metrics])),
                "mean_event_recall": float(np.mean([m["event_recall"] for m in family_metrics])),
                "mean_event_latency_s": float(np.nanmean([m["event_latency_median_s"] for m in family_metrics])),
            })
            LOG.info("SUPPRESS %-5s %-8s mean winR=%.3f eventR=%.3f", profile, origin.value, suppress_rows[-1]["mean_window_recall"], suppress_rows[-1]["mean_event_recall"])

            sim = SemanticSbaSimulator(seed=seed + 40_000 + list(EvidenceOrigin).index(origin))
            windows = [
                sim.generate_window(
                    40_000 + i,
                    compromised_origins=[origin],
                    compromise_mode=CompromiseMode.INJECT,
                )
                for i in range(n)
            ]
            results = [det.score(w) for w in windows]
            m = cell_metrics(windows, results)
            inject_rows.append({
                "mode": "inject",
                "persistence": persistence_label,
                "profile": profile,
                "origin": origin.value,
                "byzantine_fpr": m["window_recall"],
                "semantic_byzantine_fpr": m["semantic_window_rate"],
                "transport_byzantine_fpr": m["transport_window_rate"],
                "false_attributed_events": m["false_attributed_events"],
            })
            LOG.info("INJECT   %-5s %-8s ByzFPR=%.4f semantic=%.4f transport=%.4f", profile, origin.value, m["window_recall"], m["semantic_window_rate"], m["transport_window_rate"])

    return suppress_rows, inject_rows


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="SBA v5 Byzantine quorum ablation")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--calib-windows", type=int, default=1200)
    p.add_argument("--eval-windows", type=int, default=200)
    p.add_argument("--trust-windows", type=int, default=200)
    p.add_argument("--hidden", type=int, default=5)
    p.add_argument("--target-fpr", type=float, default=0.01)
    p.add_argument("--out-dir", default="runs/sba_semantic_v5/byzantine_ablation")
    a = p.parse_args(argv)
    setup_logging()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    LOG.info("V5 Byzantine hardening | seed=%d | calib=%d | eval=%d | trust=%d | h=%d", a.seed, a.calib_windows, a.eval_windows, a.trust_windows, a.hidden)
    calibration = make_calibration(a.seed, a.calib_windows)

    all_benign = []
    all_attacks = []
    all_suppress = []
    all_inject = []
    null_auc = {}

    for persistence in (False, True):
        label = "on" if persistence else "off"
        LOG.info("=" * 90)
        LOG.info("SOFT PERSISTENCE %s", label.upper())
        dets = fitted_detectors(calibration, a.target_fpr, persistence)
        benign, auc = evaluate_benign(a.seed, a.eval_windows, dets, label)
        attacks = evaluate_attacks(a.seed, a.eval_windows, a.hidden, dets, label)
        suppress, inject = evaluate_compromise(a.seed, a.trust_windows, a.hidden, dets, label)
        all_benign.extend(benign)
        all_attacks.extend(attacks)
        all_suppress.extend(suppress)
        all_inject.extend(inject)
        null_auc[label] = auc

    benign_df = pd.DataFrame(all_benign)
    attack_df = pd.DataFrame(all_attacks)
    suppress_df = pd.DataFrame(all_suppress)
    inject_df = pd.DataFrame(all_inject)
    benign_df.to_csv(out / "benign_fpr.csv", index=False)
    attack_df.to_csv(out / "adaptive_attack_recall.csv", index=False)
    suppress_df.to_csv(out / "suppression_ablation.csv", index=False)
    inject_df.to_csv(out / "byzantine_injection_fpr.csv", index=False)

    summary_rows = []
    for persistence in ("off", "on"):
        for profile in QUORUM_PROFILES:
            b = benign_df[(benign_df.persistence == persistence) & (benign_df.profile == profile)].iloc[0]
            arows = attack_df[(attack_df.persistence == persistence) & (attack_df.profile == profile)]
            srows = suppress_df[(suppress_df.persistence == persistence) & (suppress_df.profile == profile)]
            irows = inject_df[(inject_df.persistence == persistence) & (inject_df.profile == profile)]
            summary_rows.append({
                "persistence": persistence,
                "profile": profile,
                "benign_fpr": float(b.fpr),
                "semantic_fpr": float(b.semantic_fpr),
                "mean_attack_window_recall": float(arows.window_recall.mean()),
                "mean_attack_event_recall": float(arows.event_recall.mean()),
                "mean_suppress_window_recall": float(srows.mean_window_recall.mean()),
                "worst_suppress_window_recall": float(srows.mean_window_recall.min()),
                "mean_byzantine_fpr": float(irows.byzantine_fpr.mean()),
                "worst_byzantine_fpr": float(irows.byzantine_fpr.max()),
                "mean_semantic_byzantine_fpr": float(irows.semantic_byzantine_fpr.mean()),
                "worst_semantic_byzantine_fpr": float(irows.semantic_byzantine_fpr.max()),
                "null_auc": float(null_auc[persistence]),
            })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out / "quorum_ablation_summary.csv", index=False)
    summary = {
        "seed": a.seed,
        "hidden": a.hidden,
        "profiles": list(QUORUM_PROFILES),
        "null_auc": null_auc,
        "rows": summary_rows,
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    LOG.info("=" * 90)
    LOG.info("FINAL QUORUM ABLATION")
    for row in summary_rows:
        LOG.info("%-3s %-5s attackR=%.3f suppressWorst=%.3f ByzFPR mean/worst=%.3f/%.3f semanticWorst=%.3f", row["persistence"], row["profile"], row["mean_attack_window_recall"], row["worst_suppress_window_recall"], row["mean_byzantine_fpr"], row["worst_byzantine_fpr"], row["worst_semantic_byzantine_fpr"])
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
