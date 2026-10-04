"""Run the event-level 5G SBA semantic-consistency experiment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sba_semantic_v3 import (
    AttackFamily,
    CompromiseMode,
    SemanticQuorumDetector,
    SemanticSbaSimulator,
    TrustDomain,
    empirical_auc,
)


def run(seed: int, calib_windows: int, eval_windows: int, hidden: int, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    sim = SemanticSbaSimulator(seed=seed)
    contexts = list(sim.CONTROL_CONTEXTS)

    calib = [
        sim.generate_window(i, context=contexts[i % len(contexts)])
        for i in range(calib_windows)
    ]
    det = SemanticQuorumDetector(target_fpr=0.01).fit(calib)

    benign = [
        sim.generate_window(10_000 + i, context=contexts[i % len(contexts)])
        for i in range(eval_windows)
    ]

    fpr_rows = []
    for context in contexts:
        part = [w for w in benign if w.context == context]
        fpr_rows.append({
            "context": context,
            "n": len(part),
            "fpr": float(np.mean([det.score(w).alert for w in part])) if part else np.nan,
        })
    fpr_rows.append({
        "context": "ALL",
        "n": len(benign),
        "fpr": float(np.mean([det.score(w).alert for w in benign])),
    })
    pd.DataFrame(fpr_rows).to_csv(out_dir / "fpr_by_context.csv", index=False)

    attack_rows = []
    wid = 20_000
    for family in AttackFamily:
        if family == AttackFamily.NONE:
            continue
        alerts = []
        facts = {}
        for _ in range(eval_windows):
            w = sim.generate_window(wid, attack_family=family, hidden_calls=hidden)
            wid += 1
            r = det.score(w)
            alerts.append(r.alert)
            for fact in r.fired_facts:
                facts[fact.value] = facts.get(fact.value, 0) + 1
        attack_rows.append({
            "family": family.value,
            "hidden_calls": hidden,
            "n": eval_windows,
            "recall": float(np.mean(alerts)),
            "dominant_facts": json.dumps(facts, ensure_ascii=False, sort_keys=True),
        })
    pd.DataFrame(attack_rows).to_csv(out_dir / "per_attack.csv", index=False)

    trust_rows = []
    for domain in TrustDomain:
        for family in AttackFamily:
            if family == AttackFamily.NONE:
                continue
            alerts = []
            for _ in range(eval_windows):
                w = sim.generate_window(
                    wid,
                    attack_family=family,
                    hidden_calls=hidden,
                    compromised_domains=[domain],
                    compromise_mode=CompromiseMode.SUPPRESS,
                )
                wid += 1
                alerts.append(det.score(w).alert)
            trust_rows.append({
                "domain": domain.value,
                "family": family.value,
                "recall": float(np.mean(alerts)),
            })
    pd.DataFrame(trust_rows).to_csv(out_dir / "trust_domain_suppression.csv", index=False)

    rng = np.random.default_rng(seed + 424242)
    labels = rng.integers(0, 2, size=len(benign))
    scores = [det.score(w).score for w in benign]
    auc = empirical_auc(labels, scores)

    summary = {
        "seed": seed,
        "calib_windows": calib_windows,
        "eval_windows": eval_windows,
        "hidden_calls": hidden,
        "null_auc": auc,
        "overall_fpr": fpr_rows[-1]["fpr"],
        "mean_attack_recall": float(pd.DataFrame(attack_rows)["recall"].mean()),
        "transport_thresholds": det.transport_thresholds,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--calib-windows", type=int, default=120)
    p.add_argument("--eval-windows", type=int, default=80)
    p.add_argument("--hidden", type=int, default=40)
    p.add_argument("--out-dir", default="runs/sba_semantic_v3")
    p.add_argument("--quick", action="store_true")
    a = p.parse_args(argv)
    if a.quick:
        a.calib_windows = min(a.calib_windows, 36)
        a.eval_windows = min(a.eval_windows, 20)
    summary = run(a.seed, a.calib_windows, a.eval_windows, a.hidden, Path(a.out_dir))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
