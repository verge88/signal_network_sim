from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


FOCUS = [
    "compromised_lie",
    "adaptive_lie",
    "slow_drift_lie",
    "compromised_truth",
    "rapid_reset",
    "oauth_abuse",
    "nrf_poisoning",
    "cross_slice",
    "coordinated_majority",
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=Path(__file__).parent / "results_cascade" / "summary.csv")
    p.add_argument("--output", type=Path, default=Path(__file__).parent / "results_cascade" / "cascade_recall.png")
    p.add_argument("--severity", type=float, default=1.0)
    args = p.parse_args()

    df = pd.read_csv(args.input)
    sub = df[
        df["scenario"].isin(FOCUS)
        & (df["severity"].round(6) == round(args.severity, 6))
    ].copy()
    keep = [
        "iforest_sbi",
        "fusion_maxq",
        "cascade_preserve_sbi",
        "cascade_equal_fpr",
    ]
    sub = sub[sub["method"].isin(keep)]
    pivot = sub.pivot(index="scenario", columns="method", values="recall_mean").reindex(FOCUS)

    ax = pivot.plot(kind="bar", figsize=(14, 6))
    ax.set_ylabel("Recall")
    ax.set_xlabel("Scenario")
    ax.set_ylim(0.0, 1.05)
    ax.set_title(f"Guarded cascade vs SBI and decision fusion, severity={args.severity}")
    ax.legend(loc="best", fontsize=8)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=170)


if __name__ == "__main__":
    main()
