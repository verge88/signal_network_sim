from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--results",
        type=Path,
        default=Path(__file__).parent / "results" / "metrics_per_seed.csv",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "results" / "ablation_recall.png",
    )
    args = p.parse_args()

    df = pd.read_csv(args.results)
    df = df[
        (df["model"] == "iforest")
        & (df["severity"] == 1.0)
        & (df["scenario"] != "normal")
    ]
    pivot = (
        df.groupby(["scenario", "feature_set"])["recall"]
        .mean()
        .unstack("feature_set")
        .reindex(columns=["SBI", "SBI_TEMPORAL", "SBI_ZONE", "SBI_ACTIVE", "FULL"])
    )
    ax = pivot.plot(kind="bar", figsize=(14, 6))
    ax.set_ylabel("Recall")
    ax.set_xlabel("Scenario")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("5G SBA anomaly-detection ablation (Isolation Forest)")
    plt.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=160)
    print(args.output)


if __name__ == "__main__":
    main()
