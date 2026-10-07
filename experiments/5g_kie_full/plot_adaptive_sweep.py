from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _label(method: str) -> str:
    if method == "iforest_sbi":
        return "SBI-only"
    if method == "fusion_maxq":
        return "fusion_maxq"
    if method == "iforest_full_legacy":
        return "legacy FULL"
    if method.startswith("cascade_active_"):
        return method.replace("cascade_active_", "cascade active=")
    return method


def plot_recall_alpha(summary: pd.DataFrame, output: Path, severity: float) -> None:
    sub = summary[
        (summary["scenario"] == "adaptive_lie")
        & (summary["severity"].round(6) == round(severity, 6))
    ].copy()

    selected = [
        "iforest_sbi",
        "iforest_full_legacy",
        "fusion_maxq",
        "cascade_active_0.00010",
        "cascade_active_0.00100",
        "cascade_active_0.00500",
    ]
    fig, ax = plt.subplots(figsize=(11, 6))
    for method in selected:
        m = sub[sub["method"] == method].sort_values("report_alpha")
        if m.empty:
            continue
        ax.plot(m["report_alpha"], m["recall_mean"], marker="o", label=_label(method))
    ax.set_xlabel("Attacker report mimicry alpha")
    ax.set_ylabel("Recall")
    ax.set_ylim(0.0, 1.05)
    ax.set_xlim(-0.02, 1.02)
    ax.set_title(f"Adaptive attacker robustness, severity={severity}")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170)
    plt.close(fig)


def plot_pareto(pareto: pd.DataFrame, output: Path, severity: float) -> None:
    alphas = [0.75, 0.9, 0.95, 0.99, 1.0]
    fig, ax = plt.subplots(figsize=(10, 6))
    for alpha in alphas:
        m = pareto[
            (pareto["severity"].round(6) == round(severity, 6))
            & (pareto["report_alpha"].round(6) == round(alpha, 6))
        ].sort_values("normal_fpr_mean")
        if m.empty:
            continue
        ax.plot(
            100.0 * m["normal_fpr_mean"],
            m["recall"],
            marker="o",
            label=f"alpha={alpha:g}",
        )
    ax.set_xlabel("Normal false-positive rate, %")
    ax.set_ylabel("Recall")
    ax.set_ylim(0.0, 1.05)
    ax.set_title(f"ACTIVE budget Pareto sweep, severity={severity}")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    base = Path(__file__).parent / "results_adaptive_sweep"
    p.add_argument("--summary", type=Path, default=base / "summary.csv")
    p.add_argument("--pareto", type=Path, default=base / "pareto.csv")
    p.add_argument("--output-dir", type=Path, default=base)
    p.add_argument("--severity", type=float, default=1.0)
    args = p.parse_args()

    summary = pd.read_csv(args.summary)
    pareto = pd.read_csv(args.pareto)
    plot_recall_alpha(
        summary,
        args.output_dir / "recall_vs_report_alpha.png",
        args.severity,
    )
    plot_pareto(
        pareto,
        args.output_dir / "pareto_recall_vs_fpr.png",
        args.severity,
    )


if __name__ == "__main__":
    main()
