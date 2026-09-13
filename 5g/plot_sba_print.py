"""
plot_sba_print.py
Текстовые (ASCII) кривые обнаружимости и recall(budget) — без внешних зависимостей
для графики; при наличии matplotlib дополнительно сохраняет PNG.

Пример: python plot_sba_print.py --curve results_sba_curve.csv --adv results_sba_adversary.csv
"""

from __future__ import annotations

import argparse
import os

import pandas as pd


def ascii_curve(df: pd.DataFrame, x: str, y: str, group: str, width: int = 40) -> None:
    for gname, sub in df.groupby(group):
        print(f"\n-- {group}={gname}")
        sub = sub.groupby(x, as_index=False)[y].median().sort_values(x)
        for _, r in sub.iterrows():
            v = float(r[y])
            bar = "#" * int(round(max(min(v, 1.0), 0.0) * width))
            print(f"   {x}={r[x]:>8} | {bar:<{width}} {v:5.3f}")


def maybe_png(df: pd.DataFrame, x: str, y: str, group: str, path: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    for gname, sub in df.groupby(group):
        sub = sub.groupby(x, as_index=False)[y].median().sort_values(x)
        ax.plot(sub[x], sub[y], marker="o", label=str(gname))
    ax.set_xlabel(x), ax.set_ylabel(y), ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3), ax.legend(fontsize=8)
    fig.tight_layout(), fig.savefig(path, dpi=150)
    print(f"   [png] {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--curve", type=str, default="")
    ap.add_argument("--adv", type=str, default="")
    args = ap.parse_args()

    if args.curve and os.path.exists(args.curve):
        df = pd.read_parquet(args.curve) if args.curve.endswith(".parquet") \
            else pd.read_csv(args.curve)
        print("=== recall_combined vs hidden_calls ===")
        ascii_curve(df, "hidden_calls", "recall_combined", "via_scope")
        maybe_png(df, "hidden_calls", "recall_combined", "via_scope", "curve_detect.png")
        if "recall_attribution" in df.columns:
            print("\n=== recall_attribution vs hidden_calls ===")
            ascii_curve(df, "hidden_calls", "recall_attribution", "via_scope")

    if args.adv and os.path.exists(args.adv):
        df = pd.read_parquet(args.adv) if args.adv.endswith(".parquet") \
            else pd.read_csv(args.adv)
        print("\n=== recall_combined vs adversary budget ===")
        ascii_curve(df, "budget", "recall_combined", "family")
        maybe_png(df, "budget", "recall_combined", "family", "curve_budget.png")


if __name__ == "__main__":
    main()
