"""
run_seeds_sba.py
Мультисидовый прогон с агрегацией (median + IQR) и ПАТЧ 5: fallback на CSV,
если parquet-движок недоступен.

Пример:  python run_seeds_sba.py --seeds 5 --out results_sba
         python run_seeds_sba.py --seeds 2 --quick --out results_smoke
"""

from __future__ import annotations

import argparse
import os
import traceback
from typing import Dict, List

import numpy as np
import pandas as pd

from sba_eval import run_one


def _write(df: pd.DataFrame, path_noext: str) -> str:
    """ПАТЧ 5: пробуем parquet, при любой ошибке пишем CSV."""
    try:
        path = f"{path_noext}.parquet"
        df.to_parquet(path, index=False)
        return path
    except Exception as exc:
        path = f"{path_noext}.csv"
        df.to_csv(path, index=False)
        print(f"  [warn] parquet недоступен ({type(exc).__name__}), записан {path}")
        return path


def aggregate(df: pd.DataFrame, keys: List[str], vals: List[str]) -> pd.DataFrame:
    vals = [v for v in vals if v in df.columns]
    g = df.groupby(keys, as_index=False)[vals].agg(["median", "mean", "std", "count"])
    g.columns = ["_".join([c for c in col if c]) for col in g.columns.to_flat_index()]
    return g.reset_index() if g.index.name else g


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--no-adversary", action="store_true")
    ap.add_argument("--out", type=str, default="results_sba")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    all_groups, all_ctrl, all_curve, all_lofo, all_adv, all_diag = [], [], [], [], [], []

    for k in range(args.seeds):
        seed = args.seed0 + k
        print(f"\n[seed {seed}] запуск ({'quick' if args.quick else 'full'}) ...")
        try:
            res = run_one(seed=seed, quick=args.quick, alpha=args.alpha,
                          with_adversary=not args.no_adversary, with_sensitivity=False)
        except Exception:
            print(f"[seed {seed}] ОШИБКА:\n{traceback.format_exc()}")
            continue

        for name, store in (("by_group", all_groups), ("controls", all_ctrl),
                            ("curve", all_curve), ("lofo", all_lofo),
                            ("adversary", all_adv)):
            df = res[name]
            if isinstance(df, pd.DataFrame) and not df.empty:
                df = df.copy()
                df["seed"] = seed
                store.append(df)

        d = dict(res["diagnostics"])
        d["seed"] = seed
        d["n_windows"] = float(res["dataset"].df.shape[0])
        all_diag.append(d)
        print(f"[seed {seed}] mean_sigma={d.get('mean_sigma', float('nan')):.3f} "
              f"worst_benign_fpr={d.get('worst_benign_fpr_combined', float('nan')):.3f} "
              f"theory_err={d.get('mean_abs_theory_error', float('nan')):.3f}")

    if all_diag:
        diag = pd.DataFrame(all_diag)
        _write(diag, f"{args.out}_diagnostics")
        print("\n=== diagnostics по сидам (median) ===")
        print(diag.median(numeric_only=True).to_string(float_format=lambda v: f"{v:10.4f}"))

    if all_groups:
        df = pd.concat(all_groups, ignore_index=True)
        _write(df, f"{args.out}_by_group")
        print("\n=== recall по (family x via_scope), median по сидам ===")
        print(df.groupby(["family", "via_scope"])[
            ["recall_invariant", "recall_attribution", "recall_combined"]]
            .median().to_string(float_format=lambda v: f"{v:8.3f}"))

    if all_ctrl:
        df = pd.concat(all_ctrl, ignore_index=True)
        _write(df, f"{args.out}_controls")
        print("\n=== FPR негативных контролов, median ===")
        print(df.groupby("control")[["fpr_invariant", "fpr_combined", "gate_rate"]]
              .median().to_string(float_format=lambda v: f"{v:8.3f}"))

    if all_curve:
        df = pd.concat(all_curve, ignore_index=True)
        _write(df, f"{args.out}_curve")

    if all_lofo:
        df = pd.concat(all_lofo, ignore_index=True)
        _write(df, f"{args.out}_lofo")
        print("\n=== LOFO: out-of-family recall @1% FPR, median ===")
        print(df.pivot_table(index="holdout_family", columns="feature_set",
                             values="recall_at_fpr1", aggfunc="median")
              .to_string(float_format=lambda v: f"{v:8.3f}"))

    if all_adv:
        df = pd.concat(all_adv, ignore_index=True)
        _write(df, f"{args.out}_adversary")
        print("\n=== адаптивный противник: recall_combined(budget), median ===")
        print(df.pivot_table(index=["family", "via_scope"], columns="budget",
                             values="recall_combined", aggfunc="median")
              .to_string(float_format=lambda v: f"{v:8.3f}"))

    print(f"\nготово. префикс файлов: {args.out}_*")


if __name__ == "__main__":
    main()
