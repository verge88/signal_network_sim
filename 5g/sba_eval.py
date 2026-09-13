"""
sba_eval.py
Полный протокол оценки:
  1) посадка детекторов на calib-части чистых окон (2-pass профиль);
  2) invariant_by_group  — recall по (family x via_scope), группировка режимов;
  3) negative_controls   — FPR на day/night, scale-in, api-change, upgrade;
  4) detectability_curve — recall(h) + аналитический прогноз (теорема);
  5) lofo_ablation       — leave-one-masking-family-out, STAT_only vs INV_ATTRIB;
  6) adaptive_adversary  — recall(budget) против white-box оптимизатора;
  7) sensitivity_analysis— дрейф параметров (lam, granularity, E[T], n_consumers);
  8) diagnostics         — sigma_ratio ~ 1, worst_benign_fpr ~ 0.01, theory error < 0.15.

Запуск:  python sba_eval.py --seed 0            (полный)
         python sba_eval.py --seed 0 --quick    (смоук)
"""

from __future__ import annotations

import argparse
import math
from dataclasses import asdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from sba_adversary import optimize_masking
from sba_invariant import (FittedDetectors, THEOREM, WindowObs, compute_invariants,
                           min_detectable_split_budget, norm_sf, sigma_boundary,
                           window_feature_row)
from sba_masking import FAMILIES, eta_of
from sba_sim_v1 import (Dataset, DatasetSpec, SbaConfig, Simulator, generate_dataset)
from sba_training_v1 import train_eval

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 60)


# --------------------------------------------------------------------------------------
# 1. Посадка детекторов
# --------------------------------------------------------------------------------------

def fit_detectors(ds: Dataset, alpha: float = 0.01) -> FittedDetectors:
    calib = ds.calib()
    if len(calib) < 30:
        raise ValueError(f"мало calib-окон: {len(calib)}")
    return FittedDetectors.fit(calib, alpha=alpha)


# --------------------------------------------------------------------------------------
# 2. Recall по группам
# --------------------------------------------------------------------------------------

def invariant_by_group(det: FittedDetectors, windows: Sequence[WindowObs]) -> pd.DataFrame:
    rows = []
    for w in windows:
        if w.label != 1:
            continue
        out = det.score(w)
        rows.append({
            "family": w.family, "via_scope": w.via_scope, "hidden_calls": w.hidden_calls,
            "budget": w.budget,
            "recall_invariant": float(out.fired_inv),
            "recall_attribution": float(out.fired_att and out.culprit_hat == w.culprit),
            "recall_combined": float(out.fired_combined),
            "gate": float(out.gate_conformance),
            "z1s": out.z1s, "atts": out.atts,
        })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    g = df.groupby(["family", "via_scope"], as_index=False).agg(
        n=("recall_invariant", "size"),
        recall_invariant=("recall_invariant", "mean"),
        recall_attribution=("recall_attribution", "mean"),
        recall_combined=("recall_combined", "mean"),
        gate_rate=("gate", "mean"),
        mean_z1s=("z1s", "mean"),
        mean_atts=("atts", "mean"))
    return g.sort_values(["via_scope", "family"]).reset_index(drop=True)


# --------------------------------------------------------------------------------------
# 3. Негативные контролы
# --------------------------------------------------------------------------------------

def negative_controls(det: FittedDetectors, ds: Dataset) -> pd.DataFrame:
    groups: Dict[str, List[WindowObs]] = {"clean_eval": ds.eval_benign()}
    for w in ds.eval_controls():
        groups.setdefault(w.control, []).append(w)
    rows = []
    for name, ws in groups.items():
        if not ws:
            continue
        outs = [det.score(w) for w in ws]
        rows.append({
            "control": name, "n": len(ws),
            "fpr_invariant": float(np.mean([o.fired_inv for o in outs])),
            "fpr_attribution": float(np.mean([o.fired_att for o in outs])),
            "fpr_combined": float(np.mean([o.fired_combined for o in outs])),
            "gate_rate": float(np.mean([o.gate_conformance for o in outs])),
            "mean_z1s": float(np.mean([o.z1s for o in outs])),
            "p99_z1s": float(np.quantile([o.z1s for o in outs], 0.99)),
        })
    return pd.DataFrame(rows).sort_values("control").reset_index(drop=True)


# --------------------------------------------------------------------------------------
# 4. Кривая обнаружимости + аналитический прогноз
# --------------------------------------------------------------------------------------

def theory_recall(h: int, family: str, via_scope: str, budget: float,
                  sigma: float, thr: float) -> float:
    """Прогноз по теореме. Для gate-каналов (Δ2/Δ3) обнаружение детерминировано."""
    if via_scope in ("no_token", "notify_abuse"):
        if family == "M5_token_backfill":
            residual = h * (1.0 - 0.9 * budget)     # ETA_M5
            return 1.0 if residual >= 1.0 else 0.0
        return 1.0
    residual = h * (1.0 - eta_of(family) * budget)
    return float(norm_sf(thr - residual / max(sigma, 1e-9)) +
                 norm_sf(thr + residual / max(sigma, 1e-9)))


def detectability_curve(det: FittedDetectors, windows: Sequence[WindowObs]) -> pd.DataFrame:
    rows = []
    for w in windows:
        if w.label != 1:
            continue
        out = det.score(w)
        sig = sigma_boundary(w.lam_hat, w.et_hat, w.export_granularity,
                             n_counters=2 * w.n_consumers)
        eff_thr = det.thr_z + max(det.thr_margin, 0.0)
        rows.append({
            "via_scope": w.via_scope, "family": w.family, "hidden_calls": w.hidden_calls,
            "budget": w.budget,
            "recall_combined": float(out.fired_combined),
            "recall_invariant": float(out.fired_inv),
            "recall_attribution": float(out.fired_att and out.culprit_hat == w.culprit),
            "theory": theory_recall(w.hidden_calls, w.family, w.via_scope, w.budget,
                                    sig, eff_thr),
            "h_min_theory": min_detectable_split_budget(sig, eff_thr, 1,
                                                        eta_of(w.family) * w.budget),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    g = df.groupby(["via_scope", "hidden_calls"], as_index=False).agg(
        n=("recall_combined", "size"),
        recall_invariant=("recall_invariant", "mean"),
        recall_attribution=("recall_attribution", "mean"),
        recall_combined=("recall_combined", "mean"),
        theory=("theory", "mean"),
        h_min_theory=("h_min_theory", "mean"))
    g["abs_theory_error"] = (g["recall_combined"] - g["theory"]).abs()
    return g.reset_index(drop=True)


# --------------------------------------------------------------------------------------
# 5. LOFO ablation (ML-контроль)
# --------------------------------------------------------------------------------------

def lofo_ablation(ds: Dataset, det: FittedDetectors,
                  feature_sets: Sequence[str] = ("STAT_only", "INV_ATTRIB", "ALL"),
                  seed: int = 0) -> pd.DataFrame:
    df = pd.DataFrame([window_feature_row(w, det) for w in ds.windows])
    benign = df[(df["label"] == 0)]
    attacks = df[df["label"] == 1]
    rows = []
    for fam in sorted(attacks["family"].unique()):
        tr = pd.concat([benign[benign["mode"] == "calib"],
                        attacks[attacks["family"] != fam]], ignore_index=True)
        te = pd.concat([benign[benign["mode"] == "eval"],
                        attacks[attacks["family"] == fam]], ignore_index=True)
        if tr["label"].nunique() < 2 or te["label"].sum() == 0:
            continue
        for fs in feature_sets:
            r = train_eval(tr, te, fs, holdout_family=fam, seed=seed)
            rows.append({"holdout_family": fam, "feature_set": fs,
                         "recall_at_fpr1": r.recall_at_fpr1, "auc": r.auc,
                         "n_train": r.n_train, "n_test_pos": r.n_test_pos})
        # аналитический детектор на том же held-out семействе (эталон)
        ws = [w for w in ds.attacks() if w.family == fam]
        outs = [det.score(w) for w in ws]
        rows.append({"holdout_family": fam, "feature_set": "ANALYTIC_INVARIANT",
                     "recall_at_fpr1": float(np.mean([o.fired_combined for o in outs])),
                     "auc": float("nan"), "n_train": 0, "n_test_pos": len(ws)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# 6. Адаптивный противник
# --------------------------------------------------------------------------------------

def adaptive_adversary(det: FittedDetectors, spec: DatasetSpec,
                       budgets: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
                       hidden: int = 40, n_bases: int = 12,
                       families: Optional[Sequence[str]] = None) -> pd.DataFrame:
    sim = Simulator(spec.cfg, seed=spec.seed + 991)
    bases = [sim.benign_window(i, mode="eval") for i in range(n_bases)]
    rows = []
    for fam in (families or FAMILIES):
        for via in spec.via_scopes:
            for b in budgets:
                r = optimize_masking(det, bases, hidden, via, fam, b,
                                     rng=np.random.default_rng(spec.seed + 7), iters=30)
                rows.append({"family": fam, "via_scope": via, "hidden_calls": hidden,
                             "budget": b, "recall_invariant": r.recall_inv,
                             "recall_attribution": r.recall_att,
                             "recall_combined": r.recall_combined,
                             "mean_margin": r.mean_margin})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# 7. Анализ чувствительности к дрейфу параметров
# --------------------------------------------------------------------------------------

DRIFTS: Dict[str, Dict[str, float]] = {
    "baseline": {},
    "lam_x2": {"lam_base": 2.0},
    "granularity_x4": {"export_granularity": 4.0},
    "et_x2": {"et_mean": 2.0},
    "consumers_x2": {"n_consumers": 2.0},
    "window_half": {"window_s": 0.5},
}


def sensitivity_analysis(base_spec: DatasetSpec, alpha: float = 0.01,
                         quick: bool = True) -> pd.DataFrame:
    rows = []
    for name, mult in DRIFTS.items():
        cfg = SbaConfig(**{**asdict(base_spec.cfg)})
        for k, m in mult.items():
            val = getattr(cfg, k)
            setattr(cfg, k, int(round(val * m)) if k == "n_consumers" else val * m)
        spec = DatasetSpec.quick(seed=base_spec.seed + 13) if quick else base_spec
        spec.cfg = cfg
        try:
            ds = generate_dataset(spec)
            det = fit_detectors(ds, alpha=alpha)
            atk = ds.attacks()
            outs = [det.score(w) for w in atk]
            ben = [det.score(w) for w in (ds.eval_benign() + ds.eval_controls())]
            rows.append({
                "drift": name,
                "sigma_ratio": det.profile.sigma_ratio,
                "thr_z": det.thr_z, "thr_att": det.thr_att, "thr_cusum": det.thr_cusum,
                "recall_invariant": float(np.mean([o.fired_inv for o in outs])),
                "recall_attribution": float(np.mean(
                    [o.fired_att and o.culprit_hat == w.culprit
                     for o, w in zip(outs, atk)])),
                "recall_combined": float(np.mean([o.fired_combined for o in outs])),
                "fpr_combined": float(np.mean([o.fired_combined for o in ben])),
                "n_attacks": len(atk), "n_benign": len(ben),
            })
        except Exception as exc:               # дрейф не должен ломать прогон
            rows.append({"drift": name, "error": repr(exc)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# 8. Диагностика
# --------------------------------------------------------------------------------------

def diagnostics(det: FittedDetectors, ds: Dataset, curve: pd.DataFrame,
                controls: pd.DataFrame, lofo: pd.DataFrame) -> Dict[str, float]:
    d: Dict[str, float] = {}
    d["mean_sigma"] = float(det.profile.sigma_ratio)
    d["n_calib"] = float(len(ds.calib()))
    d["thr_z"] = float(det.thr_z)
    d["thr_att"] = float(det.thr_att)
    d["thr_cusum"] = float(det.thr_cusum)
    d["thr_margin"] = float(det.thr_margin)

    calib_outs = [det.score(w) for w in ds.calib()]
    d["calib_fpr_combined"] = float(np.mean([o.fired_combined for o in calib_outs]))
    d["calib_gate_rate"] = float(np.mean([o.gate_conformance for o in calib_outs]))
    if not controls.empty:
        d["worst_benign_fpr_combined"] = float(controls["fpr_combined"].max())
        d["worst_benign_gate_rate"] = float(controls["gate_rate"].max())
    if not curve.empty:
        d["mean_abs_theory_error"] = float(curve["abs_theory_error"].mean())
    if not lofo.empty:
        piv = lofo.pivot_table(index="holdout_family", columns="feature_set",
                               values="recall_at_fpr1", aggfunc="mean")
        for fs in ("STAT_only", "INV_ATTRIB", "ANALYTIC_INVARIANT"):
            if fs in piv.columns:
                d[f"mean_out_of_family_{fs}"] = float(piv[fs].mean())
        if "STAT_only" in piv.columns and "INV_ATTRIB" in piv.columns:
            d["out_of_family_gap_INV_minus_STAT"] = float(
                (piv["INV_ATTRIB"] - piv["STAT_only"]).mean())
    return d


# --------------------------------------------------------------------------------------
# Точка входа
# --------------------------------------------------------------------------------------

def run_one(seed: int = 0, quick: bool = False, alpha: float = 0.01,
            with_adversary: bool = True, with_sensitivity: bool = True) -> Dict[str, object]:
    spec = DatasetSpec.quick(seed=seed) if quick else DatasetSpec(seed=seed)
    ds = generate_dataset(spec)
    det = fit_detectors(ds, alpha=alpha)
    ds.df = pd.DataFrame([window_feature_row(w, det) for w in ds.windows])

    by_group = invariant_by_group(det, ds.attacks())
    controls = negative_controls(det, ds)
    curve = detectability_curve(det, ds.attacks())
    lofo = lofo_ablation(ds, det, seed=seed)
    adv = adaptive_adversary(det, spec, hidden=40,
                             families=["M1_additive", "M4_split", "M5_token_backfill"]
                             if quick else None) if with_adversary else pd.DataFrame()
    sens = sensitivity_analysis(spec, alpha=alpha, quick=True) if with_sensitivity \
        else pd.DataFrame()
    diag = diagnostics(det, ds, curve, controls, lofo)
    return {"spec": spec, "dataset": ds, "detectors": det, "by_group": by_group,
            "controls": controls, "curve": curve, "lofo": lofo, "adversary": adv,
            "sensitivity": sens, "diagnostics": diag}


def main() -> None:
    ap = argparse.ArgumentParser(description="5G SBA invariant evaluation")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--no-adversary", action="store_true")
    ap.add_argument("--no-sensitivity", action="store_true")
    ap.add_argument("--theorem", action="store_true", help="напечатать формулировку теоремы")
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    if args.theorem:
        print(THEOREM)

    res = run_one(seed=args.seed, quick=args.quick, alpha=args.alpha,
                  with_adversary=not args.no_adversary,
                  with_sensitivity=not args.no_sensitivity)
    ds: Dataset = res["dataset"]

    print(f"\n=== dataset === shape={ds.df.shape}  calib={len(ds.calib())} "
          f"benign_eval={len(ds.eval_benign())} controls={len(ds.eval_controls())} "
          f"attacks={len(ds.attacks())}")

    print("\n=== invariant_by_group (family x via_scope) ===")
    print(res["by_group"].to_string(index=False, float_format=lambda v: f"{v:8.3f}"))

    print("\n=== negative_controls (FPR) ===")
    print(res["controls"].to_string(index=False, float_format=lambda v: f"{v:8.3f}"))

    print("\n=== detectability_curve (recall vs hidden_calls) ===")
    print(res["curve"].to_string(index=False, float_format=lambda v: f"{v:8.3f}"))

    print("\n=== LOFO ablation (recall @ FPR=1%, held-out masking family) ===")
    lofo: pd.DataFrame = res["lofo"]
    if not lofo.empty:
        print(lofo.pivot_table(index="holdout_family", columns="feature_set",
                               values="recall_at_fpr1", aggfunc="mean")
              .to_string(float_format=lambda v: f"{v:8.3f}"))

    if isinstance(res["adversary"], pd.DataFrame) and not res["adversary"].empty:
        print("\n=== adaptive white-box adversary (recall vs budget) ===")
        print(res["adversary"].pivot_table(index=["family", "via_scope"], columns="budget",
                                           values="recall_combined", aggfunc="mean")
              .to_string(float_format=lambda v: f"{v:8.3f}"))

    if isinstance(res["sensitivity"], pd.DataFrame) and not res["sensitivity"].empty:
        print("\n=== sensitivity to parameter drift ===")
        print(res["sensitivity"].to_string(index=False, float_format=lambda v: f"{v:8.3f}"))

    print("\n=== diagnostics ===")
    for k, v in res["diagnostics"].items():
        print(f"  {k:38s} {v:10.4f}")
    print("\nожидаемые ориентиры: mean_sigma ~ 1.0 | calib_gate_rate = 0.0 | "
          "worst_benign_fpr_combined ~ 0.01 | mean_abs_theory_error < 0.15")

    if args.out:
        ds.df.to_csv(f"{args.out}_windows.csv", index=False)
        res["by_group"].to_csv(f"{args.out}_by_group.csv", index=False)
        res["curve"].to_csv(f"{args.out}_curve.csv", index=False)
        print(f"\nсохранено: {args.out}_*.csv")


if __name__ == "__main__":
    main()
