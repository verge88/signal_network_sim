"""
sba_adversary.py
Адаптивный white-box противник: покоординатный спуск + (1+1)-ES по параметрам
маскирования; цель — минимизировать маржу детектора при фиксированном бюджете.
Используется для построения кривых recall(budget), а не для обучения детектора.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from sba_invariant import FittedDetectors, HiddenPlan, WindowObs
from sba_masking import PARAM_DIM, apply_masking


@dataclass
class AdversaryResult:
    family: str
    via_scope: str
    hidden_calls: int
    budget: float
    best_params: np.ndarray
    mean_margin: float
    recall_inv: float
    recall_att: float
    recall_combined: float
    n_eval: int


def _objective(det: FittedDetectors, bases: Sequence[WindowObs], plans: Sequence[HiddenPlan],
               family: str, params: np.ndarray, budget: float,
               rng: np.random.Generator) -> Tuple[float, float, float, float]:
    margins, inv, att, comb = [], 0, 0, 0
    for base, plan in zip(bases, plans):
        p = HiddenPlan(hidden_calls=plan.hidden_calls, culprit_idx=plan.culprit_idx,
                       via_scope=plan.via_scope)
        w = apply_masking(base, p, family, params, budget, rng)
        out = det.score(w)
        margins.append(min(out.margin, 50.0))
        inv += int(out.fired_inv)
        att += int(out.fired_att and out.culprit_hat == w.culprit)
        comb += int(out.fired_combined)
    n = max(len(bases), 1)
    return float(np.mean(margins)), inv / n, att / n, comb / n


def optimize_masking(det: FittedDetectors, bases: Sequence[WindowObs], hidden: int,
                     via_scope: str, family: str, budget: float,
                     rng: Optional[np.random.Generator] = None,
                     iters: int = 40) -> AdversaryResult:
    rng = rng or np.random.default_rng(0)
    d = PARAM_DIM.get(family, 2)
    plans = [HiddenPlan(hidden_calls=int(hidden),
                        culprit_idx=int(rng.integers(0, b.n_consumers)),
                        via_scope=via_scope) for b in bases]

    # --- фаза 1: покоординатная сетка
    best = np.full(d, 0.5)
    best_val = _objective(det, bases, plans, family, best, budget, rng)[0]
    for j in range(d):
        for v in np.linspace(0.0, 1.0, 9):
            cand = best.copy()
            cand[j] = v
            val = _objective(det, bases, plans, family, cand, budget, rng)[0]
            if val < best_val:
                best_val, best = val, cand

    # --- фаза 2: (1+1)-ES с адаптацией шага
    sigma = 0.25
    for _ in range(iters):
        cand = np.clip(best + rng.normal(0.0, sigma, size=d), 0.0, 1.0)
        val = _objective(det, bases, plans, family, cand, budget, rng)[0]
        if val < best_val:
            best_val, best = val, cand
            sigma = min(sigma * 1.3, 0.5)
        else:
            sigma = max(sigma * 0.85, 0.02)

    m, r_inv, r_att, r_comb = _objective(det, bases, plans, family, best, budget, rng)
    return AdversaryResult(family=family, via_scope=via_scope, hidden_calls=int(hidden),
                           budget=float(budget), best_params=best, mean_margin=m,
                           recall_inv=r_inv, recall_att=r_att, recall_combined=r_comb,
                           n_eval=len(bases))
