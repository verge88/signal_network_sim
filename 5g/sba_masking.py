"""
sba_masking.py
Пять семейств маскирования (для протокола leave-one-masking-family-out).

Каждое семейство получает ЧИСТОЕ окно (benign WindowObs) + HiddenPlan и возвращает
репортируемое окно. Параметры params in [0,1]^d, budget in [0,1] — масштаб манипуляции.
Константы eta_* задают предел переноса объёма между наблюдаемыми каналами
(они же входят в нижнюю границу теоремы).
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from sba_invariant import HiddenPlan, WindowObs

FAMILIES = ["M1_additive", "M2_reparam", "M3_temporal", "M4_split", "M5_token_backfill"]
PARAM_DIM: Dict[str, int] = {f: 2 for f in FAMILIES}

ETA_M1 = 0.85       # макс. доля h, снимаемая аддитивной подгонкой счётчиков
LEAK_M1 = 0.15      # доля снятого объёма, всплывающая в непарных ответах
ETA_M5 = 0.90       # макс. доля h, покрываемая бэкфилл-токенами с валидным nbf
MAX_REPARAM_LAM = 0.35
MAX_REPARAM_ET = 0.50


# --------------------------------------------------------------------------------------
# Инъекция скрытых вызовов в наблюдаемые каналы (до маскирования)
# --------------------------------------------------------------------------------------

def _alloc_default(plan: HiddenPlan, n_consumers: int) -> np.ndarray:
    a = np.zeros(n_consumers, dtype=float)
    idx = plan.culprit_idx if 0 <= plan.culprit_idx < n_consumers else 0
    a[idx] = float(plan.hidden_calls)
    return a


def _slot_alloc_default(plan: HiddenPlan, n_slots: int, rng: np.random.Generator) -> np.ndarray:
    w = rng.dirichlet(np.full(n_slots, 3.0))
    return w * float(plan.hidden_calls)


def inject_hidden(w: WindowObs, plan: HiddenPlan, alloc: np.ndarray,
                  slot_alloc: np.ndarray) -> WindowObs:
    """
    Размещение h вызовов в каналах телеметрии согласно via_scope.
      no_token     : вызов идёт через SCP, токена нет      -> Δ2 (ворота), Δ1 ~ 0
      scp_bypass   : вызов минует SCP (нет streams)        -> Δ1 = +h, ворота молчат
      notify_abuse : нотификация без подписки              -> Δ3 (ворота), Δ1 = -h
    """
    v = w.copy()
    h = float(plan.hidden_calls)
    if h <= 0:
        return v

    if plan.via_scope == "no_token":
        v.served_total += h
        v.streams_total += h
        v.cons_served = v.cons_served + alloc
        v.cons_streams = v.cons_streams + alloc
        v.slot_served = v.slot_served + slot_alloc
        v.slot_streams = v.slot_streams + slot_alloc
        v.out_of_scope += h
    elif plan.via_scope == "scp_bypass":
        v.served_total += h
        v.cons_served = v.cons_served + alloc
        v.slot_served = v.slot_served + slot_alloc
    elif plan.via_scope == "notify_abuse":
        v.streams_total += h
        v.cons_streams = v.cons_streams + alloc
        v.slot_streams = v.slot_streams + slot_alloc
        v.notif_delivered += h
        v.orphan_notifications += h
    else:
        raise ValueError(f"unknown via_scope: {plan.via_scope}")

    v.label = 1
    v.hidden_calls = int(plan.hidden_calls)
    v.via_scope = plan.via_scope
    if 0 <= plan.culprit_idx < len(v.consumer_ids):
        v.culprit = v.consumer_ids[plan.culprit_idx]
    return v


# --------------------------------------------------------------------------------------
# Семейства
# --------------------------------------------------------------------------------------

def _deflate(v: WindowObs, amount: float, on_served: float, alloc: np.ndarray,
             slot_alloc: np.ndarray) -> None:
    """Аддитивное снятие объёма со счётчиков ledger/streams."""
    h = max(float(alloc.sum()), 1e-9)
    a_served = amount * on_served
    a_streams = amount * (1.0 - on_served)
    v.served_total -= a_served
    v.streams_total -= a_streams
    v.cons_served = v.cons_served - alloc * (a_served / h)
    v.cons_streams = v.cons_streams - alloc * (a_streams / h)
    sh = max(float(slot_alloc.sum()), 1e-9)
    v.slot_served = v.slot_served - slot_alloc * (a_served / sh)
    v.slot_streams = v.slot_streams - slot_alloc * (a_streams / sh)


def apply_masking(base: WindowObs, plan: HiddenPlan, family: str, params: np.ndarray,
                  budget: float, rng: np.random.Generator) -> WindowObs:
    """Главная точка входа: чистое окно + план -> репортируемое окно."""
    params = np.clip(np.asarray(params, dtype=float).ravel(), 0.0, 1.0)
    if params.size < PARAM_DIM.get(family, 2):
        params = np.pad(params, (0, PARAM_DIM.get(family, 2) - params.size))
    budget = float(np.clip(budget, 0.0, 1.0))
    n_c = int(base.n_consumers)
    n_s = int(base.n_slots)
    h = float(plan.hidden_calls)

    # --- распределение по консьюмерам (M4 — единственное семейство-сплиттер)
    if family == "M4_split":
        k = 1 + int(round(params[0] * (n_c - 1)))
        idx = [plan.culprit_idx] + [i for i in range(n_c) if i != plan.culprit_idx][:k - 1]
        alloc = np.zeros(n_c, dtype=float)
        weights = np.linspace(1.0, 0.5, num=len(idx))
        weights = weights / weights.sum()
        for i, wi in zip(idx, weights):
            alloc[i] = h * wi
        plan.split_k = k
    else:
        alloc = _alloc_default(plan, n_c)
        plan.split_k = 1

    # --- распределение по слотам (M3 — концентрация/размазывание)
    if family == "M3_temporal":
        conc = 0.5 + 8.0 * params[0] * budget
        wts = rng.dirichlet(np.full(n_s, conc))
        slot_alloc = wts * h
    else:
        slot_alloc = _slot_alloc_default(plan, n_s, rng)

    plan.alloc = alloc
    plan.slot_alloc = slot_alloc

    v = inject_hidden(base, plan, alloc, slot_alloc)
    v.family = family
    v.budget = budget

    if family == "M1_additive":
        amount = ETA_M1 * budget * h
        _deflate(v, amount, on_served=params[0], alloc=alloc, slot_alloc=slot_alloc)
        v.unmatched_responses += LEAK_M1 * amount

    elif family == "M2_reparam":
        # раздувание репортируемых lam/E[T] -> рост sigma1, но рост Δ5
        v.lam_hat *= (1.0 + MAX_REPARAM_LAM * budget * params[0])
        v.et_hat *= (1.0 + MAX_REPARAM_ET * budget * params[1])
        amount = 0.5 * ETA_M1 * budget * h
        _deflate(v, amount, on_served=params[0], alloc=alloc, slot_alloc=slot_alloc)

    elif family == "M3_temporal":
        # тотальные суммы не меняются, подавляется только CUSUM
        pass

    elif family == "M4_split":
        amount = 0.3 * ETA_M1 * budget * h
        _deflate(v, amount, on_served=0.5, alloc=alloc, slot_alloc=slot_alloc)

    elif family == "M5_token_backfill":
        covered = np.floor(v.out_of_scope * ETA_M5 * budget * params[0])
        v.out_of_scope -= covered
        # бэкфилл оставляет токены с nbf > времени первого использования
        v.late_token += np.floor(covered * (1.0 - ETA_M5 * budget * params[1]))
    else:
        raise ValueError(f"unknown masking family: {family}")

    v.served_total = max(v.served_total, 0.0)
    v.streams_total = max(v.streams_total, 0.0)
    v.out_of_scope = max(v.out_of_scope, 0.0)
    v.late_token = max(v.late_token, 0.0)
    v.slot_served = np.maximum(v.slot_served, 0.0)
    v.slot_streams = np.maximum(v.slot_streams, 0.0)
    return v


def eta_of(family: str) -> float:
    return {"M1_additive": ETA_M1, "M2_reparam": 0.5 * ETA_M1, "M3_temporal": 0.0,
            "M4_split": 0.3 * ETA_M1, "M5_token_backfill": 0.0}.get(family, 0.0)
