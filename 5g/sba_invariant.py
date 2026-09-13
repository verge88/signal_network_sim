"""
sba_invariant.py
Аналитические (необучаемые) инварианты 5G SBA + детекторы.

Нормативная база наблюдаемых величин:
  - 3GPP TS 23.501/23.502  : SBA, SCP/NRF, модель сервисных вызовов
  - 3GPP TS 29.500         : SBI, 3gpp-Sbi-Sender-Timestamp, 3gpp-Sbi-Correlation-Info
  - 3GPP TS 29.510         : Nnrf_AccessToken (OAuth2 scope/nbf/exp)
  - 3GPP TS 33.501         : авторизация SBI, PRINS/N32
  - RFC 9113               : HTTP/2 streams (транспортный счётчик)

ТЕОРЕМА (дилемма обнаружения–атрибуции) — формулировка ниже в THEOREM.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

THEOREM = """
Пусть в окне W длительности T скрытно выполнено h сервисных вызовов.
Наблюдаемые агрегаты: S (producer ledger, выполненные вызовы), X (HTTP/2 streams на SCP),
K (вызовы, покрытые валидным токеном NRF со совпадающим scope), Lt (вызовы, покрытые
токеном, выданным позже первого использования), O (нотификации без подписки).

Определим
  D1 = S - X                      (транспортная невязка, статистическая)
  D2 = (число вызовов вне scope) + Lt   (неконформность, ДЕТЕРМИНИРОВАННАЯ: в чистом
                                         окне D2 = 0 по построению выдачи токенов)
  D3 = O                          (неконформность подписок/нотификаций, D3 = 0 в чистом окне)
  D5 = lam_hat * T - X            (согласованность репортируемой нагрузки и транспорта)

Утверждение. Любая стратегия маскирования, сохраняющая реальное выполнение h вызовов,
удовлетворяет
      |D1| + D2 + D3 + Att  >=  h * (1 - eta),
где eta < 1 — максимальная доля h, которую механизм маскирования способен перенести из
одного наблюдаемого канала в другой (для аддитивного переноса eta = ETA_M1 = 0.85,
для бэкфилла токенов eta = 0.9), а Att — атрибуционный избыток
      Att = max_c |S_c - X_c|.
Следствие 1 (нижняя граница обнаружения). При равномерном "размазывании" остатка
r = h(1-eta) по k сущностям порог z_thr достигается только если
      r >= z_thr * sigma * sqrt(k),
что даёт min_detectable_split_budget().
Следствие 2 (дилемма). Уменьшение Att (размазывание по k) не уменьшает |D1|+D2+D3;
уменьшение |D1| (аддитивная подгонка счётчиков) увеличивает D5 либо D2. Обнаружение и
атрибуция не могут быть подавлены одновременно в пределах одного бюджета.
"""

# --------------------------------------------------------------------------------------
# Вспомогательная статистика (без scipy)
# --------------------------------------------------------------------------------------

_A = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
      1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
_B = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
      6.680131188771972e+01, -1.328068155288572e+01]
_C = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
      -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
_D = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
      3.754408661907416e+00]


def norm_ppf(p: float) -> float:
    """Обратная функция стандартного нормального распределения (Acklam)."""
    p = min(max(float(p), 1e-16), 1.0 - 1e-16)
    if p < 0.02425:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]) / \
               ((((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1.0)
    if p > 1.0 - 0.02425:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]) / \
               ((((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1.0)
    q = p - 0.5
    r = q * q
    return (((((_A[0] * r + _A[1]) * r + _A[2]) * r + _A[3]) * r + _A[4]) * r + _A[5]) * q / \
           (((((_B[0] * r + _B[1]) * r + _B[2]) * r + _B[3]) * r + _B[4]) * r + 1.0)


def norm_sf(z: float) -> float:
    return 0.5 * math.erfc(float(z) / math.sqrt(2.0))


def robust_scale(x: np.ndarray, floor: float = 0.25) -> float:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return 1.0
    mad = float(np.median(np.abs(x - np.median(x))))
    return max(1.4826 * mad, floor)


# --------------------------------------------------------------------------------------
# ПАТЧ 3: шумовая модель. eps > 0 строго; закон Литтла + гранулярность экспорта.
# --------------------------------------------------------------------------------------

SIGMA_FLOOR = 0.5


def sigma_boundary(lam: float, et: float, granularity: float, n_counters: int = 2) -> float:
    """
    sigma невязки двух счётчиков, считающих одни и те же события.

    Источники дисперсии:
      * незавершённые транзакции на границах окна: L = lam * E[T] (закон Литтла),
        две границы, Пуассон  -> var = 2L
      * укрупнение экспорта телеметрии до кратного g. При g = 1 счётчики уже
        целочисленные, поэтому дополнительного квантования нет; при g > 1
        используется консервативная эмпирическая поправка на бакетизацию.
    """
    lam = max(float(lam), 0.0)
    et = max(float(et), 0.0)
    g = max(float(granularity), 0.0)
    quant_var = max(int(n_counters), 1) * max(g - 1.0, 0.0) / 8.0
    var = 2.0 * lam * et + quant_var
    return max(math.sqrt(max(var, 0.0)), SIGMA_FLOOR)


def min_detectable_split_budget(sigma: float, z_thr: float, k_split: int = 1,
                                eta: float = 0.0) -> float:
    """
    Аналитический минимум скрытых вызовов h, при котором остаток h*(1-eta),
    размазанный по k_split сущностям, всё ещё превышает порог z_thr.
    """
    k_split = max(int(k_split), 1)
    denom = max(1.0 - float(eta), 1e-6)
    return float(z_thr) * float(sigma) * math.sqrt(k_split) / denom


# --------------------------------------------------------------------------------------
# Модель данных окна
# --------------------------------------------------------------------------------------

@dataclass
class HiddenPlan:
    """Истинный (ground-truth) план скрытых вызовов."""
    hidden_calls: int = 0
    culprit_idx: int = -1
    via_scope: str = ""          # no_token | scp_bypass | notify_abuse
    split_k: int = 1
    alloc: Optional[np.ndarray] = None       # распределение h по консьюмерам
    slot_alloc: Optional[np.ndarray] = None  # распределение h по слотам


@dataclass
class WindowObs:
    """Репортируемая телеметрия одного окна + ground truth."""
    # --- идентификация / истина
    seed: int = 0
    window_id: int = 0
    mode: str = "eval"           # calib | eval
    label: int = 0               # 1 = атака
    family: str = "none"         # семейство маскирования
    via_scope: str = ""
    control: str = ""            # negative control
    hidden_calls: int = 0
    culprit: str = ""
    budget: float = 0.0

    # --- операционный контекст
    window_s: float = 600.0
    n_slots: int = 10
    export_granularity: float = 1.0
    lam_hat: float = 0.0
    et_hat: float = 0.05
    n_consumers: int = 12
    consumer_ids: List[str] = field(default_factory=list)
    cons_lam: np.ndarray = field(default_factory=lambda: np.zeros(0))

    # --- репортируемые агрегаты
    served_total: float = 0.0
    streams_total: float = 0.0
    out_of_scope: float = 0.0
    late_token: float = 0.0
    orphan_notifications: float = 0.0
    unmatched_responses: float = 0.0
    unmatched_expected: float = 0.0
    notif_delivered: float = 0.0
    subs_active: float = 0.0

    slot_served: np.ndarray = field(default_factory=lambda: np.zeros(0))
    slot_streams: np.ndarray = field(default_factory=lambda: np.zeros(0))
    cons_served: np.ndarray = field(default_factory=lambda: np.zeros(0))
    cons_streams: np.ndarray = field(default_factory=lambda: np.zeros(0))
    cons_tokens: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def copy(self) -> "WindowObs":
        return WindowObs(
            seed=self.seed, window_id=self.window_id, mode=self.mode, label=self.label,
            family=self.family, via_scope=self.via_scope, control=self.control,
            hidden_calls=self.hidden_calls, culprit=self.culprit, budget=self.budget,
            window_s=self.window_s, n_slots=self.n_slots,
            export_granularity=self.export_granularity, lam_hat=self.lam_hat,
            et_hat=self.et_hat, n_consumers=self.n_consumers,
            consumer_ids=list(self.consumer_ids), cons_lam=np.array(self.cons_lam, dtype=float),
            served_total=self.served_total, streams_total=self.streams_total,
            out_of_scope=self.out_of_scope, late_token=self.late_token,
            orphan_notifications=self.orphan_notifications,
            unmatched_responses=self.unmatched_responses,
            unmatched_expected=self.unmatched_expected,
            notif_delivered=self.notif_delivered, subs_active=self.subs_active,
            slot_served=np.array(self.slot_served, dtype=float),
            slot_streams=np.array(self.slot_streams, dtype=float),
            cons_served=np.array(self.cons_served, dtype=float),
            cons_streams=np.array(self.cons_streams, dtype=float),
            cons_tokens=np.array(self.cons_tokens, dtype=float),
        )


# --------------------------------------------------------------------------------------
# ПАТЧ B: расчёт инвариантов; delta1 (транспорт) отделён от delta2 (конформность)
# --------------------------------------------------------------------------------------

CUSUM_DRIFT = 0.5
ATTRIBUTION_TAIL_MULT = 1.30


@dataclass
class Invariants:
    delta1: float = 0.0
    delta2: float = 0.0
    delta3: float = 0.0
    delta4: float = 0.0
    delta5: float = 0.0
    sigma1: float = 1.0
    z1: float = 0.0
    z4: float = 0.0
    z5: float = 0.0
    cusum: float = 0.0
    att_z: np.ndarray = field(default_factory=lambda: np.zeros(0))
    att_zmax: float = 0.0
    att_z2: float = 0.0
    att_arg: int = -1
    gate_conformance: bool = False


def compute_invariants(w: WindowObs) -> Invariants:
    g = w.export_granularity
    n_c = max(int(w.n_consumers), 1)

    # --- Δ1: транспортная невязка (статистический канал)
    delta1 = float(w.served_total - w.streams_total)
    sigma1 = sigma_boundary(w.lam_hat, w.et_hat, g, n_counters=2 * n_c)
    z1 = delta1 / sigma1

    # --- Δ2/Δ3: конформность (детерминированные ворота, в чистом окне строго 0)
    delta2 = float(w.out_of_scope + w.late_token)
    delta3 = float(w.orphan_notifications)
    gate = bool(delta2 > 1e-9 or delta3 > 1e-9)

    # --- Δ4: непарные ответы (статистический канал, в норме ненулевой из-за таймаутов)
    mu4 = max(float(w.unmatched_expected), 1.0)
    delta4 = float(w.unmatched_responses - mu4)
    z4 = delta4 / math.sqrt(mu4)

    # --- Δ5: согласованность репортируемой нагрузки и транспорта (ловит репараметризацию)
    delta5 = float(w.lam_hat * w.window_s - w.streams_total)
    sigma5 = sigma_boundary(w.lam_hat, w.et_hat, g, n_counters=2 * n_c)
    sigma5 = math.sqrt(sigma5 ** 2 + max(w.lam_hat * w.window_s, 1.0))
    z5 = delta5 / sigma5

    # --- Атрибуция: покомпонентная невязка по консьюмерам
    cs = np.asarray(w.cons_served, dtype=float)
    cx = np.asarray(w.cons_streams, dtype=float)
    lam_c = np.asarray(w.cons_lam, dtype=float)
    if cs.size == 0:
        att_z = np.zeros(0)
    else:
        sig_c = np.array([sigma_boundary(l, w.et_hat, g, n_counters=2) for l in lam_c])
        att_z = (cs - cx) / sig_c
    if att_z.size:
        order = np.argsort(-np.abs(att_z))
        att_arg = int(order[0])
        att_zmax = float(abs(att_z[att_arg]))
        att_z2 = float(abs(att_z[order[1]])) if att_z.size > 1 else 0.0
    else:
        att_arg, att_zmax, att_z2 = -1, 0.0, 0.0

    # --- ПАТЧ 4: CUSUM строго внутри окна (state сбрасывается, межоконной утечки нет)
    ss = np.asarray(w.slot_served, dtype=float)
    sx = np.asarray(w.slot_streams, dtype=float)
    cusum = 0.0
    if ss.size and ss.size == sx.size:
        lam_slot = max(w.lam_hat, 0.0)
        sig_slot = sigma_boundary(lam_slot, w.et_hat, g, n_counters=2)
        r = (ss - sx) / sig_slot
        s_pos = 0.0
        s_neg = 0.0
        for rk in r:
            s_pos = max(0.0, s_pos + rk - CUSUM_DRIFT)
            s_neg = max(0.0, s_neg - rk - CUSUM_DRIFT)
            cusum = max(cusum, s_pos, s_neg)

    return Invariants(delta1=delta1, delta2=delta2, delta3=delta3, delta4=delta4,
                      delta5=delta5, sigma1=sigma1, z1=z1, z4=z4, z5=z5, cusum=cusum,
                      att_z=att_z, att_zmax=att_zmax, att_z2=att_z2, att_arg=att_arg,
                      gate_conformance=gate)


# --------------------------------------------------------------------------------------
# ПАТЧ C: базовый операционный профиль — робастная стандартизация статистик
# --------------------------------------------------------------------------------------

STAT_KEYS = ("z1", "z4", "z5", "cusum", "att_zmax")


@dataclass
class OpProfileBaseline:
    loc: Dict[str, float] = field(default_factory=dict)
    scale: Dict[str, float] = field(default_factory=dict)
    emp_q: Dict[str, float] = field(default_factory=dict)
    sigma_ratio: float = 1.0
    n_fit: int = 0

    @staticmethod
    def fit(windows: Sequence[WindowObs], alpha: float = 0.01,
            n_tests: int = 16) -> "OpProfileBaseline":
        """2-проходная посадка на ЧИСТЫХ (calib) окнах."""
        clean = [w for w in windows if w.label == 0]
        if not clean:
            raise ValueError("OpProfileBaseline.fit: нет чистых окон")
        inv = [compute_invariants(w) for w in clean]
        prof = OpProfileBaseline(n_fit=len(clean))

        raw = {k: np.array([getattr(i, k) for i in inv], dtype=float) for k in STAT_KEYS}

        # проход 1: грубая локация/масштаб
        for k, v in raw.items():
            prof.loc[k] = float(np.median(v))
            prof.scale[k] = robust_scale(v)

        # проход 2: отбрасываем 2% наиболее выбивающихся и пересчитываем
        for k, v in raw.items():
            z = np.abs((v - prof.loc[k]) / prof.scale[k])
            keep = v[z <= np.quantile(z, 0.98)] if v.size > 20 else v
            prof.loc[k] = float(np.median(keep))
            prof.scale[k] = robust_scale(keep)

        # эмпирические квантили для не-нормальных статистик (CUSUM, max-тип)
        q = 1.0 - alpha / max(int(n_tests), 1)
        for k in ("cusum", "att_zmax"):
            std = (raw[k] - prof.loc[k]) / prof.scale[k]
            prof.emp_q[k] = float(np.quantile(std, min(q, 0.999)))

        # диагностика: sigma_ratio = std(z1) / 1.0, должно быть ~1
        prof.sigma_ratio = float(np.std(raw["z1"]))
        return prof

    def std(self, key: str, value: float) -> float:
        return (float(value) - self.loc.get(key, 0.0)) / self.scale.get(key, 1.0)


@dataclass
class DetectorOut:
    z1s: float = 0.0
    z4s: float = 0.0
    z5s: float = 0.0
    cusums: float = 0.0
    atts: float = 0.0
    gate_conformance: bool = False
    fired_inv: bool = False
    fired_att: bool = False
    fired_combined: bool = False
    margin: float = 0.0          # max(stat - thr); >0 => срабатывание
    culprit_hat: str = ""
    inv: Optional[Invariants] = None


@dataclass
class FittedDetectors:
    profile: OpProfileBaseline
    alpha: float = 0.01
    n_tests: int = 16
    thr_z: float = 3.2
    thr_att: float = 3.2
    thr_cusum: float = 5.0
    thr_margin: float = 0.0

    @staticmethod
    def fit(calib_windows: Sequence[WindowObs], alpha: float = 0.01) -> "FittedDetectors":
        n_c = int(calib_windows[0].n_consumers) if calib_windows else 12
        n_tests = 4 + 1 + max(n_c, 1)          # Δ1,Δ4,Δ5,Att + CUSUM + атрибуция
        prof = OpProfileBaseline.fit(calib_windows, alpha=alpha, n_tests=n_tests)
        thr_z = norm_ppf(1.0 - alpha / n_tests)
        thr_att = max(norm_ppf(1.0 - alpha / n_tests), prof.emp_q.get("att_zmax", 0.0))
        thr_cusum = max(prof.emp_q.get("cusum", 0.0), 2.0)

        margins = []
        for w in calib_windows:
            inv = compute_invariants(w)
            z1s = abs(prof.std("z1", inv.z1))
            z4s = abs(prof.std("z4", inv.z4))
            z5s = abs(prof.std("z5", inv.z5))
            cus = prof.std("cusum", inv.cusum)
            atts = prof.std("att_zmax", inv.att_zmax)
            inv_margin = max(z1s - thr_z, z4s - thr_z, z5s - thr_z,
                             cus - thr_cusum)
            att_margin = atts - thr_att
            margins.append(max(inv_margin, att_margin))
        q = min(max(1.0 - alpha / 2.0, 0.0), 1.0)
        thr_margin = max(0.0, float(np.quantile(margins, q))) if margins else 0.0
        thr_att *= ATTRIBUTION_TAIL_MULT
        thr_cusum += thr_margin

        return FittedDetectors(profile=prof, alpha=alpha, n_tests=n_tests,
                               thr_z=thr_z, thr_att=thr_att,
                               thr_cusum=thr_cusum, thr_margin=thr_margin)

    def score(self, w: WindowObs) -> DetectorOut:
        inv = compute_invariants(w)
        p = self.profile
        z1s = abs(p.std("z1", inv.z1))
        z4s = abs(p.std("z4", inv.z4))
        z5s = abs(p.std("z5", inv.z5))
        cus = p.std("cusum", inv.cusum)
        atts = p.std("att_zmax", inv.att_zmax)

        # ПАТЧ B: gate_conformance — детерминированный флаг, не статистика
        gate = inv.gate_conformance
        inv_margin = max(z1s - self.thr_z, z4s - self.thr_z,
                         z5s - self.thr_z, cus - self.thr_cusum)
        att_margin = atts - self.thr_att
        fired_stat = inv_margin > self.thr_margin
        fired_inv = bool(gate or fired_stat)
        fired_att = bool(att_margin > self.thr_margin)

        margin = float(max(inv_margin, att_margin) - self.thr_margin)
        if gate:
            margin = max(margin, 1e3)         # ворота доминируют

        culprit_hat = ""
        if inv.att_arg >= 0 and inv.att_arg < len(w.consumer_ids):
            culprit_hat = w.consumer_ids[inv.att_arg]

        return DetectorOut(z1s=z1s, z4s=z4s, z5s=z5s, cusums=cus, atts=atts,
                           gate_conformance=gate, fired_inv=fired_inv, fired_att=fired_att,
                           fired_combined=bool(fired_inv or fired_att), margin=margin,
                           culprit_hat=culprit_hat, inv=inv)


# --------------------------------------------------------------------------------------
# Признаки для ML-контрольной группы
# --------------------------------------------------------------------------------------

def window_feature_row(w: WindowObs, det: Optional[FittedDetectors] = None) -> Dict[str, float]:
    out = det.score(w) if det is not None else None
    inv = out.inv if out is not None else compute_invariants(w)

    ss = np.asarray(w.slot_served, dtype=float)
    sx = np.asarray(w.slot_streams, dtype=float)
    cs = np.asarray(w.cons_served, dtype=float)

    row: Dict[str, float] = {
        # --- идентификация / истина
        "seed": w.seed, "window_id": w.window_id, "mode": w.mode, "label": w.label,
        "family": w.family, "via_scope": w.via_scope, "control": w.control,
        "hidden_calls": w.hidden_calls, "culprit": w.culprit, "budget": w.budget,
        # --- STAT_only
        "stat_served_total": w.served_total,
        "stat_streams_total": w.streams_total,
        "stat_lam_hat": w.lam_hat,
        "stat_et_hat": w.et_hat,
        "stat_n_consumers": float(w.n_consumers),
        "stat_notif_delivered": w.notif_delivered,
        "stat_subs_active": w.subs_active,
        "stat_unmatched": w.unmatched_responses,
        "stat_gran": w.export_granularity,
        "stat_slot_served_mean": float(ss.mean()) if ss.size else 0.0,
        "stat_slot_served_std": float(ss.std()) if ss.size else 0.0,
        "stat_slot_served_max": float(ss.max()) if ss.size else 0.0,
        "stat_slot_streams_mean": float(sx.mean()) if sx.size else 0.0,
        "stat_slot_streams_std": float(sx.std()) if sx.size else 0.0,
        "stat_slot_streams_max": float(sx.max()) if sx.size else 0.0,
        "stat_cons_served_max": float(cs.max()) if cs.size else 0.0,
        "stat_cons_served_std": float(cs.std()) if cs.size else 0.0,
        "stat_load_ratio": float(w.served_total / max(w.lam_hat * w.window_s, 1.0)),
        # --- INV_ATTRIB
        "inv_delta1": inv.delta1, "inv_delta2": inv.delta2, "inv_delta3": inv.delta3,
        "inv_delta4": inv.delta4, "inv_delta5": inv.delta5,
        "inv_sigma1": inv.sigma1,
        "inv_z1": inv.z1, "inv_absz1": abs(inv.z1),
        "inv_z4": inv.z4, "inv_z5": inv.z5,
        "inv_cusum": inv.cusum,
        "inv_att_zmax": inv.att_zmax, "inv_att_z2": inv.att_z2,
        "inv_att_gap": inv.att_zmax - inv.att_z2,
        "inv_gate": float(inv.gate_conformance),
        "inv_scope_rate": float(inv.delta2 / max(w.served_total, 1.0)),
    }
    if out is not None:
        row.update({
            "inv_z1s": out.z1s, "inv_z4s": out.z4s, "inv_z5s": out.z5s,
            "inv_cusums": out.cusums, "inv_atts": out.atts,
            "inv_margin": min(out.margin, 1e3),
            "det_fired_inv": float(out.fired_inv),
            "det_fired_att": float(out.fired_att),
            "det_fired_combined": float(out.fired_combined),
            "det_culprit_hat": out.culprit_hat,
        })
    return row


FEATURE_PREFIX = {"STAT_only": ("stat_",), "INV_ATTRIB": ("inv_",), "ALL": ("stat_", "inv_")}
