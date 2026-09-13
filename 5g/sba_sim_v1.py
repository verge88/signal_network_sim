"""
sba_sim_v1.py
Симулятор 5G SBA-сигнализации: HTTP/2-транспорт, NRF/OAuth2, producer ledger,
телеметрия окнами по 600 с (ПАТЧ D) и стратифицированный сплит calib/eval (ПАТЧ E).

ПАТЧ 1: UUID генерируется из 16 случайных байт (int64-overflow устранён).
ПАТЧ 2: GOAWAY несёт last_stream_id последнего обработанного клиентского потока.
ПАТЧ A: выдача access-токенов гарантированно покрывает окно (nbf <= t0, exp >= t1),
        fallback на «активный токен» удалён — неконформность больше не маскируется.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from sba_invariant import (HiddenPlan, WindowObs, window_feature_row)
from sba_masking import FAMILIES, PARAM_DIM, apply_masking

# --------------------------------------------------------------------------------------
# ПАТЧ 1: UUID
# --------------------------------------------------------------------------------------

def rng_uuid4(rng: np.random.Generator) -> uuid.UUID:
    """UUIDv4 из 16 случайных байт. НЕ использовать rng.integers(0, 2**128)."""
    return uuid.UUID(bytes=bytes(rng.integers(0, 256, size=16, dtype=np.uint8).tobytes()),
                     version=4)


# --------------------------------------------------------------------------------------
# HTTP/2 (RFC 9113)
# --------------------------------------------------------------------------------------

@dataclass
class Http2Connection:
    conn_id: str
    max_concurrent_streams: int = 128
    initial_window: int = 65535
    _next_client_stream: int = 1
    _last_peer_stream: int = 0
    _open: Dict[int, float] = field(default_factory=dict)
    goaway_sent: bool = False
    goaway_last_stream_id: int = 0
    streams_opened: int = 0
    streams_refused: int = 0

    def open_stream(self, t: float) -> Optional[int]:
        if self.goaway_sent:
            self.streams_refused += 1
            return None
        if len(self._open) >= self.max_concurrent_streams:
            self.streams_refused += 1
            return None
        sid = self._next_client_stream
        self._next_client_stream += 2               # клиентские потоки — нечётные
        self._open[sid] = t
        self._last_peer_stream = max(self._last_peer_stream, sid)
        self.streams_opened += 1
        return sid

    def close_stream(self, sid: int) -> None:
        self._open.pop(sid, None)

    def send_goaway(self) -> int:
        """ПАТЧ 2: last_stream_id = наибольший ОБРАБОТАННЫЙ поток инициатора."""
        self.goaway_sent = True
        self.goaway_last_stream_id = self._last_peer_stream
        return self.goaway_last_stream_id


# --------------------------------------------------------------------------------------
# NRF / OAuth2 (TS 29.510), ПАТЧ A
# --------------------------------------------------------------------------------------

@dataclass
class AccessToken:
    token_id: str
    consumer: str
    producer_type: str
    service: str
    nbf: float
    exp: float

    def covers(self, t: float, service: str) -> bool:
        return (self.service == service) and (self.nbf <= t <= self.exp)


class NrfTokenService:
    """Выдаёт токены так, чтобы КАЖДАЯ легитимная пара (consumer, service) была
    покрыта на всём интервале окна: nbf <= t0 - margin, exp >= t1 + margin."""

    def __init__(self, rng: np.random.Generator, lifetime_s: float = 1800.0,
                 margin_s: float = 60.0):
        self.rng = rng
        self.lifetime_s = lifetime_s
        self.margin_s = margin_s
        self.store: Dict[Tuple[str, str], AccessToken] = {}
        self.issued = 0

    def ensure_coverage(self, consumer: str, producer_type: str, service: str,
                        t0: float, t1: float) -> AccessToken:
        key = (consumer, service)
        tok = self.store.get(key)
        if tok is None or tok.nbf > t0 - self.margin_s or tok.exp < t1 + self.margin_s:
            tok = AccessToken(token_id=str(rng_uuid4(self.rng)), consumer=consumer,
                              producer_type=producer_type, service=service,
                              nbf=t0 - self.margin_s,
                              exp=t1 + self.margin_s + self.lifetime_s)
            self.store[key] = tok
            self.issued += 1
        return tok


@dataclass
class ProducerLedger:
    """Счётчик выполненных вызовов с гранулярностью экспорта g (>0)."""
    granularity: float = 1.0

    def export(self, value: float) -> float:
        g = max(float(self.granularity), 1e-9)
        return float(np.round(value / g) * g)


# --------------------------------------------------------------------------------------
# Конфигурация
# --------------------------------------------------------------------------------------

SERVICES = ["nudm-sdm", "nudm-uecm", "nausf-auth", "npcf-smpolicycontrol",
            "nsmf-pdusession", "namf-comm"]
PRODUCER_TYPE = {"nudm-sdm": "UDM", "nudm-uecm": "UDM", "nausf-auth": "AUSF",
                 "npcf-smpolicycontrol": "PCF", "nsmf-pdusession": "SMF",
                 "namf-comm": "AMF"}


@dataclass
class SbaConfig:
    window_s: float = 600.0            # ПАТЧ D
    n_slots: int = 10
    n_consumers: int = 12
    lam_base: float = 40.0             # вызовов/с суммарно
    et_mean: float = 0.05              # E[T] сервисного вызова, с
    export_granularity: float = 1.0
    unmatched_rate: float = 0.002      # доля непарных ответов (таймауты) в норме
    subs_per_consumer: float = 40.0
    notif_rate: float = 0.05           # нотификаций на подписку за окно
    diurnal_amp: float = 0.35
    max_concurrent_streams: int = 128


@dataclass
class DatasetSpec:
    seed: int = 0
    cfg: SbaConfig = field(default_factory=SbaConfig)
    n_benign_windows: int = 700        # ПАТЧ E
    calib_frac: float = 0.5
    hidden_grid: Tuple[int, ...] = (5, 10, 20, 40, 80, 160)
    via_scopes: Tuple[str, ...] = ("no_token", "scp_bypass", "notify_abuse")
    families: Tuple[str, ...] = tuple(FAMILIES)
    reps_per_cell: int = 2
    budgets: Tuple[float, ...] = (0.0, 0.5, 1.0)
    controls: Tuple[str, ...] = ("diurnal_peak", "scale_in", "api_change", "code_upgrade")
    n_control_windows: int = 60

    @staticmethod
    def quick(seed: int = 0) -> "DatasetSpec":
        return DatasetSpec(seed=seed, n_benign_windows=220, hidden_grid=(10, 40, 160),
                           reps_per_cell=1, n_control_windows=25)


# --------------------------------------------------------------------------------------
# Симулятор
# --------------------------------------------------------------------------------------

class Simulator:
    def __init__(self, cfg: SbaConfig, seed: int = 0):
        self.cfg = cfg
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.nrf = NrfTokenService(self.rng)
        self.ledger = ProducerLedger(granularity=cfg.export_granularity)
        self.conns: Dict[str, Http2Connection] = {}
        self.t = 0.0
        self.consumer_ids = [f"nf-consumer-{i:02d}" for i in range(cfg.n_consumers)]

    # --- операционный контекст окна -----------------------------------------------
    def _context(self, window_id: int, control: str) -> Tuple[float, float, int]:
        cfg = self.cfg
        phase = 2.0 * math.pi * (window_id % 144) / 144.0
        lam = cfg.lam_base * (1.0 + cfg.diurnal_amp * math.sin(phase))
        et = cfg.et_mean
        n_active = cfg.n_consumers
        if control == "diurnal_peak":
            lam *= 3.0
        elif control == "scale_in":
            n_active = max(cfg.n_consumers // 2, 2)
            lam *= 1.6
        elif control == "api_change":
            et *= 1.5
        elif control == "code_upgrade":
            et *= 0.6
            lam *= 1.15
        return float(lam), float(et), int(n_active)

    def benign_window(self, window_id: int, mode: str = "eval",
                      control: str = "") -> WindowObs:
        """Чистое окно. По построению out_of_scope = 0 и orphan = 0 (ПАТЧ A)."""
        cfg = self.cfg
        rng = self.rng
        lam, et, n_active = self._context(window_id, control)
        t0 = self.t
        t1 = t0 + cfg.window_s
        g = cfg.export_granularity
        n_c = cfg.n_consumers
        n_s = cfg.n_slots

        # --- распределение нагрузки по консьюмерам
        shares = np.zeros(n_c)
        w_act = rng.dirichlet(np.full(n_active, 8.0))
        shares[:n_active] = w_act
        lam_c = lam * shares

        # --- ПАТЧ A: полное токенное покрытие всех используемых пар
        for i in range(n_c):
            if lam_c[i] <= 0:
                continue
            for svc in SERVICES:
                self.nrf.ensure_coverage(self.consumer_ids[i], PRODUCER_TYPE[svc], svc, t0, t1)

        # --- HTTP/2 соединения
        for i in range(n_active):
            cid = f"{self.consumer_ids[i]}->scp"
            if cid not in self.conns:
                self.conns[cid] = Http2Connection(
                    conn_id=cid, max_concurrent_streams=cfg.max_concurrent_streams)

        # --- события
        arrivals_c = rng.poisson(np.maximum(lam_c * cfg.window_s, 0.0))
        inflight_start = rng.poisson(np.maximum(lam_c * et, 0.0))
        inflight_end = rng.poisson(np.maximum(lam_c * et, 0.0))
        served_c_true = arrivals_c + inflight_start - inflight_end
        streams_c_true = arrivals_c

        cons_served = np.array([self.ledger.export(v) for v in served_c_true], dtype=float)
        cons_streams = np.array([self.ledger.export(v) for v in streams_c_true], dtype=float)
        served_total = float(cons_served.sum())
        streams_total = float(cons_streams.sum())

        # --- слоты
        arr_total = int(arrivals_c.sum())
        slot_w = rng.dirichlet(np.full(n_s, 20.0))
        slot_arr = rng.multinomial(arr_total, slot_w).astype(float)
        s_start = rng.poisson(max(lam * et, 0.0), size=n_s).astype(float)
        s_end = rng.poisson(max(lam * et, 0.0), size=n_s).astype(float)
        slot_served = np.array([self.ledger.export(v) for v in (slot_arr + s_start - s_end)])
        slot_streams = np.array([self.ledger.export(v) for v in slot_arr])

        # --- подписки / нотификации / непарные ответы
        subs_active = float(n_active * cfg.subs_per_consumer)
        notif_delivered = float(rng.poisson(subs_active * cfg.notif_rate))
        mu4 = max(cfg.unmatched_rate * lam * cfg.window_s, 1.0)
        unmatched = float(rng.poisson(mu4))

        # --- измеренные (репортируемые) параметры нагрузки
        lam_hat = streams_total / cfg.window_s
        et_hat = et * float(rng.normal(1.0, 0.03))

        self.t = t1
        return WindowObs(
            seed=self.seed, window_id=window_id, mode=mode, label=0, family="none",
            via_scope="", control=control, hidden_calls=0, culprit="", budget=0.0,
            window_s=cfg.window_s, n_slots=n_s, export_granularity=g,
            lam_hat=float(lam_hat), et_hat=float(max(et_hat, 1e-4)), n_consumers=n_c,
            consumer_ids=list(self.consumer_ids), cons_lam=lam_c,
            served_total=served_total, streams_total=streams_total,
            out_of_scope=0.0, late_token=0.0, orphan_notifications=0.0,
            unmatched_responses=unmatched, unmatched_expected=mu4,
            notif_delivered=notif_delivered, subs_active=subs_active,
            slot_served=slot_served, slot_streams=slot_streams,
            cons_served=cons_served, cons_streams=cons_streams,
            cons_tokens=np.ones(n_c, dtype=float),
        )

    def attack_window(self, window_id: int, hidden: int, via_scope: str, family: str,
                      budget: float, params: Optional[np.ndarray] = None,
                      mode: str = "eval") -> WindowObs:
        base = self.benign_window(window_id, mode=mode, control="")
        culprit_idx = int(self.rng.integers(0, base.n_consumers))
        plan = HiddenPlan(hidden_calls=int(hidden), culprit_idx=culprit_idx,
                          via_scope=via_scope)
        if params is None:
            params = self.rng.random(PARAM_DIM.get(family, 2))
        w = apply_masking(base, plan, family, params, budget, self.rng)
        w.mode = mode
        return w


# --------------------------------------------------------------------------------------
# Генерация датасета (ПАТЧ E: стратифицированный сплит calib/eval)
# --------------------------------------------------------------------------------------

@dataclass
class Dataset:
    df: pd.DataFrame
    windows: List[WindowObs]
    spec: DatasetSpec

    def calib(self) -> List[WindowObs]:
        return [w for w in self.windows if w.mode == "calib"]

    def eval_benign(self) -> List[WindowObs]:
        return [w for w in self.windows if w.mode == "eval" and w.label == 0 and not w.control]

    def eval_controls(self) -> List[WindowObs]:
        return [w for w in self.windows if w.mode == "eval" and w.label == 0 and w.control]

    def attacks(self) -> List[WindowObs]:
        return [w for w in self.windows if w.label == 1]


def generate_dataset(spec: DatasetSpec, det=None) -> Dataset:
    sim = Simulator(spec.cfg, seed=spec.seed)
    rng = sim.rng
    windows: List[WindowObs] = []
    wid = 0

    # --- чистые окна: стратификация по фазе суток между calib и eval
    n_cal = int(round(spec.n_benign_windows * spec.calib_frac))
    for i in range(spec.n_benign_windows):
        mode = "calib" if (i % 2 == 0 and sum(1 for w in windows if w.mode == "calib") < n_cal) \
            else "eval"
        windows.append(sim.benign_window(wid, mode=mode, control=""))
        wid += 1

    # --- негативные контролы (только eval)
    for ctrl in spec.controls:
        for _ in range(spec.n_control_windows):
            windows.append(sim.benign_window(wid, mode="eval", control=ctrl))
            wid += 1

    # --- атаки
    for family in spec.families:
        for via in spec.via_scopes:
            for h in spec.hidden_grid:
                for _ in range(spec.reps_per_cell):
                    b = float(rng.choice(np.asarray(spec.budgets, dtype=float)))
                    windows.append(sim.attack_window(wid, h, via, family, b, mode="eval"))
                    wid += 1

    df = pd.DataFrame([window_feature_row(w, det) for w in windows])
    return Dataset(df=df, windows=windows, spec=spec)
