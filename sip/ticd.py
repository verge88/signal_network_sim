"""
TICD — Transaction-Invariant Consensus Detector
=================================================
Обнаружение компрометации узла в иерархической SIP-сети БЕЗ МЕТОК АТАК
(one-class, semi-supervised на норме). Опирается на два физически
обоснованных инварианта транзакционного автомата RFC 3261 §13:

  I1 (закон сохранения состояния):
       Δ = |1 - (dcr + ddr)|
     Для честного узла Δ ≈ 0. Теоретически (см. analyze_invariant.py):
       E[Δ] ≥ (1-δ0)·cr - 2·εz    (не устраним никакой маскировкой)

  I2 (согласованность с ACK-потоком, Вариант A):
       M = max(0, dcr_report - dcr_physical),  dcr_physical = n_ack/n_200
     Мастер независимо наблюдает ACK; заявить замыканий больше физических
     невозможно без вскрытия.

TICD НЕ использует is_anomaly / attack_type при обучении. Он строит
модель НОРМЫ в зональном контексте и детектирует нарушение инвариантов
относительно зонального консенсуса.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.covariance import MinCovDet


class TICDetector:
    """
    One-class детектор. Обучается ТОЛЬКО на норме (is_anomaly==0), включая
    легитимные события (traffic_spike/maintenance/...) для устойчивости к FP.

    Скор = взвешенная комбинация:
      s_inv  — нарушение транзакционных инвариантов I1, I2 (физика RFC §13);
      s_ctx  — зональное отклонение инвариантов (robust Mahalanobis по норме);
      s_cons — консенсусное отклонение узла от зоны по объёмным метрикам.

    Инвариантная часть s_inv не обучается на данных — она вытекает из
    структуры автомата, поэтому робастна к adaptive-противнику, стирающему
    корреляционные следы.
    """

    # Признаки-инварианты (физика автомата)
    # Признаки-инварианты (физика автомата, Вариант A).
    # ГЛАВНЫЙ — I2 (транзитный ACK-разрыв), неустранимый маскировкой.
    # I1 (dcr+ddr) оставлен как слабый вторичный сигнал (ловит naive).
    INVARIANT_FEATURES = [
        "cu_ack_closure_mismatch",       # I2 — ОСНОВНОЙ рычаг (Вариант A)
        "dev_ack_mismatch_vs_zone",      # I2 в зональном контексте
        "cu_txn_state_consistency_error",  # I1 — слабый, ловит naive
    ]

    # Контекстные признаки для зонального консенсуса (нормальная модель)
    CONTEXT_FEATURES = [
        "obs_dialog_completion_ratio", "obs_dangling_dialog_ratio",
        "obs_forking_amplification", "obs_inbound_outbound_ratio",
        "dev_dialog_completion_ratio_vs_zone",
        "dev_dangling_dialog_ratio_vs_zone",
        "dev_forking_amplification_vs_zone", "dev_io_ratio_vs_zone",
        "obs_invite_timeout_ratio", "obs_invite_ack_ratio",
    ]

    def __init__(self, label_col="is_anomaly",
                 w_inv=0.6, w_ctx=0.25, w_cons=0.15,
                 use_zone_context=True, seed=42):

        self.label_col = label_col
        self.w_inv, self.w_ctx, self.w_cons = w_inv, w_ctx, w_cons
        self.use_zone_context = use_zone_context
        self.seed = seed
        self.inv_cols = None
        self.ctx_cols = None
        self.ctx_scaler = StandardScaler()
        self.ctx_imputer = SimpleImputer(strategy="median")
        self.robust_cov = None
        self.inv_norm = {}      # нормировка инвариантов по норме (P99)
        self.fitted = False

    def _present(self, df, cols):
        return [c for c in cols if c in df.columns]

    def fit(self, train_df, val_df=None):
        # ВАЖНО: обучаемся ТОЛЬКО на норме, без меток атак.
        norm = train_df[train_df[self.label_col] == 0].copy()
        self.inv_cols = self._present(train_df, self.INVARIANT_FEATURES)
        self.ctx_cols = self._present(train_df, self.CONTEXT_FEATURES)

        # Нормировка инвариантов: P99 нормальных значений как масштаб.
        for c in self.inv_cols:
            v = norm[c].to_numpy(dtype=float)
            v = v[np.isfinite(v)]
            p99 = np.percentile(np.abs(v), 99) if v.size else 1.0
            self.inv_norm[c] = max(p99, 1e-6)

        # Robust Mahalanobis по контекстным признакам НОРМЫ (для s_ctx).
        if self.use_zone_context and self.ctx_cols:
            Xc = norm[self.ctx_cols]
            Xc = self.ctx_imputer.fit_transform(Xc)
            Xc = self.ctx_scaler.fit_transform(Xc)
            try:
                self.robust_cov = MinCovDet(random_state=self.seed).fit(Xc)
            except Exception:
                self.robust_cov = None
        self.fitted = True
        return self

    def _invariant_score(self, df):
        """Физический скор. I2 (транзитный ACK) доминирует над I1."""
        weights = {
            "cu_ack_closure_mismatch": 2.0,     # I2: главный, ×2
            "dev_ack_mismatch_vs_zone": 2.0,    # I2 зональный, ×2
            "cu_txn_state_consistency_error": 0.5,  # I1: слабый, ×0.5
        }
        s = np.zeros(len(df), dtype=float)
        wsum = 0.0
        for c in self.inv_cols:
            w = weights.get(c, 1.0)
            v = np.nan_to_num(df[c].to_numpy(dtype=float), nan=0.0)
            s = s + w * np.clip(np.abs(v) / self.inv_norm[c], 0, 5)
            wsum += w
        return s / max(1e-6, wsum)


    def _context_score(self, df):
        """Robust Mahalanobis-расстояние от нормальной модели контекста."""
        if self.robust_cov is None or not self.ctx_cols:
            return np.zeros(len(df))
        Xc = self.ctx_imputer.transform(df[self.ctx_cols])
        Xc = self.ctx_scaler.transform(Xc)
        md = self.robust_cov.mahalanobis(Xc)
        return np.sqrt(np.clip(md, 0, None))

    def _consensus_score(self, df):
        """
        Зональный консенсус: насколько узел отклоняется от медианы своей
        зоны по объёмным dev_-признакам. Ловит координированную и
        одиночную компрометацию как разрыв зонального согласия.
        """
        cols = self._present(df, [
            "dev_total_messages_vs_zone", "dev_io_ratio_vs_zone",
            "dev_var_total_vs_zone"])
        if not cols:
            return np.zeros(len(df))
        s = np.zeros(len(df))
        for c in cols:
            v = df[c].to_numpy(dtype=float)
            v = np.nan_to_num(v, nan=0.0)
            sd = np.std(v) + 1e-6
            s = s + np.clip(np.abs(v) / sd, 0, 5)
        return s / len(cols)

    def score(self, df):
        assert self.fitted, "TICD не обучен"
        s_inv = self._invariant_score(df)
        s_ctx = self._context_score(df)
        s_cons = self._consensus_score(df)

        def norm01(x):
            p1, p99 = np.percentile(x, 1), np.percentile(x, 99)
            return np.clip((x - p1) / (p99 - p1 + 1e-9), 0, 1)

        return (self.w_inv * norm01(s_inv) +
                self.w_ctx * norm01(s_ctx) +
                self.w_cons * norm01(s_cons))

    def score_components(self, df):
        """Для абляции: вернуть компоненты по отдельности."""
        return {
            "invariant": self._invariant_score(df),
            "context": self._context_score(df),
            "consensus": self._consensus_score(df),
        }
