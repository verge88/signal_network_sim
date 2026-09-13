"""
sba_training_v1.py
ML только как КОНТРОЛЬНАЯ ГРУППА для ablation (STAT_only vs INV_ATTRIB vs ALL).
Детектирование в основном тракте выполняется аналитическими инвариантами.

Порог выбирается по чистым окнам train-части на уровне FPR = 1 % (base-rate-correct),
метрика — recall на held-out семействе маскирования (LOFO).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd

from sba_invariant import FEATURE_PREFIX

try:                                     # sklearn опционален
    from sklearn.ensemble import HistGradientBoostingClassifier
    _HAS_SK = True
except Exception:                        # pragma: no cover
    _HAS_SK = False


class NumpyLogReg:
    """Fallback-классификатор: логистическая регрессия на numpy."""

    def __init__(self, lr: float = 0.1, epochs: int = 600, l2: float = 1e-3):
        self.lr, self.epochs, self.l2 = lr, epochs, l2
        self.w: Optional[np.ndarray] = None
        self.b = 0.0
        self.mu: Optional[np.ndarray] = None
        self.sd: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NumpyLogReg":
        X = np.asarray(X, dtype=float)
        self.mu = X.mean(axis=0)
        self.sd = np.maximum(X.std(axis=0), 1e-6)
        Z = (X - self.mu) / self.sd
        n, d = Z.shape
        self.w = np.zeros(d)
        y = np.asarray(y, dtype=float)
        for _ in range(self.epochs):
            p = 1.0 / (1.0 + np.exp(-(Z @ self.w + self.b)))
            gw = Z.T @ (p - y) / n + self.l2 * self.w
            gb = float(np.mean(p - y))
            self.w -= self.lr * gw
            self.b -= self.lr * gb
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Z = (np.asarray(X, dtype=float) - self.mu) / self.sd
        p = 1.0 / (1.0 + np.exp(-(Z @ self.w + self.b)))
        return np.column_stack([1.0 - p, p])


def feature_columns(df: pd.DataFrame, feature_set: str) -> List[str]:
    prefixes = FEATURE_PREFIX[feature_set]
    cols = [c for c in df.columns
            if any(c.startswith(p) for p in prefixes)
            and pd.api.types.is_numeric_dtype(df[c])]
    return sorted(cols)


def _make_model(seed: int = 0):
    if _HAS_SK:
        return HistGradientBoostingClassifier(max_depth=4, max_iter=180,
                                              learning_rate=0.08, random_state=seed)
    return NumpyLogReg()


@dataclass
class TrainResult:
    feature_set: str
    holdout_family: str
    recall_at_fpr1: float
    threshold: float
    auc: float
    n_train: int
    n_test_pos: int
    n_test_neg: int


def _auc(y: np.ndarray, s: np.ndarray) -> float:
    y = np.asarray(y, dtype=int)
    s = np.asarray(s, dtype=float)
    pos, neg = s[y == 1], s[y == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, order.size + 1)
    r_pos = ranks[:pos.size].sum()
    return float((r_pos - pos.size * (pos.size + 1) / 2.0) / (pos.size * neg.size))


def train_eval(df_train: pd.DataFrame, df_test: pd.DataFrame, feature_set: str,
               holdout_family: str = "", seed: int = 0,
               target_fpr: float = 0.01) -> TrainResult:
    cols = feature_columns(df_train, feature_set)
    Xtr = df_train[cols].to_numpy(dtype=float)
    ytr = df_train["label"].to_numpy(dtype=int)
    model = _make_model(seed).fit(Xtr, ytr)

    s_tr_neg = model.predict_proba(Xtr[ytr == 0])[:, 1]
    thr = float(np.quantile(s_tr_neg, 1.0 - target_fpr))

    Xte = df_test[cols].to_numpy(dtype=float)
    yte = df_test["label"].to_numpy(dtype=int)
    s_te = model.predict_proba(Xte)[:, 1]
    pos = yte == 1
    recall = float(np.mean(s_te[pos] > thr)) if pos.any() else float("nan")

    return TrainResult(feature_set=feature_set, holdout_family=holdout_family,
                       recall_at_fpr1=recall, threshold=thr, auc=_auc(yte, s_te),
                       n_train=int(Xtr.shape[0]), n_test_pos=int(pos.sum()),
                       n_test_neg=int((~pos).sum()))
