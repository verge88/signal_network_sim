"""
ss7_training_v7.py — пайплайн обучения и оценки SS7-детекторов (исправленная версия).

Заменяет полностью закомментированный ss7_training_v7.py (3778 строк).

Что исправлено относительно исходной версии
-------------------------------------------
 1. НУЛЕВОЙ ТЕСТ КАК ШЛЮЗ. Перед любым обучением проверяется, что при h=0
    компрометированный узел неотличим от честного (AUC ~ 0.5). Провал = отказ
    обучаться. Обойти можно только явным --i-know-results-are-untrusted.
 2. РЕЕСТР ПРИЗНАКОВ. Ни один столбец не попадает в матрицу, если он не
    выводится из FEATURE_REGISTRY. Guard ловит и утечки (label, hidden_msgs,
    episode_id, mask_family, soph, nid, t), и «случайно добавленные» признаки
    вроде rtt_ms, которых нет в предрегистрации.
 3. БЛОЧНЫЙ СПЛИТ по времени с purge-зазором >= окна признаков + эпизод целиком
    в одном сплите. Вместо train_test_split по i.i.d.-записям.
 4. ПОРОГ ТОЛЬКО ПО VAL, из операционного бюджета тревог/сутки. Оракульный
    recall@FPR по тестовым негативам и max-F1 на тех же данных удалены.
 5. ИНЦИДЕНТНЫЕ МЕТРИКИ: recall по эпизодам, время до обнаружения, тревоги в
    сутки на сеть, инцидентная точность по кластеризованным ложным сериям,
    плюс пересчёт точности на операционную априорную частоту.
 6. RULE-BASELINE обязателен. Без него непонятно, что даёт ML поверх правил.
 7. FOCAL LOSS ДЛЯ АВТОЭНКОДЕРА УБРАН: он требует метки и превращает
    «unsupervised» модель в скрытно-надзорную. Автоэнкодеры учатся только на
    benign-части train.
 8. СТЭКИНГ БЕЗ УТЕЧКИ: базовые модели учатся на группе A внутри train,
    мета-признаки и мета-модель — на группе B (группы = эпизоды/блоки).
 9. ПОСЛЕДОВАТЕЛЬНЫЕ МОДЕЛИ не склеивают окна через границы сплита и через
    разрывы по времени; окна только из прошлого.
10. МУЛЬТИСИД С ПЕРЕГЕНЕРАЦИЕЙ сети и трафика (regen по умолчанию), бутстрэп
    по сидам, поправка Холма, объявленный минимально обнаружимый эффект.
11. СТРАТИФИКАЦИЯ ПО ИЗОЩРЁННОСТИ: головная цифра — ADAPTIVE, а не среднее.
12. НЕТ ПОБОЧНЫХ ЭФФЕКТОВ ПРИ ИМПОРТЕ: ни filterwarnings, ни OMP_NUM_THREADS,
    ни matplotlib.use на уровне модуля.
13. Датасет генерируется и кэшируется по хешу конфига; 100-МБ CSV в репозиторий
    не коммитится.

Зависимости: numpy, pandas, scikit-learn, networkx. torch — опционален
(без него нейросетевые модели молча выпадают из зоопарка, PCA-baseline остаётся).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.ensemble import (HistGradientBoostingClassifier, IsolationForest,
                              RandomForestClassifier)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

# --- импорт стенда: fix.py хранит объединённую реализацию модулей -----------
try:
    from .fix import (INTERVALS_PER_DAY, KeyCustody, SimConfig,
                      Sophistication, MASK_FAMILIES,
                      assert_mask_is_identity_at_zero, build_dataset,
                      rule_baseline, threshold_for_alert_budget,
                      FEATURE_REGISTRY, FEATURE_SETS, INTEG, INV, OWN,
                      null_test, RngBook, blocked_split)
except ImportError:                                             # прямой запуск
    from fix import (INTERVALS_PER_DAY, KeyCustody, SimConfig,
                     Sophistication, MASK_FAMILIES,
                     assert_mask_is_identity_at_zero, build_dataset,
                     rule_baseline, threshold_for_alert_budget,
                     FEATURE_REGISTRY, FEATURE_SETS, INTEG, INV, OWN,
                     null_test, RngBook, blocked_split)

try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except Exception:                                               # pragma: no cover
    _HAS_TORCH = False


# =============================================================================
#  Конфигурация обучения
# =============================================================================

DERIVED_SUFFIXES = ("__mean", "__std", "__z", "__dev_zone")

# Столбцы, присутствие которых в матрице признаков — утечка ground truth.
LEAKY_EXACT = frozenset({
    "label", "hidden_msgs", "episode_id", "mask_family", "soph",
    "nid", "t", "zone", "ntype",
})
LEAKY_PREFIXES = ("gt_", "true_", "hidden_", "_debug", "culprit")

# Предрегистрированные сравнения (объявлены ДО прогонов, менять post hoc нельзя).
DEFAULT_COMPARISONS: Tuple[Tuple[str, str], ...] = (
    ("random_forest", "rule_baseline"),
    ("hist_gb", "rule_baseline"),
    ("meta_ensemble", "random_forest"),
    ("autoencoder", "pca_recon"),
    ("isolation_forest", "rule_baseline"),
)


@dataclass
class TrainingConfig:
    seed: int = 42
    days: float = 6.0
    key_custody: str = KeyCustody.NODE_SOFTWARE.value
    sophistication: str = Sophistication.ADAPTIVE.value
    feature_set: str = "OWN+INV"

    # операционные параметры оценки
    alerts_per_day: float = 20.0        # бюджет тревог на всю сеть
    k_consec: int = 3                   # подряд идущих окон для инцидента
    incident_gap: int = 10              # склейка ложных серий в один инцидент
    op_prevalence: float = 1e-4         # реальная априорная частота компрометации

    # модели
    use_torch: bool = True
    ae_epochs: int = 120
    ae_hidden: int = 32
    ae_latent: int = 8
    seq_len: int = 24
    seq_epochs: int = 60
    ocsvm_max_train: int = 6000
    quick: bool = False

    # воспроизводимость / артефакты
    out_dir: str = "ss7_runs"
    cache_dir: str = ".ss7_cache"
    use_cache: bool = True

    def sim_config(self) -> SimConfig:
        return SimConfig(seed=self.seed, days=self.days,
                         key_custody=KeyCustody(self.key_custody))

    def fingerprint(self) -> str:
        d = {k: v for k, v in asdict(self).items()
             if k not in ("out_dir", "cache_dir", "use_cache", "quick")}
        return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:16]


# =============================================================================
#  Guard утечек и сборка матрицы признаков
# =============================================================================

class LeakageError(RuntimeError):
    pass


def resolve_feature_columns(df: pd.DataFrame, base_names: Sequence[str]) -> List[str]:
    """Разворачивает реестр в реальные столбцы (база + разрешённые суффиксы)."""
    cols: List[str] = []
    for b in base_names:
        if b in df.columns:
            cols.append(b)
        for s in DERIVED_SUFFIXES:
            c = f"{b}{s}"
            if c in df.columns:
                cols.append(c)
    return cols


def assert_no_leakage(cols: Sequence[str]) -> None:
    """Каждый столбец обязан выводиться из FEATURE_REGISTRY. Иначе — отказ."""
    bad_leak, bad_unregistered = [], []
    for c in cols:
        if c in LEAKY_EXACT or any(c.startswith(p) for p in LEAKY_PREFIXES):
            bad_leak.append(c)
            continue
        base = c
        for s in DERIVED_SUFFIXES:
            if c.endswith(s):
                base = c[: -len(s)]
                break
        if base not in FEATURE_REGISTRY:
            bad_unregistered.append(c)
    if bad_leak or bad_unregistered:
        raise LeakageError(
            f"guard признаков: утечка={bad_leak}, вне реестра={bad_unregistered}. "
            f"Признак без обоснования в FEATURE_REGISTRY в матрицу не попадает.")


def group_key(df: pd.DataFrame, block: int = 200) -> np.ndarray:
    """Группы для кросс-валидации: эпизод целиком либо блок времени."""
    ep = df["episode_id"].to_numpy()
    t = df["t"].to_numpy()
    return np.where(ep >= 0, ep, -1 - (t // block))


@dataclass
class Matrices:
    cols: List[str]
    Xtr: np.ndarray
    Xva: np.ndarray
    Xte: np.ndarray
    ytr: np.ndarray
    yva: np.ndarray
    yte: np.ndarray
    gtr: np.ndarray
    tr: pd.DataFrame
    va: pd.DataFrame
    te: pd.DataFrame


def prepare_matrices(tr: pd.DataFrame, va: pd.DataFrame, te: pd.DataFrame,
                     cols: Sequence[str]) -> Matrices:
    """Импьютер и скейлер обучаются ТОЛЬКО на train."""
    assert_no_leakage(cols)
    cols = list(cols)
    imp = SimpleImputer(strategy="median").fit(tr[cols])
    sc = StandardScaler().fit(imp.transform(tr[cols]))
    f = lambda d: np.asarray(sc.transform(imp.transform(d[cols])), dtype=np.float64)
    return Matrices(cols=cols, Xtr=f(tr), Xva=f(va), Xte=f(te),
                    ytr=tr["label"].to_numpy(int), yva=va["label"].to_numpy(int),
                    yte=te["label"].to_numpy(int), gtr=group_key(tr),
                    tr=tr, va=va, te=te)


# =============================================================================
#  Инцидентные метрики
# =============================================================================

def _runs(mask: np.ndarray, times: np.ndarray, gap: int) -> List[Tuple[int, int]]:
    """Склеивает срабатывания в серии, разрывы <= gap считаются одной серией."""
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    out, s = [], idx[0]
    for a, b in zip(idx[:-1], idx[1:]):
        if times[b] - times[a] > gap:
            out.append((s, a))
            s = b
    out.append((s, idx[-1]))
    return out


def incident_metrics(df: pd.DataFrame, scores: np.ndarray, thr: float,
                     cfg: TrainingConfig) -> Dict[str, float]:
    """Recall по инцидентам, TTD, тревоги/сутки, инцидентная точность."""
    d = df.copy()
    d["score"] = scores
    d["fire"] = (np.nan_to_num(scores, nan=-np.inf) > thr).astype(int)

    det, ttd = [], []
    for _, g in d[d["episode_id"] >= 0].groupby("episode_id"):
        g = g.sort_values("t")
        f = g["fire"].to_numpy()
        run, hit = 0, None
        for i, v in enumerate(f):
            run = run + 1 if v else 0
            if run >= cfg.k_consec:
                hit = i - cfg.k_consec + 1
                break
        det.append(hit is not None)
        if hit is not None:
            ttd.append(int(g["t"].iloc[hit] - g["t"].iloc[0]))

    # ложные инциденты: серии срабатываний на чистых окнах, по узлам
    false_inc, node_days = 0, 0.0
    for nid, g in d.groupby("nid"):
        g = g.sort_values("t")
        benign = g["label"].to_numpy() == 0
        t = g["t"].to_numpy()
        fire = (g["fire"].to_numpy() == 1) & benign
        for a, b in _runs(fire, t, cfg.incident_gap):
            if (b - a + 1) >= cfg.k_consec:
                false_inc += 1
        node_days += (t.max() - t.min() + 1) / INTERVALS_PER_DAY if len(t) else 0.0

    span = (d["t"].max() - d["t"].min() + 1) / INTERVALS_PER_DAY if len(d) else 1.0
    neg = d[d["label"] == 0]
    tp = int(np.sum(det))
    prec_inc = tp / (tp + false_inc) if (tp + false_inc) > 0 else np.nan

    win_fpr = float(neg["fire"].mean()) if len(neg) else np.nan
    win_rec = float(d.loc[d["label"] == 1, "fire"].mean()) if (d["label"] == 1).any() else np.nan
    pi = cfg.op_prevalence
    prec_op = (pi * win_rec) / (pi * win_rec + (1 - pi) * win_fpr) \
        if (win_rec == win_rec and win_fpr == win_fpr and win_fpr > 0) else np.nan

    return {
        "episode_recall": float(np.mean(det)) if det else np.nan,
        "n_episodes": int(len(det)),
        "median_ttd_intervals": float(np.median(ttd)) if ttd else np.nan,
        "p90_ttd_intervals": float(np.percentile(ttd, 90)) if ttd else np.nan,
        "incident_precision": float(prec_inc),
        "false_incidents_per_day": float(false_inc / max(span, 1e-9)),
        "alerts_per_day_network": float(neg["fire"].sum() / max(span, 1e-9)),
        "window_fpr": win_fpr,
        "window_recall": win_rec,
        "precision_at_op_prevalence": float(prec_op) if prec_op == prec_op else np.nan,
        "n_nodes": int(d["nid"].nunique()),
    }


def score_row(name: str, family: str, M: Matrices, s_va: np.ndarray,
              s_te: np.ndarray, cfg: TrainingConfig) -> Dict[str, float]:
    """Порог — только из val-негативов, по бюджету тревог. Тест не участвует."""
    v = np.nan_to_num(s_va, nan=np.nanmedian(s_va) if np.isfinite(s_va).any() else 0.0)
    thr = threshold_for_alert_budget(v[M.yva == 0], M.te["nid"].nunique(),
                                     cfg.alerts_per_day)
    row = {"model": name, "family": family, "threshold": float(thr)}
    ok = np.isfinite(s_te)
    if ok.sum() > 10 and len(np.unique(M.yte[ok])) > 1:
        row["auc"] = float(roc_auc_score(M.yte[ok], s_te[ok]))
        row["ap"] = float(average_precision_score(M.yte[ok], s_te[ok]))
    else:
        row["auc"] = row["ap"] = np.nan
    row["coverage"] = float(ok.mean())
    row.update(incident_metrics(M.te, s_te, thr, cfg))
    return row


# =============================================================================
#  Зоопарк моделей.  Интерфейс: (M, cfg, seed) -> (s_va, s_te)
# =============================================================================

def m_rule_baseline(M: Matrices, cfg: TrainingConfig, seed: int):
    return rule_baseline(M.va), rule_baseline(M.te)


def m_pca_recon(M: Matrices, cfg: TrainingConfig, seed: int):
    """Всегда доступный baseline реконструкции (не требует torch)."""
    Xb = M.Xtr[M.ytr == 0]
    k = max(2, min(Xb.shape[1] // 3, 12))
    p = PCA(n_components=k, random_state=seed).fit(Xb)
    err = lambda X: ((X - p.inverse_transform(p.transform(X))) ** 2).mean(1)
    return err(M.Xva), err(M.Xte)


def m_isolation_forest(M: Matrices, cfg: TrainingConfig, seed: int):
    f = IsolationForest(n_estimators=400, max_samples="auto",
                        contamination="auto", random_state=seed,
                        n_jobs=1).fit(M.Xtr[M.ytr == 0])
    return -f.score_samples(M.Xva), -f.score_samples(M.Xte)


def m_ocsvm(M: Matrices, cfg: TrainingConfig, seed: int):
    Xb = M.Xtr[M.ytr == 0]
    if len(Xb) > cfg.ocsvm_max_train:
        idx = np.random.default_rng(seed).choice(len(Xb), cfg.ocsvm_max_train,
                                                 replace=False)
        Xb = Xb[idx]
    f = OneClassSVM(kernel="rbf", nu=0.05, gamma="scale").fit(Xb)
    return -f.score_samples(M.Xva), -f.score_samples(M.Xte)


def m_random_forest(M: Matrices, cfg: TrainingConfig, seed: int):
    f = RandomForestClassifier(n_estimators=500, min_samples_leaf=4,
                               class_weight="balanced_subsample",
                               random_state=seed, n_jobs=1).fit(M.Xtr, M.ytr)
    return f.predict_proba(M.Xva)[:, 1], f.predict_proba(M.Xte)[:, 1]


def m_hist_gb(M: Matrices, cfg: TrainingConfig, seed: int):
    w = np.where(M.ytr == 1, max((M.ytr == 0).sum() / max((M.ytr == 1).sum(), 1), 1.0), 1.0)
    f = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06,
                                       min_samples_leaf=20, l2_regularization=1.0,
                                       random_state=seed).fit(M.Xtr, M.ytr,
                                                              sample_weight=w)
    return f.predict_proba(M.Xva)[:, 1], f.predict_proba(M.Xte)[:, 1]


def m_cascade(M: Matrices, cfg: TrainingConfig, seed: int):
    """Двухуровневый каскад: unsupervised-гейт, затем supervised-уточнение.

    Исходная версия использовала «общий детектор аномалий» на метках, которых
    в датасете нет (ext_attack и нормальные события не размечены), — здесь
    гейт честно неразмеченный.
    """
    gate = IsolationForest(n_estimators=300, random_state=seed,
                           n_jobs=1).fit(M.Xtr[M.ytr == 0])
    g_tr, g_va, g_te = (-gate.score_samples(X) for X in (M.Xtr, M.Xva, M.Xte))
    thr = float(np.percentile(g_tr[M.ytr == 0], 80))
    sel = g_tr > thr
    if sel.sum() < 50 or len(np.unique(M.ytr[sel])) < 2:
        return g_va, g_te
    st2 = RandomForestClassifier(n_estimators=400, min_samples_leaf=4,
                                 class_weight="balanced_subsample",
                                 random_state=seed, n_jobs=1).fit(M.Xtr[sel],
                                                                  M.ytr[sel])
    def comb(g, X):
        s = np.zeros(len(X))
        m = g > thr
        if m.any():
            s[m] = 0.5 + 0.5 * st2.predict_proba(X[m])[:, 1]
        s[~m] = 0.5 * (g[~m] - g.min()) / max(thr - g.min(), 1e-9)
        return s
    return comb(g_va, M.Xva), comb(g_te, M.Xte)


# ---- нейросетевые модели (опционально) --------------------------------------

def _torch_setup(seed: int):
    torch.manual_seed(seed)
    torch.set_num_threads(1)


class _MLPAE(nn.Module):
    def __init__(self, d, h, z):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Linear(h, z))
        self.dec = nn.Sequential(nn.Linear(z, h), nn.ReLU(), nn.Linear(h, d))

    def forward(self, x):
        return self.dec(self.enc(x))


class _VAE(nn.Module):
    def __init__(self, d, h, z):
        super().__init__()
        self.b = nn.Sequential(nn.Linear(d, h), nn.ReLU())
        self.mu, self.lv = nn.Linear(h, z), nn.Linear(h, z)
        self.dec = nn.Sequential(nn.Linear(z, h), nn.ReLU(), nn.Linear(h, d))

    def forward(self, x):
        e = self.b(x)
        mu, lv = self.mu(e), self.lv(e).clamp(-8, 8)
        zs = mu + torch.randn_like(mu) * (0.5 * lv).exp()
        return self.dec(zs), mu, lv


def _fit_dense(model, Xb, epochs, lr, loss_fn, seed):
    _torch_setup(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    X = torch.tensor(Xb, dtype=torch.float32)
    n = len(X)
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, 256):
            xb = X[perm[i:i + 256]]
            opt.zero_grad()
            loss_fn(model, xb).backward()
            opt.step()
    model.eval()
    return model


def m_autoencoder(M: Matrices, cfg: TrainingConfig, seed: int):
    d = M.Xtr.shape[1]
    net = _MLPAE(d, cfg.ae_hidden, cfg.ae_latent)
    # ПРОСТОЙ MSE. Focal loss убран: он требует метки (см. docstring, п.7).
    net = _fit_dense(net, M.Xtr[M.ytr == 0], cfg.ae_epochs, 1e-3,
                     lambda m, x: ((m(x) - x) ** 2).mean(), seed)
    with torch.no_grad():
        f = lambda X: ((net(torch.tensor(X, dtype=torch.float32))
                        - torch.tensor(X, dtype=torch.float32)) ** 2
                       ).mean(1).numpy()
        return f(M.Xva), f(M.Xte)


def m_vae(M: Matrices, cfg: TrainingConfig, seed: int):
    d = M.Xtr.shape[1]
    net = _VAE(d, cfg.ae_hidden, cfg.ae_latent)

    def loss(m, x):
        xh, mu, lv = m(x)
        rec = ((xh - x) ** 2).sum(1).mean()
        kl = (-0.5 * (1 + lv - mu ** 2 - lv.exp()).sum(1)).mean()
        return rec + 0.1 * kl

    net = _fit_dense(net, M.Xtr[M.ytr == 0], cfg.ae_epochs, 1e-3, loss, seed)
    with torch.no_grad():
        def f(X):
            x = torch.tensor(X, dtype=torch.float32)
            xh, mu, lv = net(x)
            rec = ((xh - x) ** 2).sum(1)
            kl = -0.5 * (1 + lv - mu ** 2 - lv.exp()).sum(1)
            return (rec + 0.1 * kl).numpy()
        return f(M.Xva), f(M.Xte)


# ---- последовательные модели -------------------------------------------------

def build_sequences(df: pd.DataFrame, X: np.ndarray, L: int
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Окна длины L строго внутри одного узла и БЕЗ разрывов по времени.

    Возвращает (tensor, индексы строк-концов, метки концов). Границы блоков
    сплита разрывают последовательность, поэтому склейки через purge-зазор
    не происходит.
    """
    seqs, ends = [], []
    order = np.argsort(df["t"].to_numpy(), kind="stable")
    for nid, g in df.assign(_pos=np.arange(len(df))).groupby("nid", sort=False):
        g = g.sort_values("t")
        pos = g["_pos"].to_numpy()
        t = g["t"].to_numpy()
        brk = np.flatnonzero(np.diff(t) != 1) + 1
        for seg in np.split(np.arange(len(t)), brk):
            if len(seg) < L:
                continue
            for j in range(L - 1, len(seg)):
                w = pos[seg[j - L + 1:j + 1]]
                seqs.append(X[w])
                ends.append(pos[seg[j]])
    if not seqs:
        return np.zeros((0, L, X.shape[1])), np.zeros(0, int), np.zeros(0, int)
    ends = np.asarray(ends, int)
    return np.stack(seqs), ends, df["label"].to_numpy(int)[ends]


class _SeqAE(nn.Module):
    def __init__(self, d, h, mode="lstm"):
        super().__init__()
        self.mode = mode
        self.enc = nn.LSTM(d, h, batch_first=True)
        self.att = nn.Linear(h, 1)
        self.dec = nn.Linear(h, d)

    def forward(self, x):
        o, (hn, _) = self.enc(x)
        if self.mode == "attn":
            w = torch.softmax(self.att(o), dim=1)
            ctx = (w * o).sum(1)
        else:
            ctx = hn[-1]
        rep = ctx.unsqueeze(1).expand(-1, x.size(1), -1)
        return self.dec(rep)


def _seq_model(mode: str):
    def f(M: Matrices, cfg: TrainingConfig, seed: int):
        L = cfg.seq_len
        Str, Etr, Ytr = build_sequences(M.tr, M.Xtr, L)
        Sva, Eva, _ = build_sequences(M.va, M.Xva, L)
        Ste, Ete, _ = build_sequences(M.te, M.Xte, L)
        if len(Str) < 200 or len(Sva) == 0 or len(Ste) == 0:
            return np.full(len(M.Xva), np.nan), np.full(len(M.Xte), np.nan)
        _torch_setup(seed)
        net = _SeqAE(M.Xtr.shape[1], cfg.ae_hidden, mode)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        Xb = torch.tensor(Str[Ytr == 0], dtype=torch.float32)
        g = torch.Generator().manual_seed(seed)
        for _ in range(cfg.seq_epochs):
            perm = torch.randperm(len(Xb), generator=g)
            for i in range(0, len(Xb), 128):
                xb = Xb[perm[i:i + 128]]
                opt.zero_grad()
                ((net(xb) - xb) ** 2).mean().backward()
                opt.step()
        net.eval()

        def sc(S, E, n):
            out = np.full(n, np.nan)
            with torch.no_grad():
                x = torch.tensor(S, dtype=torch.float32)
                e = ((net(x) - x) ** 2).mean(dim=(1, 2)).numpy()
            out[E] = e
            return out

        return sc(Sva, Eva, len(M.Xva)), sc(Ste, Ete, len(M.Xte))
    return f


# ---- ансамбли ---------------------------------------------------------------

def _group_holdout(g: np.ndarray, seed: int, frac_b: float = 0.4):
    """Делит train на A (базовые модели) и B (мета-модель) по ГРУППАМ."""
    uniq = np.unique(g)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(uniq)
    nb = max(1, int(round(frac_b * len(uniq))))
    b = set(perm[:nb].tolist())
    mask_b = np.array([x in b for x in g])
    return ~mask_b, mask_b


def m_meta_ensemble(M: Matrices, cfg: TrainingConfig, seed: int):
    """Стэкинг без утечки: базовые на A, мета на B, оценка на val/test."""
    a, b = _group_holdout(M.gtr, seed)
    if a.sum() < 200 or b.sum() < 100 or len(np.unique(M.ytr[b])) < 2:
        return m_random_forest(M, cfg, seed)

    MA = Matrices(cols=M.cols, Xtr=M.Xtr[a], Xva=M.Xtr[b], Xte=M.Xte,
                  ytr=M.ytr[a], yva=M.ytr[b], yte=M.yte, gtr=M.gtr[a],
                  tr=M.tr[a], va=M.tr[b], te=M.te)
    bases: List[Callable] = [m_isolation_forest, m_pca_recon, m_random_forest,
                             m_hist_gb]
    Zb, Zte = [], []
    for f in bases:
        s_b, s_te = f(MA, cfg, seed)
        Zb.append(np.nan_to_num(s_b))
        Zte.append(np.nan_to_num(s_te))
    Zb, Zte = np.column_stack(Zb), np.column_stack(Zte)

    # признаки для val считаем теми же базовыми моделями, обученными на A
    Zva = []
    MB = Matrices(cols=M.cols, Xtr=M.Xtr[a], Xva=M.Xva, Xte=M.Xte,
                  ytr=M.ytr[a], yva=M.yva, yte=M.yte, gtr=M.gtr[a],
                  tr=M.tr[a], va=M.va, te=M.te)
    for f in bases:
        s_va, _ = f(MB, cfg, seed)
        Zva.append(np.nan_to_num(s_va))
    Zva = np.column_stack(Zva)

    sc = StandardScaler().fit(Zb)
    meta = LogisticRegression(max_iter=2000, class_weight="balanced").fit(
        sc.transform(Zb), M.ytr[b])
    return (meta.predict_proba(sc.transform(Zva))[:, 1],
            meta.predict_proba(sc.transform(Zte))[:, 1])


def model_zoo(cfg: TrainingConfig) -> Dict[str, Tuple[str, Callable]]:
    z: Dict[str, Tuple[str, Callable]] = {
        "rule_baseline": ("rules", m_rule_baseline),
        "pca_recon": ("unsup", m_pca_recon),
        "isolation_forest": ("unsup", m_isolation_forest),
        "ocsvm": ("unsup", m_ocsvm),
        "random_forest": ("sup", m_random_forest),
        "hist_gb": ("sup", m_hist_gb),
        "cascade": ("hybrid", m_cascade),
        "meta_ensemble": ("stack", m_meta_ensemble),
    }
    if cfg.quick:
        for k in ("ocsvm", "meta_ensemble"):
            z.pop(k, None)
        return z
    if _HAS_TORCH and cfg.use_torch:
        z["autoencoder"] = ("unsup", m_autoencoder)
        z["vae"] = ("unsup", m_vae)
        z["lstm_ae"] = ("seq", _seq_model("lstm"))
        z["attn_ae"] = ("seq", _seq_model("attn"))
    return z


def run_zoo(M: Matrices, cfg: TrainingConfig, seed: int,
            only: Optional[Sequence[str]] = None) -> pd.DataFrame:
    rows = []
    for name, (fam, fn) in model_zoo(cfg).items():
        if only and name not in only:
            continue
        t0 = time.time()
        try:
            s_va, s_te = fn(M, cfg, seed)
        except Exception as e:                                  # pragma: no cover
            rows.append({"model": name, "family": fam, "error": repr(e)})
            continue
        r = score_row(name, fam, M, np.asarray(s_va, float),
                      np.asarray(s_te, float), cfg)
        r["fit_seconds"] = round(time.time() - t0, 2)
        rows.append(r)
    return pd.DataFrame(rows)


# =============================================================================
#  Диагностика, аблации, кривые
# =============================================================================

def gate_null_test(cfg: TrainingConfig, cols: Sequence[str],
                   allow_untrusted: bool) -> Dict:
    """Шлюз: без прохождения нулевого теста обучение не начинается."""
    assert_mask_is_identity_at_zero(np.random.default_rng(cfg.seed))
    try:
        rep = null_test(cfg.sim_config(), cols)
        rep["passed"] = True
        return rep
    except AssertionError as e:
        if not allow_untrusted:
            raise
        return {"passed": False, "reason": str(e)}


def feature_set_ablation(df: pd.DataFrame, cfg: TrainingConfig,
                         seed: int) -> pd.DataFrame:
    rng = RngBook(cfg.seed)["split"]
    tr, va, te = blocked_split(df, cfg.sim_config(), rng)
    out = []
    for name, base in FEATURE_SETS.items():
        cols = resolve_feature_columns(df, base)
        if not cols:
            continue
        M = prepare_matrices(tr, va, te, cols)
        r = run_zoo(M, cfg, seed,
                    only=("rule_baseline", "random_forest", "isolation_forest"))
        r["feature_set"] = name
        r["n_features"] = len(cols)
        out.append(r)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def temporal_ablation(df: pd.DataFrame, cfg: TrainingConfig,
                      seed: int) -> pd.DataFrame:
    """Вклад скользящих/зональных преобразований отдельно от «сырых» признаков."""
    rng = RngBook(cfg.seed)["split"]
    tr, va, te = blocked_split(df, cfg.sim_config(), rng)
    base = FEATURE_SETS[cfg.feature_set]
    variants = {
        "raw_only": [c for c in resolve_feature_columns(df, base)
                     if not c.endswith(DERIVED_SUFFIXES)],
        "no_zone": [c for c in resolve_feature_columns(df, base)
                    if not c.endswith("__dev_zone")],
        "full": resolve_feature_columns(df, base),
    }
    out = []
    for name, cols in variants.items():
        if not cols:
            continue
        M = prepare_matrices(tr, va, te, cols)
        r = run_zoo(M, cfg, seed, only=("random_forest", "rule_baseline"))
        r["variant"] = name
        r["n_features"] = len(cols)
        out.append(r)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def detectability_curve(cfg: TrainingConfig, rates=(0.0, 0.05, 0.15, 0.4, 1.0, 2.5)
                        ) -> pd.DataFrame:
    """Кривая recall(h) С ПРОХОДОМ ЧЕРЕЗ НОЛЬ.

    При h=0 recall обязан совпасть с бюджетом ложных тревог; если он выше,
    в генераторе осталась подпись, не связанная со скрытой активностью.
    """
    base = FEATURE_SETS[cfg.feature_set]
    rows = []
    for h in rates:
        sc = cfg.sim_config()
        sc.seed = cfg.seed + 4000 + int(h * 1000)
        df, _, _ = build_dataset(sc, episodes_kw=dict(
            n_compromise=10, n_normal=6, n_ext=4, force_hidden=float(h),
            soph=Sophistication(cfg.sophistication)))
        cols = resolve_feature_columns(df, base)
        tr, va, te = blocked_split(df, sc, RngBook(sc.seed)["split"])
        if te["label"].sum() < 10 or tr["label"].sum() < 10:
            continue
        M = prepare_matrices(tr, va, te, cols)
        r = run_zoo(M, cfg, sc.seed, only=("random_forest", "rule_baseline"))
        r["hidden_rate"] = h
        rows.append(r)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def sophistication_sweep(cfg: TrainingConfig) -> pd.DataFrame:
    """Головная цифра — ADAPTIVE. Среднее по уровням не публикуется."""
    base = FEATURE_SETS[cfg.feature_set]
    out = []
    for s in Sophistication:
        sc = cfg.sim_config()
        sc.seed = cfg.seed + 7000 + hash(s.value) % 1000
        df, _, _ = build_dataset(sc, episodes_kw=dict(soph=s))
        cols = resolve_feature_columns(df, base)
        tr, va, te = blocked_split(df, sc, RngBook(sc.seed)["split"])
        if te["label"].sum() < 10:
            continue
        M = prepare_matrices(tr, va, te, cols)
        r = run_zoo(M, cfg, sc.seed,
                    only=("rule_baseline", "random_forest", "isolation_forest"))
        r["sophistication"] = s.value
        out.append(r)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def lomfo_ablation(cfg: TrainingConfig) -> pd.DataFrame:
    """Leave-one-masking-family-out: обучение без семейства, тест на нём."""
    sc = cfg.sim_config()
    df, _, _ = build_dataset(sc)
    cols = resolve_feature_columns(df, FEATURE_SETS[cfg.feature_set])
    rng = RngBook(sc.seed)["split"]
    out = []
    for fam in MASK_FAMILIES:
        tr, va, te = blocked_split(df, sc, rng)
        tr = tr[(tr["label"] == 0) | (tr["mask_family"] != fam)]
        va = va[(va["label"] == 0) | (va["mask_family"] != fam)]
        te = te[(te["label"] == 0) | (te["mask_family"] == fam)]
        if te["label"].sum() < 15 or tr["label"].sum() < 15:
            continue
        M = prepare_matrices(tr, va, te, cols)
        r = run_zoo(M, cfg, sc.seed, only=("random_forest", "rule_baseline"))
        r["holdout_family"] = fam
        out.append(r)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


# =============================================================================
#  Мультисид: перегенерация + бутстрэп + Холм
# =============================================================================

def holm(pvals: Dict[str, float], alpha: float = 0.05) -> pd.DataFrame:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, rows, rejected = len(items), [], True
    for i, (k, p) in enumerate(items):
        thr = alpha / (m - i)
        rejected = rejected and (p <= thr)
        rows.append({"comparison": k, "p": p, "holm_threshold": thr,
                     "significant": bool(rejected)})
    return pd.DataFrame(rows)


def bootstrap_paired(res: pd.DataFrame, a: str, b: str, metric: str,
                     n_boot: int = 10000, seed: int = 0) -> Dict[str, float]:
    pa = res[res.model == a].set_index("seed")[metric]
    pb = res[res.model == b].set_index("seed")[metric]
    common = pa.index.intersection(pb.index)
    d = (pa.loc[common] - pb.loc[common]).dropna().to_numpy(float)
    if len(d) < 3:
        return {"n": len(d), "p": np.nan}
    rng = np.random.default_rng(seed)
    bs = rng.choice(d, (n_boot, len(d)), replace=True).mean(1)
    p = 2 * min((bs <= 0).mean(), (bs >= 0).mean())
    return {"n": int(len(d)), "mean_diff": float(d.mean()),
            "sd": float(d.std(ddof=1)),
            "ci_lo": float(np.percentile(bs, 2.5)),
            "ci_hi": float(np.percentile(bs, 97.5)),
            "p": float(min(p, 1.0)),
            "mde_95": float(1.96 * d.std(ddof=1) / math.sqrt(len(d)))}


def multiseed(cfg: TrainingConfig, seeds: Sequence[int],
              metric: str = "episode_recall",
              comparisons: Sequence[Tuple[str, str]] = DEFAULT_COMPARISONS
              ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Каждый сид ПЕРЕГЕНЕРИРУЕТ сеть, трафик и эпизоды."""
    base = FEATURE_SETS[cfg.feature_set]
    per_seed = []
    for s in seeds:
        c = TrainingConfig(**{**asdict(cfg), "seed": int(s)})
        sc = c.sim_config()
        df, _, _ = build_dataset(sc, episodes_kw=dict(
            soph=Sophistication(cfg.sophistication)))
        cols = resolve_feature_columns(df, base)
        tr, va, te = blocked_split(df, sc, RngBook(sc.seed)["split"])
        M = prepare_matrices(tr, va, te, cols)
        r = run_zoo(M, c, int(s))
        r["seed"] = int(s)
        per_seed.append(r)
    res = pd.concat(per_seed, ignore_index=True)

    agg = (res.groupby("model")[["episode_recall", "auc", "ap",
                                 "median_ttd_intervals",
                                 "alerts_per_day_network",
                                 "incident_precision"]]
           .agg(["mean", "std", "count"]))
    agg.columns = ["_".join(c) for c in agg.columns]

    tests, pv = [], {}
    for a, b in comparisons:
        if a not in set(res.model) or b not in set(res.model):
            continue
        d = bootstrap_paired(res, a, b, metric, seed=cfg.seed)
        d.update(comparison=f"{a} vs {b}", metric=metric)
        tests.append(d)
        if d.get("p") == d.get("p"):
            pv[f"{a} vs {b}"] = d["p"]
    tests_df = pd.DataFrame(tests)
    if pv:
        tests_df = tests_df.merge(holm(pv), on="comparison", how="left")
    return res, agg.reset_index(), tests_df


# =============================================================================
#  Кэш датасета
# =============================================================================

def load_or_build(cfg: TrainingConfig) -> pd.DataFrame:
    sc = cfg.sim_config()
    key = hashlib.sha256(
        json.dumps({"seed": sc.seed, "days": sc.days,
                    "custody": sc.key_custody.value,
                    "soph": cfg.sophistication,
                    "fw": sc.feat_window, "zw": sc.z_window},
                   sort_keys=True).encode()).hexdigest()[:16]
    path = os.path.join(cfg.cache_dir, f"ss7_{key}.parquet")
    if cfg.use_cache and os.path.exists(path):
        return pd.read_parquet(path)
    df, _, _ = build_dataset(sc, episodes_kw=dict(
        soph=Sophistication(cfg.sophistication)))
    if cfg.use_cache:
        os.makedirs(cfg.cache_dir, exist_ok=True)
        try:
            df.to_parquet(path, index=False)
        except Exception:
            pass
    return df


# =============================================================================
#  main
# =============================================================================

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--days", type=float, default=6.0)
    ap.add_argument("--feature-set", default="OWN+INV", choices=list(FEATURE_SETS))
    ap.add_argument("--custody", default=KeyCustody.NODE_SOFTWARE.value,
                    choices=[k.value for k in KeyCustody])
    ap.add_argument("--soph", default=Sophistication.ADAPTIVE.value,
                    choices=[s.value for s in Sophistication])
    ap.add_argument("--alerts-per-day", type=float, default=20.0)
    ap.add_argument("--op-prevalence", type=float, default=1e-4)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--metric", default="episode_recall")
    ap.add_argument("--ablations", action="store_true")
    ap.add_argument("--lomfo", action="store_true")
    ap.add_argument("--curve", action="store_true")
    ap.add_argument("--soph-sweep", action="store_true")
    ap.add_argument("--no-torch", action="store_true")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out-dir", default="ss7_runs")
    ap.add_argument("--i-know-results-are-untrusted", action="store_true",
                    dest="untrusted",
                    help="продолжить при провале нулевого теста; "
                         "результаты помечаются как недоверенные")
    a = ap.parse_args(argv)

    # побочные эффекты — только здесь, не при импорте
    import warnings
    warnings.simplefilter("ignore", category=FutureWarning)
    pd.set_option("display.width", 170)
    pd.set_option("display.max_columns", 60)

    cfg = TrainingConfig(seed=a.seed, days=(1.5 if a.quick else a.days),
                         key_custody=a.custody, sophistication=a.soph,
                         feature_set=a.feature_set,
                         alerts_per_day=a.alerts_per_day,
                         op_prevalence=a.op_prevalence,
                         use_torch=not a.no_torch, use_cache=not a.no_cache,
                         quick=a.quick, out_dir=a.out_dir)
    run_dir = os.path.join(cfg.out_dir, f"{cfg.fingerprint()}_seed{cfg.seed}")
    os.makedirs(run_dir, exist_ok=True)
    save = lambda name, obj: (obj.to_csv(os.path.join(run_dir, name), index=False)
                              if isinstance(obj, pd.DataFrame) else None)

    print("=" * 84)
    print(f"SS7 training  |  fingerprint={cfg.fingerprint()}  out={run_dir}")
    print(f"torch={'да' if (_HAS_TORCH and cfg.use_torch) else 'нет'}  "
          f"custody={cfg.key_custody}  soph={cfg.sophistication}")

    print("\n[1] Датасет")
    df = load_or_build(cfg)
    cols = resolve_feature_columns(df, FEATURE_SETS[cfg.feature_set])
    assert_no_leakage(cols)
    print(f"    окон={len(df)}  узлов={df['nid'].nunique()}  "
          f"доля аномальных окон={df['label'].mean():.5f}  признаков={len(cols)}")
    print(f"    guard: OK, все {len(cols)} столбцов выводятся из реестра")

    print("\n[2] Нулевой тест (шлюз)")
    nt = gate_null_test(cfg, cols, a.untrusted)
    if nt.get("passed"):
        print(f"    AUC={nt['auc']:.3f} CI95={nt['auc_ci']} "
              f"p_energy={nt['p_energy']:.3f} → подписи при h=0 нет")
    else:
        print("    ПРОВАЛЕН, результаты НЕДОВЕРЕННЫЕ:", nt.get("reason", "")[:200])
    with open(os.path.join(run_dir, "null_test.json"), "w") as f:
        json.dump({k: v for k, v in nt.items() if k != "per_feature"}, f,
                  ensure_ascii=False, indent=2, default=float)

    print("\n[3] Блочный сплит")
    sc = cfg.sim_config()
    tr, va, te = blocked_split(df, sc, RngBook(sc.seed)["split"])
    print(f"    train={len(tr)} val={len(va)} test={len(te)}  "
          f"purge={sc.feat_window + sc.z_window} интервалов  "
          f"эпизодов в тесте={te.loc[te.episode_id >= 0, 'episode_id'].nunique()}")
    M = prepare_matrices(tr, va, te, cols)

    print("\n[4] Зоопарк моделей (порог из val по бюджету тревог)")
    res = run_zoo(M, cfg, cfg.seed)
    show = ["model", "family", "auc", "ap", "episode_recall",
            "median_ttd_intervals", "incident_precision",
            "false_incidents_per_day", "window_fpr",
            "precision_at_op_prevalence", "fit_seconds"]
    print(res[[c for c in show if c in res.columns]].to_string(index=False))
    res["untrusted"] = not nt.get("passed", False)
    save("models.csv", res)

    if a.ablations:
        print("\n[5] Аблация наборов признаков")
        fa = feature_set_ablation(df, cfg, cfg.seed)
        print(fa[["feature_set", "model", "n_features", "episode_recall",
                  "auc"]].to_string(index=False))
        save("ablation_feature_sets.csv", fa)

        print("\n[6] Аблация временных и зональных преобразований")
        ta = temporal_ablation(df, cfg, cfg.seed)
        print(ta[["variant", "model", "n_features", "episode_recall",
                  "auc"]].to_string(index=False))
        save("ablation_temporal.csv", ta)

    if a.lomfo:
        print("\n[7] Leave-one-masking-family-out")
        lo = lomfo_ablation(cfg)
        if not lo.empty:
            print(lo[["holdout_family", "model", "episode_recall",
                      "auc"]].to_string(index=False))
        save("lomfo.csv", lo)

    if a.curve:
        print("\n[8] Кривая обнаружимости с проходом через ноль")
        cu = detectability_curve(cfg)
        if not cu.empty:
            print(cu[["hidden_rate", "model", "episode_recall", "window_fpr",
                      "auc"]].to_string(index=False))
            z = cu[(cu.hidden_rate == 0.0) & (cu.model == "random_forest")]
            if not z.empty:
                print(f"    при h=0: recall={z.iloc[0]['episode_recall']:.3f} "
                      f"(должен быть на уровне бюджета тревог)")
        save("detectability_curve.csv", cu)

    if a.soph_sweep:
        print("\n[9] Срез по изощрённости противника (головная цифра — adaptive)")
        sw = sophistication_sweep(cfg)
        if not sw.empty:
            print(sw[["sophistication", "model", "episode_recall",
                      "auc"]].to_string(index=False))
        save("sophistication_sweep.csv", sw)

    if a.seeds:
        print(f"\n[10] Мультисид с перегенерацией: {a.seeds}")
        ms, agg, tests = multiseed(cfg, a.seeds, metric=a.metric)
        print(agg.to_string(index=False))
        if not tests.empty:
            print("\n     Бутстрэп по сидам + поправка Холма:")
            print(tests[["comparison", "n", "mean_diff", "ci_lo", "ci_hi", "p",
                         "holm_threshold", "significant",
                         "mde_95"]].to_string(index=False))
        save("multiseed_raw.csv", ms)
        save("multiseed_agg.csv", agg)
        save("multiseed_tests.csv", tests)

    manifest = {
        "config": asdict(cfg), "fingerprint": cfg.fingerprint(),
        "n_windows": int(len(df)), "feature_columns": cols,
        "null_test_passed": bool(nt.get("passed", False)),
        "trusted": bool(nt.get("passed", False)),
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                     "pandas": pd.__version__,
                     "torch": (torch.__version__ if _HAS_TORCH else None)},
        "preregistered_comparisons": [list(c) for c in DEFAULT_COMPARISONS],
    }
    with open(os.path.join(run_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"\nГотово. Артефакты: {run_dir}")
    if not nt.get("passed", False):
        print("ВНИМАНИЕ: нулевой тест не пройден — метрики интерпретировать нельзя.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
