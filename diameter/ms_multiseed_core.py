"""
ms_multiseed_core.py — общее ядро мультисид-прогонов для SS7 и Diameter.
=========================================================================
Повторяет логику SIP-проекта (train_ticd.py + run_seeds_ticd.py), но
переиспользует уже существующие пайплайны ss7_training_v7.py и
diameter_training_v1.py через import (сами файлы не изменяются).

Что делает:
  * прогон одного сида: обучение всего зоопарка моделей, метрики,
    per-attack recall, recall@1%FPR, аблации → pkl + csv;
  * агрегация по N сидам: mean±std, таблица для статьи;
  * критерий Уилкоксона (парный, знаково-ранговый) для заявленных
    сравнений + поправка Холма на множественность.

Изменчивость по сидам:
  * по умолчанию сид меняет train/val/test-разбиение и инициализацию
    моделей (как в run_seeds_ticd.py для SIP);
  * с флагом --regen дополнительно ПЕРЕГЕНЕРИРУЕТСЯ датасет тем же сидом
    (независимые реализации сети/трафика) — статистически сильнее.
"""

import os
import json
import pickle
import importlib
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Callable

import torch
from scipy.stats import wilcoxon
from sklearn.ensemble import (IsolationForest, RandomForestClassifier,
                              GradientBoostingClassifier)
from sklearn.svm import OneClassSVM
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             f1_score, confusion_matrix)

import warnings
warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════
#  Спецификация протокола
# ══════════════════════════════════════════════════════════

DEFAULT_COMPARISONS = [
    # (модель A, модель B, метрика) — проверяем A > B
    ("Cascade",           "Random Forest",      "compromise_recall"),
    ("Cascade",           "Cascade-Stage1",     "compromise_recall"),
    ("Cascade",           "Autoencoder",        "recall_at_1pct_fpr"),
    ("Meta-Ensemble",     "Random Forest",      "f1"),
    ("Random Forest",     "RF (no MS)",         "f1"),
    ("Random Forest",     "RF (no MS)",         "compromise_recall"),
    ("Random Forest",     "RF (no temporal)",   "compromise_recall"),
    ("Random Forest",     "Autoencoder",        "auc_pr"),
    ("VAE",               "Autoencoder",        "auc_pr"),
]

BASE_METRICS = ["auc_roc", "auc_pr", "f1", "fpr",
                "compromise_recall", "recall_at_1pct_fpr",
                "episode_recall", "false_incidents_per_day"]


@dataclass
class ProtocolSpec:
    name: str                        # "ss7" / "diameter"
    train_module: str                # "ss7_training_v7"
    sim_module: str                  # "ss7_simulator_v7"
    data_path: str                   # дефолтный csv
    compromise_attack: str           # "slave_compromise" / "node_compromise"
    attack_types: List[str] = field(default_factory=list)
    temporal_features: List[str] = field(default_factory=list)
    base_out: str = "seed_runs"
    comparisons: List[Tuple[str, str, str]] = \
        field(default_factory=lambda: list(DEFAULT_COMPARISONS))
    sim_config_overrides: Dict = field(default_factory=dict)
    calibration_provenance: Dict = field(default_factory=dict)

    @property
    def metrics(self) -> List[str]:
        return BASE_METRICS + [f"rec_{a}" for a in self.attack_types]


# ══════════════════════════════════════════════════════════
#  Метрики (единые с train_ticd.py)
# ══════════════════════════════════════════════════════════

def _set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def recall_at_fpr(scores, y, target_fpr=0.01):
    """Полнота при фиксированном FPR — честная метрика для статьи."""
    scores = np.nan_to_num(np.asarray(scores, float))
    y = np.asarray(y)
    if (y == 0).sum() == 0:
        return np.nan, np.nan
    thr = np.percentile(scores[y == 0], 100 * (1 - target_fpr))
    yp = (scores >= thr).astype(int)
    tp = ((yp == 1) & (y == 1)).sum()
    fn = ((yp == 0) & (y == 1)).sum()
    return 100.0 * tp / max(1, tp + fn), thr


def by_attack(scores, attack_types, thr, attack_list):
    """Полнота по типам атак при пороге thr, %."""
    yp = (np.nan_to_num(np.asarray(scores, float)) >= thr).astype(int)
    at = np.asarray(attack_types)
    out = {}
    for a in attack_list:
        m = (at == a)
        if m.sum() > 0:
            out[a] = 100.0 * yp[m].sum() / m.sum()
    return out


def normalize_score01(scores):
    """Robust [0, 1] normalization for heterogeneous stacking scores."""
    arr = np.asarray(scores, dtype=float).copy()
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros_like(arr, dtype=float)
    fill = float(np.median(arr[finite]))
    arr = np.nan_to_num(arr, nan=fill, posinf=fill, neginf=fill)
    p1, p99 = np.percentile(arr, [1, 99])
    scale = float(p99 - p1)
    if not np.isfinite(scale) or scale <= 1e-12:
        return np.zeros_like(arr, dtype=float)
    return np.clip((arr - p1) / scale, 0.0, 1.0)


def normalize_score01_from_reference(reference, scores):
    """Scale scores with quantiles fitted only on a reference partition."""
    ref = np.asarray(reference, dtype=float).copy()
    arr = np.asarray(scores, dtype=float).copy()
    finite = np.isfinite(ref)
    if not finite.any():
        return np.zeros_like(arr, dtype=float)
    fill = float(np.median(ref[finite]))
    ref = np.nan_to_num(ref, nan=fill, posinf=fill, neginf=fill)
    arr = np.nan_to_num(arr, nan=fill, posinf=fill, neginf=fill)
    p1, p99 = np.percentile(ref, [1, 99])
    scale = float(p99 - p1)
    if not np.isfinite(scale) or scale <= 1e-12:
        return np.zeros_like(arr, dtype=float)
    return np.clip((arr - p1) / scale, 0.0, 1.0)


def count_false_incidents(test_df, predictions, labels):
    """Count contiguous false-positive runs per node as incidents."""
    yp = np.asarray(predictions, dtype=int)
    y = np.asarray(labels, dtype=int)
    if len(test_df) != len(yp) or len(yp) != len(y):
        raise ValueError("test_df, predictions and labels must have equal length")
    if "node_id" not in test_df.columns or "interval" not in test_df.columns:
        return int(np.sum((yp == 1) & (y == 0)))

    tmp = test_df[["node_id", "interval"]].copy()
    tmp["_fp"] = ((yp == 1) & (y == 0)).astype(int)
    incidents = 0
    for _, part in tmp.sort_values(["node_id", "interval"]).groupby("node_id"):
        prev_interval = None
        prev_fp = 0
        for row in part.itertuples(index=False):
            cur_interval = int(row.interval)
            cur_fp = int(row._fp)
            contiguous = (
                prev_interval is not None and cur_interval == prev_interval + 1
            )
            if cur_fp and (not prev_fp or not contiguous):
                incidents += 1
            prev_interval = cur_interval
            prev_fp = cur_fp
    return int(incidents)


def build_row(name, scores, y, attack_types, thr, spec, family):
    scores = np.nan_to_num(np.asarray(scores, float))
    y = np.asarray(y)
    yp = (scores >= thr).astype(int)
    multi = len(np.unique(y)) > 1
    tn, fp, fn, tp = confusion_matrix(y, yp, labels=[0, 1]).ravel()
    r1, _ = recall_at_fpr(scores, y, 0.01)
    row = dict(
        name=name, family=family, threshold=float(thr),
        auc_roc=roc_auc_score(y, scores) if multi else np.nan,
        auc_pr=average_precision_score(y, scores) if multi else np.nan,
        f1=f1_score(y, yp, zero_division=0),
        fpr=fp / (fp + tn) if (fp + tn) else 0.0,
        tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn),
        recall_at_1pct_fpr=r1,
    )
    rec = by_attack(scores, attack_types, thr, spec.attack_types)
    for a in spec.attack_types:
        row[f"rec_{a}"] = rec.get(a, np.nan)
    row["compromise_recall"] = rec.get(spec.compromise_attack, np.nan)
    return row


# ══════════════════════════════════════════════════════════
#  Перегенерация датасета тем же сидом (опционально)
# ══════════════════════════════════════════════════════════

def regen_dataset(spec: ProtocolSpec, seed: int, out_csv: str) -> str:
    """
    Запускает симулятор с заданным сидом (подменяя SimulationConfig так,
    чтобы seed принудительно = seed) и сохраняет результат в out_csv.
    """
    if os.path.exists(out_csv):
        print(f"  [regen] уже есть: {out_csv}")
        return out_csv
    m = importlib.import_module(spec.sim_module)
    orig_cfg = m.SimulationConfig

    def patched(*args, **kwargs):
        kwargs["seed"] = seed
        kwargs.update(spec.sim_config_overrides)
        return orig_cfg(*args, **kwargs)

    m.SimulationConfig = patched
    try:
        print(f"  [regen] симуляция {spec.name}, seed={seed} ...")
        df = m.main()
    finally:
        m.SimulationConfig = orig_cfg
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    df.to_csv(out_csv, index=False)
    manifest = {
        "protocol": spec.name,
        "seed": int(seed),
        "sim_config_overrides": spec.sim_config_overrides,
        "calibration_provenance": spec.calibration_provenance,
        "shape": [int(df.shape[0]), int(df.shape[1])],
        "anomaly_rate": float(df["is_anomaly"].mean())
            if "is_anomaly" in df.columns else None,
        "attack_distribution": (
            {str(k): int(v) for k, v in
             df["attack_type"].value_counts().to_dict().items()}
            if "attack_type" in df.columns else {}
        ),
    }
    with open(out_csv + ".manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
    print(f"  [regen] сохранено: {out_csv}  shape={df.shape}")
    return out_csv


# ══════════════════════════════════════════════════════════
#  Прогон одного сида
# ══════════════════════════════════════════════════════════

def run_one_seed(spec: ProtocolSpec, seed: int, csv: Optional[str] = None,
                 out: Optional[str] = None, include_sequential: bool = True,
                 include_ensemble: bool = False, include_svm: bool = True,
                 include_neural: bool = True
                 ) -> pd.DataFrame:
    mod = importlib.import_module(spec.train_module)

    cfg = mod.TrainingConfig()
    cfg.seed = seed
    cfg.data_path = csv or spec.data_path
    out = out or os.path.join(spec.base_out, f"seed_{seed}")
    cfg.results_dir = out
    if hasattr(cfg, "plots_dir"):
        cfg.plots_dir = os.path.join(out, "plots")
    os.makedirs(out, exist_ok=True)
    _set_seed(seed)

    # ── данные ──
    pipe = mod.DataPipeline(cfg)
    train_df, val_df, test_df = pipe.load_and_prepare()

    X_train, y_train = pipe.get_features_labels(train_df, fit=True)
    X_val, y_val = pipe.get_features_labels(val_df)
    X_test, y_test = pipe.get_features_labels(test_df)
    Xtr_c, ytr_c = pipe.get_compromise_features(train_df, fit=True)
    Xte_c, _ = pipe.get_compromise_features(test_df)
    Xva_c, _ = pipe.get_compromise_features(val_df)
    X_train_norm, _ = pipe.get_normal_data(train_df, fit=False)
    X_val_norm = X_val[y_val == 0]

    atk_test = test_df["attack_type"].values
    n_feat = X_train.shape[1]

    rows: List[Dict] = []
    model_scores: Dict[str, np.ndarray] = {}
    base_val, base_test = {}, {}   # leakage-safe meta stacking

    def record(name, s_test, thr, family="unsup",
               y=None, atypes=None, keep=True):
        primary_rows = y is None and atypes is None and len(s_test) == len(test_df)
        y = y_test if y is None else y
        atypes = atk_test if atypes is None else atypes
        if keep:
            model_scores[name] = np.asarray(s_test, float)
        row = build_row(name, s_test, y, atypes, thr, spec, family)

        if primary_rows and "attack_episode_id" in test_df.columns:
            yp = (np.nan_to_num(np.asarray(s_test, float)) >= thr).astype(int)
            epi = test_df["attack_episode_id"].astype(str).to_numpy()
            attack_mask = np.asarray(y_test) == 1
            episode_ids = sorted({
                e for e in epi[attack_mask]
                if e and e.lower() not in ("nan", "none")
            })
            detected = 0
            for eid in episode_ids:
                m = (epi == eid) & attack_mask
                if np.any(yp[m] == 1):
                    detected += 1
            row["n_test_episodes"] = int(len(episode_ids))
            row["n_detected_episodes"] = int(detected)
            row["episode_recall"] = (
                100.0 * detected / len(episode_ids)
                if episode_ids else np.nan
            )

            interval_s = 15.0
            if "time_s" in test_df.columns and "interval" in test_df.columns:
                ti = (test_df[["interval", "time_s"]]
                      .drop_duplicates().sort_values("interval"))
                di = ti["interval"].diff().to_numpy(float)
                dt = ti["time_s"].diff().to_numpy(float)
                good = np.isfinite(di) & np.isfinite(dt) & (di > 0)
                if np.any(good):
                    ratios = dt[good] / di[good]
                    ratios = ratios[(ratios > 0) & np.isfinite(ratios)]
                    if len(ratios):
                        interval_s = float(np.median(ratios))
            test_days = (
                test_df["interval"].nunique() * interval_s / 86400.0
                if "interval" in test_df.columns else np.nan
            )
            false_incidents = count_false_incidents(test_df, yp, y_test)
            row["false_incidents"] = int(false_incidents)
            row["false_incidents_per_day"] = (
                float(false_incidents) / test_days
                if np.isfinite(test_days) and test_days > 0 else np.nan
            )
            row["false_positive_windows_per_day"] = (
                float(row["fp"]) / test_days
                if np.isfinite(test_days) and test_days > 0 else np.nan
            )
        else:
            row["n_test_episodes"] = np.nan
            row["n_detected_episodes"] = np.nan
            row["episode_recall"] = np.nan
            row["false_incidents"] = np.nan
            row["false_incidents_per_day"] = np.nan
            row["false_positive_windows_per_day"] = np.nan

        rows.append(row)
        ep = row.get("episode_recall", np.nan)
        ep_text = f"{ep:.1f}%" if np.isfinite(ep) else "—"
        print(f"  [{family:9s}] {name:22s} AUC-PR={row['auc_pr']:.3f} "
              f"F1={row['f1']:.3f} FPR={row['fpr']:.4f} "
              f"comp={row['compromise_recall']:.1f}% "
              f"R@1%FPR={row['recall_at_1pct_fpr']:.1f}% "
              f"episode={ep_text}")
        return row

    # Label-permutation null: a classifier trained on shuffled labels
    # should not generalise to the untouched test labels. This is a sanity
    # check for accidental label/row leakage, not a substitute for a full
    # counterfactual simulator null.
    null_aucs = []
    rng_null = np.random.default_rng(seed + 900001)
    for j in range(2):
        y_perm = rng_null.permutation(y_train)
        null_rf = RandomForestClassifier(
            n_estimators=40,
            random_state=seed + 1000 + j,
            class_weight="balanced",
            n_jobs=-1,
        )
        null_rf.fit(X_train, y_perm)
        null_score = null_rf.predict_proba(X_test)[:, 1]
        if len(np.unique(y_test)) > 1:
            null_aucs.append(float(roc_auc_score(y_test, null_score)))
    null_auc_mean = float(np.mean(null_aucs)) if null_aucs else np.nan
    null_passed = bool(
        np.isfinite(null_auc_mean) and 0.35 <= null_auc_mean <= 0.65
    )
    print(
        f"  permutation-null AUCs={null_aucs}, "
        f"mean={null_auc_mean:.4f}, passed={null_passed}"
    )

    pct = cfg.threshold_percentile
    print(f"\n--- [{spec.name}] seed={seed}: обучение моделей "
          f"({n_feat} признаков) ---")

    # ── 1. Isolation Forest ──
    iso = IsolationForest(n_estimators=cfg.iso_forest_trees,
                          contamination=cfg.iso_forest_contamination,
                          random_state=seed).fit(X_train_norm)
    iso_v, iso_t = -iso.score_samples(X_val), -iso.score_samples(X_test)
    th, _ = mod.find_threshold(iso_v, y_val, pct)
    record("Isolation Forest", iso_t, th)
    base_val["iso"] = iso_v; base_test["iso"] = iso_t

    # ── 2. One-Class SVM ──
    if include_svm:
        svm = OneClassSVM(nu=cfg.ocsvm_nu, kernel=cfg.ocsvm_kernel,
                          gamma="scale").fit(X_train_norm)
        svm_v, svm_t = -svm.score_samples(X_val), -svm.score_samples(X_test)
        th, _ = mod.find_threshold(svm_v, y_val, pct)
        record("One-Class SVM", svm_t, th)
        base_val["svm"] = svm_v
        base_test["svm"] = svm_t

    ae_hist, vae_hist = [], []

    # ── 3-4. Autoencoder / VAE ──
    if include_neural:
        ae = mod.Autoencoder(n_feat, cfg.ae_hidden_dims, cfg.ae_latent_dim)
        ae_tr = mod.AETrainer(
            ae, cfg.ae_lr, cfg.ae_epochs, cfg.ae_batch, patience=15
        )
        ae_tr.fit(X_train_norm, X_val_norm)
        ae_v = ae_tr.reconstruction_error(X_val)
        ae_t = ae_tr.reconstruction_error(X_test)
        th, _ = mod.find_threshold(ae_v, y_val, pct)
        record("Autoencoder", ae_t, th)
        base_val["ae"] = ae_v
        base_test["ae"] = ae_t
        ae_hist = getattr(ae_tr, "train_losses", [])

        vae = mod.VAE(n_feat, cfg.vae_hidden_dims, cfg.vae_latent_dim)
        vae_tr = mod.VAETrainer(
            vae, cfg.vae_lr, cfg.vae_epochs, cfg.vae_batch,
            kl_weight=cfg.vae_kl_weight, patience=15
        )
        vae_tr.fit(X_train_norm, X_val_norm)
        vae_v = vae_tr.anomaly_score(X_val)
        vae_t = vae_tr.anomaly_score(X_test)
        th, _ = mod.find_threshold(vae_v, y_val, pct)
        record("VAE", vae_t, th)
        base_val["vae"] = vae_v
        base_test["vae"] = vae_t
        vae_hist = getattr(vae_tr, "train_losses", [])

    # ── 5. Последовательные модели (свои y: метка последнего шага) ──
    if include_sequential and include_neural:
        seq_tr, _ = pipe.get_sequential_data(
            train_df[train_df[pipe.LABEL_COL] == 0], cfg.lstm_seq_len)
        if len(seq_tr) > 100:
            seq_v, seq_yv = pipe.get_sequential_data(val_df, cfg.lstm_seq_len)
            seq_t, seq_yt = pipe.get_sequential_data(test_df, cfg.lstm_seq_len)
            seq_vn = seq_v[seq_yv == 0]
            seq_atk = []
            for nid in test_df["node_id"].unique():
                a = test_df.loc[test_df["node_id"].values == nid,
                                "attack_type"].values
                for i in range(len(a) - cfg.lstm_seq_len + 1):
                    seq_atk.append(a[i + cfg.lstm_seq_len - 1])
            seq_atk = np.array(seq_atk)

            lstm = mod.LSTMAutoencoder(n_feat, cfg.lstm_hidden,
                                       cfg.lstm_layers, cfg.lstm_seq_len,
                                       dropout=0.2)
            lt = mod.AETrainer(lstm, cfg.lstm_lr, cfg.lstm_epochs,
                               cfg.lstm_batch, patience=15)
            lt.fit(seq_tr, seq_vn if len(seq_vn) else None)
            th, _ = mod.find_threshold(lt.reconstruction_error(seq_v),
                                       seq_yv, pct)
            record("LSTM-AE", lt.reconstruction_error(seq_t), th,
                   family="unsup-seq", y=seq_yt, atypes=seq_atk, keep=False)

            tat = mod.TemporalAttentionAE(input_dim=n_feat, hidden_dim=64,
                                          n_heads=4, n_layers=2,
                                          seq_len=cfg.lstm_seq_len,
                                          dropout=0.15)
            tt = mod.AETrainer(tat, 5e-4, 100, 256, patience=15)
            tt.fit(seq_tr, seq_vn if len(seq_vn) else None)
            th, _ = mod.find_threshold(tt.reconstruction_error(seq_v),
                                       seq_yv, pct)
            record("Temporal-Attention-AE", tt.reconstruction_error(seq_t),
                   th, family="unsup-seq", y=seq_yt, atypes=seq_atk,
                   keep=False)
        else:
            print("  [skip] недостаточно последовательностей для LSTM/TA-AE")

    # ── 6-7. Supervised ──
    rf = RandomForestClassifier(n_estimators=cfg.rf_trees, random_state=seed,
                                class_weight="balanced", n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_t = rf.predict_proba(X_test)[:, 1]
    record("Random Forest", rf_t, 0.5, family="oracle")
    base_val["rf"] = rf.predict_proba(X_val)[:, 1]; base_test["rf"] = rf_t

    gb = GradientBoostingClassifier(n_estimators=cfg.gb_estimators,
                                    learning_rate=cfg.gb_lr, max_depth=5,
                                    random_state=seed)
    gb.fit(X_train, y_train)
    gb_t = gb.predict_proba(X_test)[:, 1]
    record("Gradient Boosting", gb_t, 0.5, family="oracle")
    base_val["gb"] = gb.predict_proba(X_val)[:, 1]; base_test["gb"] = gb_t

    # ── 8. Каскадный детектор + его этапы ──
    casc = mod.CascadeCompromiseDetector(cfg, seed)
    casc.fit(X_train, y_train, Xtr_c, ytr_c)
    casc_t = casc.predict_proba(X_test, Xte_c)
    record("Cascade", casc_t, 0.5, family="oracle")
    record("Cascade-Stage1", casc.predict_proba_stage1(X_test), 0.5,
           family="oracle")
    record("Cascade-Stage2", casc.predict_proba_stage2(Xte_c), 0.5,
           family="oracle")
    base_val["casc"] = casc.predict_proba(X_val, Xva_c)
    base_test["casc"] = casc_t

    # ── 9. Специализированный ансамбль (медленно → по флагу) ──
    if include_ensemble:
        ens = mod.EnsembleDetector(mod.FEATURE_GROUPS, pipe.feature_cols, seed)
        ens.fit(X_train_norm)
        s_v, s_t = ens.score(X_val), ens.score(X_test)
        for k in sorted(s_t.keys()):
            th, _ = mod.find_threshold(s_v[k], y_val, pct)
            record(f"Ens:{k}", s_t[k], th)

    # ── 10. Мета-ансамбль ──
    keys = list(base_val.keys())
    meta_val = np.column_stack([
        normalize_score01_from_reference(base_val[k], base_val[k])
        for k in keys
    ])
    meta_te = np.column_stack([
        normalize_score01_from_reference(base_val[k], base_test[k])
        for k in keys
    ])
    meta = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                      max_depth=3, random_state=seed)
    meta.fit(meta_val, y_val)
    record("Meta-Ensemble", meta.predict_proba(meta_te)[:, 1], 0.5,
           family="stacked")

    # ── 11. Аблации (для проверки значимости вклада признаков) ──
    ms = [c for c in pipe.feature_cols if c in mod.MS_FEATURES]
    for tag, drop in [("RF (no MS)", ms),
                      ("RF (no temporal)", spec.temporal_features)]:
        keep_cols = [c for c in pipe.feature_cols if c not in drop]
        idx = [pipe.feature_cols.index(c) for c in keep_cols]
        if len(idx) == len(pipe.feature_cols):
            print(f"  [skip] {tag}: нечего удалять")
            continue
        m = RandomForestClassifier(n_estimators=cfg.rf_trees,
                                   random_state=seed,
                                   class_weight="balanced", n_jobs=-1)
        m.fit(X_train[:, idx], y_train)
        record(tag, m.predict_proba(X_test[:, idx])[:, 1], 0.5,
               family="ablation")

    # ── сохранение ──
    df_res = pd.DataFrame(rows)
    df_res.insert(0, "seed", seed)
    df_res.to_csv(os.path.join(out, f"metrics_seed{seed}.csv"), index=False)

    artifacts = dict(
        protocol=spec.name, seed=seed, results=rows,
        model_scores=model_scores, y_test=y_test,
        attack_types=atk_test, feature_cols=pipe.feature_cols,
        rf_importances=rf.feature_importances_,
        ae_train_hist=ae_hist,
        vae_train_hist=vae_hist,
        cascade_s1=casc.predict_proba_stage1(X_test),
        cascade_s2=casc.predict_proba_stage2(Xte_c),
        cascade_comb=casc_t,
    )
    with open(os.path.join(out, f"artifacts_seed{seed}.pkl"), "wb") as f:
        pickle.dump(artifacts, f)

    manifest = {
        "protocol": spec.name,
        "seed": int(seed),
        "data_path": str(cfg.data_path),
        "training_config": vars(cfg),
        "sim_config_overrides": spec.sim_config_overrides,
        "calibration_provenance": spec.calibration_provenance,
        "feature_cols": list(pipe.feature_cols),
        "split_report": getattr(pipe, "split_report", {}),
        "permutation_null_auc": null_aucs,
        "permutation_null_auc_mean": null_auc_mean,
        "permutation_null_passed": null_passed,
        "test_rows": int(len(test_df)),
        "test_anomalies": int(np.sum(y_test)),
        "meta_stacking": {
            "fit_partition": "validation",
            "base_score_normalization_reference": "validation",
            "test_labels_used_for_fit": False,
        },
        "cascade_stage2": {
            "enabled": bool(getattr(casc, "stage2_enabled", True)),
            "classes": list(getattr(casc, "stage2_classes", [0, 1])),
            "disabled_reason": getattr(
                casc, "stage2_disabled_reason", None
            ),
        },
        "models": [str(x) for x in df_res["name"].tolist()],
    }
    with open(os.path.join(out, f"manifest_seed{seed}.json"),
              "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2, default=str)

    print(f"  сохранено: {out}/artifacts_seed{seed}.pkl")
    return df_res


# ══════════════════════════════════════════════════════════
#  Уилкоксон + поправка Холма
# ══════════════════════════════════════════════════════════

def _holm(pvals):
    p = np.asarray(pvals, float)
    adj = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    idx = np.where(ok)[0]
    order = idx[np.argsort(p[idx])]
    m, run = len(order), 0.0
    for i, j in enumerate(order):
        run = max(run, (m - i) * p[j])
        adj[j] = min(1.0, run)
    return adj


def wilcoxon_tests(all_df: pd.DataFrame, comparisons) -> pd.DataFrame:
    out = []
    for a, b, metric in comparisons:
        if metric not in all_df.columns:
            continue
        va = all_df[all_df["name"] == a].set_index("seed")[metric].dropna()
        vb = all_df[all_df["name"] == b].set_index("seed")[metric].dropna()
        common = va.index.intersection(vb.index)
        if len(common) < 3:
            continue
        x, y = va.loc[common].to_numpy(), vb.loc[common].to_numpy()
        d = x - y
        n_eff = int((np.abs(d) > 1e-12).sum())
        p_two = p_one = np.nan
        if n_eff >= 3:
            try:
                _, p_two = wilcoxon(x, y, alternative="two-sided",
                                    zero_method="wilcox")
                _, p_one = wilcoxon(x, y, alternative="greater",
                                    zero_method="wilcox")
            except ValueError:
                pass
        out.append(dict(metric=metric, a=a, b=b, n_seeds=len(common),
                        n_nonzero=n_eff,
                        mean_a=float(np.mean(x)), mean_b=float(np.mean(y)),
                        median_a=float(np.median(x)),
                        median_b=float(np.median(y)),
                        delta_median=float(np.median(d)),
                        a_better=int((d > 0).sum()),
                        p_two_sided=p_two, p_one_sided_a_gt_b=p_one))
    df = pd.DataFrame(out)
    if not df.empty:
        df["p_two_holm"] = _holm(df["p_two_sided"].to_numpy())
        df["significant_005"] = df["p_two_holm"] < 0.05
    return df


# ══════════════════════════════════════════════════════════
#  Агрегация по сидам
# ══════════════════════════════════════════════════════════

def _fmt(mean, std, metric):
    if not np.isfinite(mean):
        return "—"
    d = 4 if metric in ("auc_roc", "auc_pr", "f1", "fpr") else 1
    return f"{mean:.{d}f} ± {std if np.isfinite(std) else 0:.{d}f}"


def run_seeds(spec: ProtocolSpec, seeds: List[int], csv=None, out=None,
              regen=False, include_sequential=True, include_ensemble=False,
              include_svm=True, include_neural=True):
    base = out or spec.base_out
    os.makedirs(base, exist_ok=True)
    frames = []

    for sd in seeds:
        print(f"\n{'='*66}\n[{spec.name}] seed {sd}\n{'='*66}")
        data_csv = csv or spec.data_path
        if regen:
            data_csv = regen_dataset(
                spec, sd, os.path.join(base, "data",
                                       f"{spec.name}_dataset_seed{sd}.csv"))
        frames.append(run_one_seed(
            spec, sd, csv=data_csv, out=os.path.join(base, f"seed_{sd}"),
            include_sequential=include_sequential,
            include_ensemble=include_ensemble, include_svm=include_svm,
            include_neural=include_neural))

    alld = pd.concat(frames, ignore_index=True)
    alld.to_csv(f"{base}/all_seed_metrics.csv", index=False)

    metrics = [m for m in spec.metrics if m in alld.columns]
    agg = alld.groupby(["name", "family"])[metrics].agg(["mean", "std",
                                                         "count"])
    agg.to_csv(f"{base}/aggregated.csv")

    table = pd.DataFrame(index=agg.index)
    for m in metrics:
        table[m] = [_fmt(agg.loc[i, (m, "mean")], agg.loc[i, (m, "std")], m)
                    for i in agg.index]
    table.to_csv(f"{base}/table_mean_std.csv")

    sig = wilcoxon_tests(alld, spec.comparisons)
    sig.to_csv(f"{base}/significance.csv", index=False)

    # ── печать сводки ──
    print(f"\n{'='*90}\n  [{spec.name}] СВОДКА по {len(seeds)} сидам "
          f"(mean ± std)\n{'='*90}")
    head = ["auc_roc", "auc_pr", "f1", "fpr", "compromise_recall",
            "recall_at_1pct_fpr"]
    print(f"{'Модель':<26s}" + "".join(f"{h:>22s}" for h in head if h in table))
    for i in table.index:
        print(f"{i[0]:<26s}" + "".join(f"{table.loc[i, h]:>22s}"
                                       for h in head if h in table))

    if not sig.empty:
        print(f"\n{'='*90}\n  Критерий Уилкоксона (парный, N={len(seeds)}), "
              f"поправка Холма\n{'='*90}")
        for _, r in sig.iterrows():
            star = "*" if r["significant_005"] else " "
            print(f" {star} [{r['metric']:<22s}] {r['a']} vs {r['b']}: "
                  f"{r['median_a']:.4f} vs {r['median_b']:.4f} "
                  f"(Δmed={r['delta_median']:+.4f}), "
                  f"{r['a_better']}/{r['n_seeds']} сидов, "
                  f"p={r['p_two_sided']:.4g}, p_holm={r['p_two_holm']:.4g}")

    with open(f"{base}/aggregated.pkl", "wb") as f:
        pickle.dump(dict(protocol=spec.name, all_metrics=alld, aggregated=agg,
                         table=table, significance=sig, seeds=seeds,
                         spec=dict(compromise_attack=spec.compromise_attack,
                                   attack_types=spec.attack_types)), f)
    print(f"\nГотово: {base}/ (all_seed_metrics.csv, aggregated.csv, "
          f"table_mean_std.csv, significance.csv, aggregated.pkl)")
    return alld, agg, sig


# ══════════════════════════════════════════════════════════
#  Общий CLI
# ══════════════════════════════════════════════════════════

def cli(spec: ProtocolSpec, argv=None):
    import argparse
    p = argparse.ArgumentParser(
        description=f"Мультисид-прогон и критерий Уилкоксона ({spec.name})")
    p.add_argument("seeds", nargs="*", type=int,
                   help="список сидов (по умолчанию 42..51)")
    p.add_argument("--csv", default=None, help="путь к датасету")
    p.add_argument("--out", default=None, help="каталог результатов")
    p.add_argument("--regen", action="store_true",
                   help="перегенерировать датасет для каждого сида")
    p.add_argument("--no-seq", action="store_true",
                   help="без LSTM-AE и Temporal-Attention-AE")
    p.add_argument("--with-ensemble", action="store_true",
                   help="включить специализированный ансамбль (медленно)")
    p.add_argument("--no-svm", action="store_true",
                   help="без One-Class SVM (ускоряет на больших выборках)")
    p.add_argument("--no-neural", action="store_true",
                   help="без Autoencoder/VAE и последовательных нейросетей")
    a = p.parse_args(argv)
    seeds = a.seeds or list(range(42, 52))
    run_seeds(spec, seeds, csv=a.csv, out=a.out, regen=a.regen,
              include_sequential=not a.no_seq,
              include_ensemble=a.with_ensemble, include_svm=not a.no_svm,
              include_neural=not a.no_neural)
