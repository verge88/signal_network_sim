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
    ("Cascade",           "Autoencoder",        "recall_at_val_1pct_fpr"),
    ("Meta-Ensemble",     "Random Forest",      "f1"),
    ("Random Forest",     "RF (no MS)",         "f1"),
    ("Random Forest",     "RF (no MS)",         "compromise_recall"),
    ("Random Forest",     "RF (no temporal)",   "compromise_recall"),
    ("Random Forest",     "Autoencoder",        "auc_pr"),
    ("VAE",               "Autoencoder",        "auc_pr"),
]

BASE_METRICS = [
    "auc_roc", "auc_pr", "f1", "fpr", "compromise_recall",
    # Deprecated compatibility alias: this is a post-hoc ROC diagnostic whose
    # threshold is selected from TEST negatives.
    "recall_at_1pct_fpr",
    "recall_at_1pct_fpr_posthoc",
    # Deployable operating point: threshold selected from VALIDATION negatives
    # and then applied unchanged to test.
    "recall_at_val_1pct_fpr",
    "test_fpr_at_val_1pct_threshold",
    "compromise_recall_at_val_1pct_fpr",
    "episode_recall", "episode_recall_at_val_1pct_fpr",
    "false_incidents_per_day",
    "false_incidents_per_day_at_val_1pct_threshold",
    # Fixed a-priori operational debounce: alarm only after two consecutive
    # positive windows on the same node. No test-set tuning of this rule.
    "debounce2_test_fpr_at_val_1pct_threshold",
    "debounce2_compromise_recall_at_val_1pct_fpr",
    "debounce2_episode_recall_at_val_1pct_fpr",
    "debounce2_false_incidents_per_day_at_val_1pct_threshold",
    "debounce2_median_detection_delay_s_at_val_1pct_fpr",
]


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
    """Post-hoc ROC diagnostic: threshold is selected from the same labels."""
    scores = np.nan_to_num(np.asarray(scores, float))
    y = np.asarray(y)
    if (y == 0).sum() == 0:
        return np.nan, np.nan
    thr = np.percentile(scores[y == 0], 100 * (1 - target_fpr))
    yp = (scores >= thr).astype(int)
    tp = ((yp == 1) & (y == 1)).sum()
    fn = ((yp == 0) & (y == 1)).sum()
    return 100.0 * tp / max(1, tp + fn), thr


def threshold_from_reference_fpr(scores, y, target_fpr=0.01):
    """Calibrate a threshold on a reference partition's negative labels only.

    The returned threshold guarantees empirical reference FPR <= target_fpr,
    including score ties. It must then be applied unchanged to a disjoint test
    partition.
    """
    scores = np.nan_to_num(np.asarray(scores, float))
    y = np.asarray(y)
    neg = scores[y == 0]
    if len(neg) == 0:
        return np.nan, np.nan

    allowed_fp = int(np.floor(float(target_fpr) * len(neg)))
    desc = np.sort(neg)[::-1]
    if allowed_fp <= 0:
        thr = np.nextafter(float(desc[0]), np.inf)
    elif allowed_fp >= len(desc):
        thr = -np.inf
    else:
        cutoff = float(desc[allowed_fp - 1])
        # With >= threshold, ties at the cutoff can exceed the FP budget.
        # Move just above the tied value when necessary.
        thr = cutoff
        if int(np.sum(neg >= thr)) > allowed_fp:
            thr = np.nextafter(cutoff, np.inf)

    empirical_fpr = float(np.mean(neg >= thr))
    return float(thr), empirical_fpr


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


def debounce_consecutive(df, predictions, required=2):
    """Require N consecutive positive windows per node before alarming.

    The rule is fixed independently of labels. Gaps in interval numbering reset
    the run, so disconnected grouped-holdout chunks cannot prime each other.
    Returned predictions preserve the original dataframe row order.
    """
    yp = np.asarray(predictions, dtype=int)
    if len(df) != len(yp):
        raise ValueError("df and predictions must have equal length")
    if required <= 1:
        return yp.copy()
    if "node_id" not in df.columns or "interval" not in df.columns:
        raise ValueError("debounce requires node_id and interval columns")

    tmp = df[["node_id", "interval"]].copy()
    tmp["_pred"] = yp
    tmp["_row_pos"] = np.arange(len(tmp))
    out = np.zeros(len(tmp), dtype=int)

    for _, part in tmp.sort_values(
        ["node_id", "interval", "_row_pos"]
    ).groupby("node_id", sort=False):
        run = 0
        prev_interval = None
        for cur_interval, pred, row_pos in part[
            ["interval", "_pred", "_row_pos"]
        ].itertuples(index=False, name=None):
            cur_interval = int(cur_interval)
            pred = int(pred)
            contiguous = (
                prev_interval is not None
                and cur_interval == prev_interval + 1
            )
            if not contiguous:
                run = 0
            run = run + 1 if pred else 0
            out[int(row_pos)] = 1 if run >= required else 0
            prev_interval = cur_interval
    return out


def episode_alarm_stats(test_df, predictions, labels, interval_s=15.0):
    """Episode recall and delay using alarms that occur inside attack rows."""
    yp = np.asarray(predictions, dtype=int)
    y = np.asarray(labels, dtype=int)
    if len(test_df) != len(yp) or len(yp) != len(y):
        raise ValueError("test_df, predictions and labels must have equal length")
    if "attack_episode_id" not in test_df.columns:
        return np.nan, np.nan, 0, 0

    epi = test_df["attack_episode_id"].astype(str).to_numpy()
    intervals = test_df["interval"].to_numpy(dtype=int)
    attack_mask = y == 1
    episode_ids = sorted({
        e for e in epi[attack_mask]
        if e and e.lower() not in ("nan", "none")
    })
    delays = []
    detected = 0
    for eid in episode_ids:
        m = (epi == eid) & attack_mask
        alarm_intervals = intervals[m & (yp == 1)]
        if len(alarm_intervals):
            detected += 1
            start = int(np.min(intervals[m]))
            first = int(np.min(alarm_intervals))
            delays.append(max(0.0, (first - start) * float(interval_s)))

    recall = (
        100.0 * detected / len(episode_ids)
        if episode_ids else np.nan
    )
    median_delay = float(np.median(delays)) if delays else np.nan
    return recall, median_delay, int(detected), int(len(episode_ids))


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
        intervals = part["interval"].to_numpy()
        fps = part["_fp"].to_numpy()
        for cur_interval, cur_fp in zip(intervals, fps):
            cur_interval = int(cur_interval)
            cur_fp = int(cur_fp)
            contiguous = (
                prev_interval is not None and cur_interval == prev_interval + 1
            )
            if cur_fp and (not prev_fp or not contiguous):
                incidents += 1
            prev_interval = cur_interval
            prev_fp = cur_fp
    return int(incidents)


def build_row(name, scores, y, attack_types, thr, spec, family,
              validation_scores=None, validation_y=None,
              operating_target_fpr=0.01):
    scores = np.nan_to_num(np.asarray(scores, float))
    y = np.asarray(y)
    yp = (scores >= thr).astype(int)
    multi = len(np.unique(y)) > 1
    tn, fp, fn, tp = confusion_matrix(y, yp, labels=[0, 1]).ravel()

    # Historical metric retained only for compatibility. Its threshold is
    # selected from test negatives, so it is explicitly labelled post-hoc.
    r1_posthoc, _ = recall_at_fpr(scores, y, operating_target_fpr)

    op_thr = np.nan
    val_fpr = np.nan
    op_recall = np.nan
    op_test_fpr = np.nan
    op_compromise = np.nan
    if validation_scores is not None and validation_y is not None:
        op_thr, val_fpr = threshold_from_reference_fpr(
            validation_scores, validation_y, operating_target_fpr
        )
        if np.isfinite(op_thr):
            op_yp = (scores >= op_thr).astype(int)
            op_tn, op_fp, op_fn, op_tp = confusion_matrix(
                y, op_yp, labels=[0, 1]
            ).ravel()
            op_recall = 100.0 * op_tp / max(1, op_tp + op_fn)
            op_test_fpr = (
                op_fp / (op_fp + op_tn) if (op_fp + op_tn) else np.nan
            )
            op_rec = by_attack(
                scores, attack_types, op_thr, spec.attack_types
            )
            op_compromise = op_rec.get(spec.compromise_attack, np.nan)

    row = dict(
        name=name, family=family, threshold=float(thr),
        auc_roc=roc_auc_score(y, scores) if multi else np.nan,
        auc_pr=average_precision_score(y, scores) if multi else np.nan,
        f1=f1_score(y, yp, zero_division=0),
        fpr=fp / (fp + tn) if (fp + tn) else 0.0,
        tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn),
        # Deprecated alias preserved so historical tables remain readable.
        recall_at_1pct_fpr=r1_posthoc,
        recall_at_1pct_fpr_posthoc=r1_posthoc,
        val_1pct_fpr_threshold=op_thr,
        val_fpr_at_val_1pct_threshold=val_fpr,
        recall_at_val_1pct_fpr=op_recall,
        test_fpr_at_val_1pct_threshold=op_test_fpr,
        compromise_recall_at_val_1pct_fpr=op_compromise,
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
               y=None, atypes=None, keep=True,
               s_val=None, y_val_ref=None):
        primary_rows = y is None and atypes is None and len(s_test) == len(test_df)
        y = y_test if y is None else y
        atypes = atk_test if atypes is None else atypes
        if y_val_ref is None and primary_rows:
            y_val_ref = y_val
        if keep:
            model_scores[name] = np.asarray(s_test, float)
        row = build_row(
            name, s_test, y, atypes, thr, spec, family,
            validation_scores=s_val,
            validation_y=y_val_ref,
        )

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

            op_thr = row.get("val_1pct_fpr_threshold", np.nan)
            if np.isfinite(op_thr):
                yp_op = (
                    np.nan_to_num(np.asarray(s_test, float)) >= op_thr
                ).astype(int)
                op_false_incidents = count_false_incidents(
                    test_df, yp_op, y_test
                )
                row["false_incidents_at_val_1pct_threshold"] = int(
                    op_false_incidents
                )
                row["false_incidents_per_day_at_val_1pct_threshold"] = (
                    float(op_false_incidents) / test_days
                    if np.isfinite(test_days) and test_days > 0 else np.nan
                )

                epi = test_df["attack_episode_id"].astype(str).to_numpy()
                attack_mask = np.asarray(y_test) == 1
                episode_ids = sorted({
                    e for e in epi[attack_mask]
                    if e and e.lower() not in ("nan", "none")
                })
                detected_op = 0
                for eid in episode_ids:
                    m = (epi == eid) & attack_mask
                    if np.any(yp_op[m] == 1):
                        detected_op += 1
                row["episode_recall_at_val_1pct_fpr"] = (
                    100.0 * detected_op / len(episode_ids)
                    if episode_ids else np.nan
                )

                # Operational layer: fixed a-priori 2-consecutive-window
                # debounce. The rule uses only score-threshold outputs and
                # timestamps/node IDs; labels are used only afterwards to
                # evaluate test performance.
                yp_db2 = debounce_consecutive(test_df, yp_op, required=2)
                db2_tn, db2_fp, db2_fn, db2_tp = confusion_matrix(
                    y_test, yp_db2, labels=[0, 1]
                ).ravel()
                row["debounce2_test_fpr_at_val_1pct_threshold"] = (
                    db2_fp / (db2_fp + db2_tn)
                    if (db2_fp + db2_tn) else np.nan
                )

                db2_false_incidents = count_false_incidents(
                    test_df, yp_db2, y_test
                )
                row["debounce2_false_incidents"] = int(
                    db2_false_incidents
                )
                row[
                    "debounce2_false_incidents_per_day_at_val_1pct_threshold"
                ] = (
                    float(db2_false_incidents) / test_days
                    if np.isfinite(test_days) and test_days > 0 else np.nan
                )

                db2_attack_rec = {}
                at = np.asarray(atk_test)
                for attack_name in spec.attack_types:
                    m = at == attack_name
                    if np.any(m):
                        db2_attack_rec[attack_name] = (
                            100.0 * np.sum(yp_db2[m]) / np.sum(m)
                        )
                row["debounce2_compromise_recall_at_val_1pct_fpr"] = (
                    db2_attack_rec.get(spec.compromise_attack, np.nan)
                )

                (
                    row["debounce2_episode_recall_at_val_1pct_fpr"],
                    row[
                        "debounce2_median_detection_delay_s_at_val_1pct_fpr"
                    ],
                    row["debounce2_detected_episodes"],
                    row["debounce2_test_episodes"],
                ) = episode_alarm_stats(
                    test_df, yp_db2, y_test, interval_s=interval_s
                )
            else:
                row["false_incidents_at_val_1pct_threshold"] = np.nan
                row["false_incidents_per_day_at_val_1pct_threshold"] = np.nan
                row["episode_recall_at_val_1pct_fpr"] = np.nan
                row["debounce2_test_fpr_at_val_1pct_threshold"] = np.nan
                row["debounce2_false_incidents"] = np.nan
                row[
                    "debounce2_false_incidents_per_day_at_val_1pct_threshold"
                ] = np.nan
                row[
                    "debounce2_compromise_recall_at_val_1pct_fpr"
                ] = np.nan
                row["debounce2_episode_recall_at_val_1pct_fpr"] = np.nan
                row[
                    "debounce2_median_detection_delay_s_at_val_1pct_fpr"
                ] = np.nan
                row["debounce2_detected_episodes"] = np.nan
                row["debounce2_test_episodes"] = np.nan
        else:
            row["n_test_episodes"] = np.nan
            row["n_detected_episodes"] = np.nan
            row["episode_recall"] = np.nan
            row["false_incidents"] = np.nan
            row["false_incidents_per_day"] = np.nan
            row["false_positive_windows_per_day"] = np.nan
            row["false_incidents_at_val_1pct_threshold"] = np.nan
            row["false_incidents_per_day_at_val_1pct_threshold"] = np.nan
            row["episode_recall_at_val_1pct_fpr"] = np.nan
            row["debounce2_test_fpr_at_val_1pct_threshold"] = np.nan
            row["debounce2_false_incidents"] = np.nan
            row[
                "debounce2_false_incidents_per_day_at_val_1pct_threshold"
            ] = np.nan
            row["debounce2_compromise_recall_at_val_1pct_fpr"] = np.nan
            row["debounce2_episode_recall_at_val_1pct_fpr"] = np.nan
            row[
                "debounce2_median_detection_delay_s_at_val_1pct_fpr"
            ] = np.nan
            row["debounce2_detected_episodes"] = np.nan
            row["debounce2_test_episodes"] = np.nan

        rows.append(row)
        op_r = row.get("recall_at_val_1pct_fpr", np.nan)
        op_r_text = f"{op_r:.1f}%" if np.isfinite(op_r) else "—"
        db2_ep = row.get("debounce2_episode_recall_at_val_1pct_fpr", np.nan)
        db2_ep_text = f"{db2_ep:.1f}%" if np.isfinite(db2_ep) else "—"
        print(f"  [{family:9s}] {name:22s} AUC-PR={row['auc_pr']:.3f} "
              f"F1={row['f1']:.3f} FPR={row['fpr']:.4f} "
              f"comp={row['compromise_recall']:.1f}% "
              f"R@val1%FPR={op_r_text} "
              f"db2_episode={db2_ep_text}")
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
    record("Isolation Forest", iso_t, th, s_val=iso_v)
    base_val["iso"] = iso_v; base_test["iso"] = iso_t

    # ── 2. One-Class SVM ──
    if include_svm:
        svm = OneClassSVM(nu=cfg.ocsvm_nu, kernel=cfg.ocsvm_kernel,
                          gamma="scale").fit(X_train_norm)
        svm_v, svm_t = -svm.score_samples(X_val), -svm.score_samples(X_test)
        th, _ = mod.find_threshold(svm_v, y_val, pct)
        record("One-Class SVM", svm_t, th, s_val=svm_v)
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
        record("Autoencoder", ae_t, th, s_val=ae_v)
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
        record("VAE", vae_t, th, s_val=vae_v)
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
            record(
                "LSTM-AE", lt.reconstruction_error(seq_t), th,
                family="unsup-seq", y=seq_yt, atypes=seq_atk, keep=False,
                s_val=lt.reconstruction_error(seq_v), y_val_ref=seq_yv,
            )

            tat = mod.TemporalAttentionAE(input_dim=n_feat, hidden_dim=64,
                                          n_heads=4, n_layers=2,
                                          seq_len=cfg.lstm_seq_len,
                                          dropout=0.15)
            tt = mod.AETrainer(tat, 5e-4, 100, 256, patience=15)
            tt.fit(seq_tr, seq_vn if len(seq_vn) else None)
            th, _ = mod.find_threshold(tt.reconstruction_error(seq_v),
                                       seq_yv, pct)
            record(
                "Temporal-Attention-AE", tt.reconstruction_error(seq_t),
                th, family="unsup-seq", y=seq_yt, atypes=seq_atk,
                keep=False, s_val=tt.reconstruction_error(seq_v),
                y_val_ref=seq_yv,
            )
        else:
            print("  [skip] недостаточно последовательностей для LSTM/TA-AE")

    # ── 6-7. Supervised ──
    rf = RandomForestClassifier(n_estimators=cfg.rf_trees, random_state=seed,
                                class_weight="balanced", n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_v = rf.predict_proba(X_val)[:, 1]
    rf_t = rf.predict_proba(X_test)[:, 1]
    record("Random Forest", rf_t, 0.5, family="oracle", s_val=rf_v)
    base_val["rf"] = rf_v; base_test["rf"] = rf_t

    gb = GradientBoostingClassifier(n_estimators=cfg.gb_estimators,
                                    learning_rate=cfg.gb_lr, max_depth=5,
                                    random_state=seed)
    gb.fit(X_train, y_train)
    gb_v = gb.predict_proba(X_val)[:, 1]
    gb_t = gb.predict_proba(X_test)[:, 1]
    record("Gradient Boosting", gb_t, 0.5, family="oracle", s_val=gb_v)
    base_val["gb"] = gb_v; base_test["gb"] = gb_t

    # ── 8. Каскадный детектор + его этапы ──
    casc = mod.CascadeCompromiseDetector(cfg, seed)
    casc.fit(X_train, y_train, Xtr_c, ytr_c)
    casc_v = casc.predict_proba(X_val, Xva_c)
    casc_t = casc.predict_proba(X_test, Xte_c)
    casc_s1_v = casc.predict_proba_stage1(X_val)
    casc_s1_t = casc.predict_proba_stage1(X_test)
    casc_s2_v = casc.predict_proba_stage2(Xva_c)
    casc_s2_t = casc.predict_proba_stage2(Xte_c)
    record("Cascade", casc_t, 0.5, family="oracle", s_val=casc_v)
    record(
        "Cascade-Stage1", casc_s1_t, 0.5,
        family="oracle", s_val=casc_s1_v,
    )
    record(
        "Cascade-Stage2", casc_s2_t, 0.5,
        family="oracle", s_val=casc_s2_v,
    )
    base_val["casc"] = casc_v
    base_test["casc"] = casc_t

    # ── 9. Специализированный ансамбль (медленно → по флагу) ──
    if include_ensemble:
        ens = mod.EnsembleDetector(mod.FEATURE_GROUPS, pipe.feature_cols, seed)
        ens.fit(X_train_norm)
        s_v, s_t = ens.score(X_val), ens.score(X_test)
        for k in sorted(s_t.keys()):
            th, _ = mod.find_threshold(s_v[k], y_val, pct)
            record(f"Ens:{k}", s_t[k], th, s_val=s_v[k])

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
    record(
        "Meta-Ensemble", meta.predict_proba(meta_te)[:, 1], 0.5,
        family="stacked",
        # No validation-calibrated operating point here: validation labels
        # already fitted the meta-model. A nested calibration partition is
        # required before this metric can be reported without optimism.
    )

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
        m_v = m.predict_proba(X_val[:, idx])[:, 1]
        m_t = m.predict_proba(X_test[:, idx])[:, 1]
        record(tag, m_t, 0.5, family="ablation", s_val=m_v)

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
        "operating_point": {
            "target_fpr": 0.01,
            "threshold_reference": "validation_negative_labels",
            "test_labels_used_for_threshold": False,
            "posthoc_compatibility_metric": "recall_at_1pct_fpr_posthoc",
            "deployable_metric": "recall_at_val_1pct_fpr",
            "meta_ensemble_deployable_metric": (
                "not_reported_validation_is_meta_fit_partition"
            ),
            "alarm_policy": {
                "name": "two_consecutive_positive_windows",
                "required_consecutive_windows": 2,
                "selected_using_test": False,
                "window_interval_seconds": 15.0,
            },
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
    head = [
        "auc_roc", "auc_pr", "f1", "fpr", "compromise_recall",
        "recall_at_val_1pct_fpr", "test_fpr_at_val_1pct_threshold",
        "debounce2_episode_recall_at_val_1pct_fpr",
        "debounce2_false_incidents_per_day_at_val_1pct_threshold",
    ]
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
