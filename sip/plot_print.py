"""
plot_results_print_sip.py — печатная ВАК-версия графиков (SIP, RFC 3261).

Оформление — точно как в plot_results_print_diameter.py / plot_result_print.py
(приложенные файлы). Изменено только:
  * имена моделей: добавлена 12-я модель ZCAD (научная новизна) под
    sip_training_v1.py; суффиксы "(v4)" / "(ensemble)" не используются;
  * таксономия атак (SIP: signaling_dos, register_hijack,
    register_bruteforce, spit, proxy_compromise);
  * блоки специализированного ансамбля (method_profile, response_profile,
    volume_timing, ...) и префикс результатов "Ens:";
  * добавлен Рис. 17 — компонентная абляция ZCAD (zone-attention,
    contrastive loss, stability aggregation) — научная новизна;
  * ДОБАВЛЕНО: Рис. 18 — корреляционная матрица признаков по группам;
              Рис. 19 — тепловая карта важности признаков по группам;
  * функция _spread_labels встроена в файл (нет внешнего модуля).

Все стили, размеры, шрифты, штриховка, фигуры, подписи осей, заголовки,
буквенные метки, форматы сохранения (png/svg/pdf @ 300 dpi) сохранены
один в один.
"""
import os
import pickle
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
import seaborn as sns
from sklearn.metrics import (
    roc_curve, precision_recall_curve, confusion_matrix,
    f1_score, precision_recall_curve as _prc,
)


# ----------------------------------------------------------------------
# Встроенная _spread_labels (вместо импорта из plot_results_print)
# ----------------------------------------------------------------------
def _spread_labels(ax, xs, ys, labels, min_dist=0.04, fontsize=10):
    """
    Разносит подписи точек по вертикали, чтобы не перекрывались.
    Жадный алгоритм: сортируем по y и сдвигаем подпись вверх, если
    она ближе min_dist к предыдущей.
    """
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    order = np.argsort(ys)
    placed_y = ys[order].copy()
    for i in range(1, len(placed_y)):
        if placed_y[i] - placed_y[i - 1] < min_dist:
            placed_y[i] = placed_y[i - 1] + min_dist
    label_y = np.empty_like(placed_y)
    label_y[order] = placed_y
    for x, y, ly, lab in zip(xs, ys, label_y, labels):
        arrow = (dict(arrowstyle="-", color="gray", lw=0.6, alpha=0.7)
                 if abs(ly - y) > 1e-6 else None)
        ax.annotate(lab, xy=(x, y), xytext=(x + 0.005, ly),
                    fontsize=fontsize, va="center",
                    arrowprops=arrow)


# ----------------------------------------------------------------------
# Печатные стили (увеличен шрифт, ч/б палитра) — БЕЗ ИЗМЕНЕНИЙ
# ----------------------------------------------------------------------
mpl.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "font.family": "DejaVu Sans",
    "font.size": 13,
    "axes.titlesize": 15,
    "axes.labelsize": 14,
    "legend.fontsize": 11,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "grid.linestyle": ":",
    "axes.linewidth": 1.2,
    "lines.linewidth": 2.0,
    "lines.markersize": 7,
    "patch.linewidth": 1.0,
    "hatch.linewidth": 0.8,
})

# ── Имена моделей в sip_training_v1.py (12 моделей; ZCAD) ──
MAIN_MODELS = [
    "Isolation Forest",
    "One-Class SVM",
    "Autoencoder",
    "VAE",
    "LSTM-Autoencoder",
    "Temporal-Attention-AE",
    "Random Forest",
    "Gradient Boosting",
    "Cascade",
    "Meta-Ensemble",
    "ZCAD",
]

RU_NAME = {
    "Isolation Forest":        "Изолирующий лес",
    "One-Class SVM":           "Одноклассовая SVM",
    "Autoencoder":             "Автокодировщик",
    "VAE":                     "Вариац. автокодировщик",
    "LSTM-Autoencoder":        "LSTM-автокодировщик",
    "Temporal-Attention-AE":   "Временное внимание",
    "Random Forest":           "Случайный лес",
    "Gradient Boosting":       "Градиентный бустинг",
    "Cascade":                 "Каскадный детектор",
    "Meta-Ensemble":           "Мета-ансамбль",
    "ZCAD":                  "ZCAD (ведущий-ведомый)",
}

EN_NAME = {
    "Isolation Forest":        "Isolation Forest",
    "One-Class SVM":           "One-Class SVM",
    "Autoencoder":             "Autoencoder",
    "VAE":                     "VAE",
    "LSTM-Autoencoder":        "LSTM-Autoencoder",
    "Temporal-Attention-AE":   "Temporal-Attention-AE",
    "Random Forest":           "Random Forest",
    "Gradient Boosting":       "Gradient Boosting",
    "Cascade":                 "Cascade detector",
    "Meta-Ensemble":           "Meta-Ensemble",
    "ZCAD":                  "Zone-Contextual Anomaly Detector",
}


def _en(name: str) -> str:
    return EN_NAME.get(name, name)


# Чёрно-белый стиль — структура и значения параметров СОХРАНЕНЫ.
# Добавлена 11-я запись для ZCAD (буква "л").
BW_STYLE = {
    "Isolation Forest":
        {"c": "#000000", "ls": "-",       "m": "o", "lw": 1.6, "h": "///",  "letter": "а"},
    "One-Class SVM":
        {"c": "#333333", "ls": "--",      "m": "s", "lw": 1.6, "h": "\\\\\\","letter": "б"},
    "Autoencoder":
        {"c": "#555555", "ls": "-.",      "m": "^", "lw": 1.8, "h": "xxx",  "letter": "в"},
    "VAE":
        {"c": "#777777", "ls": ":",       "m": "D", "lw": 1.8, "h": "...",  "letter": "г"},
    "LSTM-Autoencoder":
        {"c": "#222222", "ls": (0, (3, 1, 1, 1)), "m": "v", "lw": 1.8, "h": "+++", "letter": "д"},
    "Temporal-Attention-AE":
        {"c": "#444444", "ls": (0, (5, 2, 1, 2)), "m": "P", "lw": 1.8, "h": "|||", "letter": "е"},
    "Random Forest":
        {"c": "#000000", "ls": "-",       "m": "X", "lw": 2.4, "h": "---",  "letter": "ж"},
    "Gradient Boosting":
        {"c": "#111111", "ls": (0, (1, 1)), "m": "*", "lw": 2.2, "h": "ooo","letter": "з"},
    "Cascade":
        {"c": "#000000", "ls": "--",      "m": "h", "lw": 2.4, "h": "OOO",  "letter": "и"},
    "Meta-Ensemble":
        {"c": "#000000", "ls": "-.",      "m": "<", "lw": 2.4, "h": "***",  "letter": "к"},
    "ZCAD":
        {"c": "#000000", "ls": (0, (4, 1, 1, 1, 1, 1)), "m": ">", "lw": 2.6, "h": "\\|/", "letter": "л"},
}

# ── Атаки SIP (вместо Diameter) ──
ATTACK_RU = {
    "proxy_compromise":     "Компрометация прокси",
    "signaling_dos":        "Сигнальный DoS (INVITE-флуд)",
    "register_hijack":      "Перехват регистрации",
    "register_bruteforce":  "Перебор учётных данных",
    "spit":                 "SPIT (спам-вызовы)",
}

# ── Блоки специализированного ансамбля (SIP) ──
BLOCK_RU = {
    "method_profile":        "Профиль методов",
    "response_profile":      "Профиль ответов",
    "volume_timing":         "Объём/время",
    "international":         "Международные",
    "control_unit":          "Управляющий узел",
    "baseline_deviation":    "Отклонение от нормы",
    "compromise_indicators": "Признаки компрометации",
    "transaction_integrity": "Транзакц. целостность",
    "spit_indicators":       "Признаки SPIT",
    "ensemble_max":          "Ансамбль: максимум",
    "ensemble_mean":         "Ансамбль: среднее",
    "ensemble_weighted":     "Ансамбль: взвешенный",
}


# ----------------------------------------------------------------------
# ГРУППЫ ПРИЗНАКОВ (для Рис. 18 и Рис. 19).
# Отображение "префикс/имя признака -> функциональная группа".
# Совпадает с DataPipeline из sip_training_v1.py. Порядок групп
# определяет порядок блоков на матрице корреляций.
# ----------------------------------------------------------------------
FEATURE_GROUP_ORDER = [
    "control_unit",
    "transaction_integrity",
    "volume_stats",
    "method_ratio",
    "response_ratio",
    "temporal",
    "zone_deviation",
    "other",
]

FEATURE_GROUP_RU = {
    "control_unit":          "Управляющий узел\n(master–slave)",
    "transaction_integrity": "Транзакционная\nцелостность SIP",
    "volume_stats":          "Объёмные /\nстатистические",
    "method_ratio":          "Доли SIP-\nметодов",
    "response_ratio":        "Доли классов\nответов",
    "temporal":              "Временные\n(var/autocorr)",
    "zone_deviation":        "Зональные\nотклонения",
    "other":                 "Прочие",
}

# Явные списки признаков для транзакционной группы (как в DataPipeline).
_TXN_FEATURES = {
    "obs_forking_amplification", "obs_invite_timeout_ratio",
    "obs_invite_mean_retransmits", "obs_invite_frac_retransmitted",
    "obs_dialog_completion_ratio", "obs_dangling_dialog_ratio",
    "obs_noninvite_timeout_ratio", "obs_txn_timeout_ratio",
    "cu_txn_state_consistency_error", "mean_txn_state_consistency_error",
    "dev_txn_consistency_vs_zone", "dev_dialog_completion_ratio_vs_zone",
    "dev_dangling_dialog_ratio_vs_zone", "dev_forking_amplification_vs_zone",
    "var_obs_dialog_completion_ratio", "var_obs_forking_amplification",
    "z_obs_dangling_dialog_ratio", "z_obs_invite_timeout_ratio",
    "z_obs_forking_amplification",
}

_CU_FEATURES = {
    "cu_delivered", "cu_rtt_delay_ms", "cu_req_delay_ms",
    "cu_resp_delay_ms", "cu_combined_loss_prob", "integrity_check",
    "consecutive_cu_losses", "cu_processing_delay_ms",
    "cu_response_time_jitter", "cu_staleness", "cu_consistency_error",
    "integrity_fail_rate",
}


def _feature_group(col: str) -> str:
    """Определяет функциональную группу признака по имени."""
    # Транзакционная целостность имеет приоритет (может пересекаться
    # с temporal/zone по префиксам var_/dev_/z_).
    if col in _TXN_FEATURES:
        return "transaction_integrity"
    if col in _CU_FEATURES or col.startswith("cu_"):
        return "control_unit"
    if col.startswith("dev_") or col == "zone_io_ratio_rank":
        return "zone_deviation"
    if col.startswith("var_") or col.startswith("autocorr_") or \
            col == "obs_destinations_cv":
        return "temporal"
    if col.startswith("z_"):
        return "temporal"
    if col.startswith("obs_ratio_resp"):
        return "response_ratio"
    if col.startswith("obs_ratio_"):
        return "method_ratio"
    if col.startswith("obs_"):
        return "volume_stats"
    return "other"


# ----------------------------------------------------------------------
# Утилиты — БЕЗ ИЗМЕНЕНИЙ
# ----------------------------------------------------------------------
def _ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def _save(fig, out_dir, name, formats=("png", "svg", "pdf")):
    _ensure_dir(out_dir)
    for ext in formats:
        fig.savefig(os.path.join(out_dir, f"{name}.{ext}"),
                    bbox_inches="tight",
                    dpi=300 if ext == "png" else None,
                    facecolor="white")
    plt.close(fig)


def _find(results_list, name):
    for r in results_list:
        if r.get("name") == name:
            return r
    return None


def _top_models_by_f1(results_list, k=4):
    cand = [r for r in results_list
            if r.get("scores") is not None
            and r.get("threshold") is not None]
    cand.sort(key=lambda r: r.get("f1", 0.0), reverse=True)
    return cand[:k]


def _ru(name):
    return RU_NAME.get(name, name)


def _ru_block(name):
    if name in RU_NAME:
        return RU_NAME[name]
    # Поддерживаем оба префикса: "Ens:" (sip_training_v1) и "Ensemble:"
    if name.startswith("Ens:"):
        key = name.replace("Ens:", "")
        return f"Ансамбль: {BLOCK_RU.get(key, key)}"
    if name.startswith("Ensemble:"):
        key = name.replace("Ensemble:", "")
        return f"Ансамбль: {BLOCK_RU.get(key, key)}"
    return name


def _style(name):
    return BW_STYLE.get(name, {"c": "#666666", "ls": "-", "m": "o",
                                "lw": 1.6, "h": "", "letter": "?"})


def _maxf1_threshold(scores, y_true):
    scores = np.asarray(scores)
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return float(np.median(scores)), 0.0
    prec, rec, ths = _prc(y_true, scores)
    f1s = 2 * prec * rec / (prec + rec + 1e-12)
    idx = int(np.argmax(f1s[:-1])) if len(ths) else 0
    th = float(ths[idx]) if len(ths) else float(np.median(scores))
    return th, float(f1s[idx])


def _recompute_metrics(scores, y_true, threshold):
    scores = np.asarray(scores)
    y_true = np.asarray(y_true)
    y_pred = (scores >= threshold).astype(int)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return float(f1), float(fpr)


def _bw_letter_label(letter, ru_name, value=None, value_name="AUC",
                     en_name=None):
    base = f"{letter} — {ru_name}"
    if en_name and en_name != ru_name:
        base += f" ({en_name})"
    if value is not None:
        base += f"; {value_name} = {value:.3f}"
    return base


# Доп. защитная проверка — некоторые модели (LSTM-AE, TA-AE) работают
# с seq_y_test другой длины. Пропускаем такие модели в "общих" графиках.
def _len_ok(r, n):
    s = r.get("scores")
    return s is not None and len(np.asarray(s)) == n


# ======================================================================
# Рис. 1. ROC-кривые — БЕЗ ИЗМЕНЕНИЙ ОФОРМЛЕНИЯ
# ======================================================================
def plot_roc_curves(results_list, y_test, out_dir, formats):
    fig, ax = plt.subplots(figsize=(11, 8))
    for name in MAIN_MODELS:
        r = _find(results_list, name)
        if r is None or r.get("scores") is None:
            continue
        if len(np.asarray(r["scores"])) != len(y_test):
            continue
        fpr, tpr, _ = roc_curve(y_test, r["scores"])
        st = _style(name)
        step = max(1, len(fpr) // 8)
        ax.plot(fpr, tpr,
                color=st["c"], linestyle=st["ls"], linewidth=st["lw"],
                marker=st["m"], markevery=step, markersize=8,
                markeredgecolor="black", markerfacecolor="white",
                label=_bw_letter_label(st["letter"], _ru(name),
                                       r["auc_roc"], "AUC"))
    ax.plot([0, 1], [0, 1], color="black", linestyle=":",
            linewidth=1.2, alpha=0.6, label="базовая линия (baseline y = x)")
    ax.set_xlabel("Доля ложных срабатываний FPR, отн. ед.\n"
                  "(False Positive Rate, rel. units)")
    ax.set_ylabel("Доля истинно положительных TPR, отн. ед.\n"
                  "(True Positive Rate, rel. units)")
    ax.set_title("Рисунок 1 — ROC-кривые моделей обнаружения аномалий\n"
                 "(ROC curves of anomaly detection models)")
    ax.legend(loc="lower right", framealpha=0.95, edgecolor="black",
              fontsize=10)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    fig.tight_layout()
    _save(fig, out_dir, "01_roc_curves", formats)


def plot_roc_zoom(results_list, y_test, out_dir, formats):
    fig, ax = plt.subplots(figsize=(11, 8))
    for name in MAIN_MODELS:
        r = _find(results_list, name)
        if r is None or r.get("scores") is None:
            continue
        if len(np.asarray(r["scores"])) != len(y_test):
            continue
        fpr, tpr, _ = roc_curve(y_test, r["scores"])
        st = _style(name)
        step = max(1, len(fpr) // 8)
        ax.plot(fpr, tpr,
                color=st["c"], linestyle=st["ls"], linewidth=st["lw"],
                marker=st["m"], markevery=step, markersize=8,
                markeredgecolor="black", markerfacecolor="white",
                label=_bw_letter_label(st["letter"], _ru(name),
                                       r["auc_roc"], "AUC"))
    ax.set_xlim(0, 0.05)
    ax.set_ylim(0.5, 1.005)
    ax.set_xlabel("Доля ложных срабатываний FPR, отн. ед.\n"
                  "(False Positive Rate, rel. units)")
    ax.set_ylabel("Доля истинно положительных TPR, отн. ед.\n"
                  "(True Positive Rate, rel. units)")
    ax.set_title("Рисунок 2 — ROC-кривые в области малых FPR\n"
                 "(ROC curves — low FPR region)")
    ax.legend(loc="lower right", framealpha=0.95, edgecolor="black",
              fontsize=10)
    fig.tight_layout()
    _save(fig, out_dir, "02_roc_zoom", formats)


# ======================================================================
# Рис. 3. PR-кривые
# ======================================================================
def plot_pr_curves(results_list, y_test, out_dir, formats):
    fig, ax = plt.subplots(figsize=(11, 8))
    for name in MAIN_MODELS:
        r = _find(results_list, name)
        if r is None or r.get("scores") is None:
            continue
        if len(np.asarray(r["scores"])) != len(y_test):
            continue
        prec, rec, _ = precision_recall_curve(y_test, r["scores"])
        st = _style(name)
        step = max(1, len(rec) // 8)
        ax.plot(rec, prec,
                color=st["c"], linestyle=st["ls"], linewidth=st["lw"],
                marker=st["m"], markevery=step, markersize=8,
                markeredgecolor="black", markerfacecolor="white",
                label=_bw_letter_label(st["letter"], _ru(name),
                                       r["auc_pr"], "AUC-PR", _en(name)))
    ax.set_xlabel("Полнота, отн. ед.\n"
                  "(Recall, rel. units)")
    ax.set_ylabel("Точность, отн. ед.\n"
                  "(Precision, rel. units)")
    ax.legend(loc="lower left", framealpha=0.95, edgecolor="black",
              fontsize=10)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    fig.tight_layout()
    _save(fig, out_dir, "03_pr_curves", formats)


# ======================================================================
# Рис. 4. Сводные метрики
# ======================================================================
def plot_metric_bars(results_list, y_test, out_dir, formats):
    rows = []
    for name in MAIN_MODELS:
        r = _find(results_list, name)
        if r is None:
            continue
        st = _style(name)
        scores = r.get("scores")
        f1_used = r.get("f1", 0)
        fpr_used = r.get("fpr", 0)
        if scores is not None and len(scores) == len(y_test) and fpr_used > 0.5:
            th_new, _ = _maxf1_threshold(scores, y_test)
            f1_used, fpr_used = _recompute_metrics(scores, y_test, th_new)
        rows.append({
            "letter":  st["letter"],
            "model":   f"{st['letter']} — {_ru(name)}",
            "hatch":   st["h"],
            "AUC-ROC": r.get("auc_roc", 0),
            "AUC-PR":  r.get("auc_pr", 0),
            "F1":      f1_used,
            "FPR":     fpr_used,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return

    metric_titles = {
        "AUC-ROC": "(а) AUC-ROC, отн. ед.\n(rel. units)",
        "AUC-PR":  "(б) AUC-PR, отн. ед.\n(rel. units)",
        "F1":      "(в) F1-мера, отн. ед.\n(F1-score, rel. units)",
        "FPR":     "(г) Доля ложных срабатываний FPR, отн. ед.\n"
                   "(False Positive Rate, rel. units)",
    }
    metrics = list(metric_titles.keys())
    fig, axes = plt.subplots(2, 2, figsize=(17, 13))
    for ax, m in zip(axes.flat, metrics):
        bars = ax.barh(df["model"], df[m],
                       color="white",
                       edgecolor="black", linewidth=1.0)
        for bar, hatch in zip(bars, df["hatch"]):
            bar.set_hatch(hatch)
        ax.set_xlabel(metric_titles[m])
        ax.set_title(metric_titles[m].split("\n")[0])
        if m == "FPR":
            ax.set_xlim(0, max(df[m].max() * 1.15, 0.05))
        else:
            ax.set_xlim(0, 1.05)
        for i, v in enumerate(df[m]):
            ax.text(v, i, f" {v:.4f}", va="center", fontsize=11)
    fig.suptitle(
        "Рисунок 4 — Сводные метрики моделей обнаружения аномалий\n"
        "(Per-model metrics summary; порог = max-F1 / threshold = max-F1)",
        fontsize=16, y=1.00)
    fig.tight_layout()
    _save(fig, out_dir, "04_metric_bars", formats)


# ======================================================================
# Рис. 5. Матрицы ошибок (фикс. 2x2)
# ======================================================================
def plot_confusion_matrices(results_list, y_test, out_dir, formats, k=4):
    top = [r for r in _top_models_by_f1(results_list, k=k)
           if len(np.asarray(r["scores"])) == len(y_test)]
    if not top:
        return

    rows_n, cols_n = 2, 2
    fig, axes = plt.subplots(rows_n, cols_n, figsize=(13, 11))
    letters = ["а", "б", "в", "г"]

    for idx in range(rows_n * cols_n):
        ax = axes[idx // cols_n, idx % cols_n]
        if idx >= len(top):
            ax.axis("off")
            continue
        r = top[idx]
        thr = r.get("threshold", 0.5)
        y_pred = (np.asarray(r["scores"]) >= thr).astype(int)
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Greys", ax=ax,
                    cbar=False, annot_kws={"size": 15, "weight": "bold"},
                    linewidths=0.8, linecolor="black",
                    vmin=0, vmax=cm.max())
        ax.set_title(
            f"({letters[idx]}) {_ru(r['name'])} ({(_en(r['name']))})\n"
            f"F1 = {r.get('f1',0):.4f}; порог (threshold) = {thr:.3f}",
            fontsize=13)
        ax.set_xlabel("Прогноз (Predicted)")
        ax.set_ylabel("Факт (Actual)")
        ax.set_xticklabels(["норма\n(normal)", "аномалия\n(anomaly)"])
        ax.set_yticklabels(["норма\n(normal)", "аномалия\n(anomaly)"],
                           rotation=0)
    fig.tight_layout()
    _save(fig, out_dir, "05_confusion_matrices", formats)


# ======================================================================
# Рис. 6. Полнота по типам атак
# ======================================================================
def plot_per_attack_recall(results_list, attack_types_test, y_test,
                           out_dir, formats):
    attacks = sorted(a for a in pd.Series(attack_types_test).unique()
                     if a != "none")
    rows, model_names = [], []
    for name in MAIN_MODELS:
        r = _find(results_list, name)
        if r is None or r.get("scores") is None:
            continue
        if len(np.asarray(r["scores"])) != len(attack_types_test):
            continue
        thr = r.get("threshold", 0.5)
        yp = (np.asarray(r["scores"]) >= thr).astype(int)
        row = []
        for a in attacks:
            m = (np.asarray(attack_types_test) == a)
            row.append(yp[m].mean() if m.sum() else np.nan)
        rows.append(row)
        st = _style(name)
        model_names.append(f"{st['letter']} — {_ru(name)}")
    if not rows:
        return
    mat = np.array(rows)
    fig, ax = plt.subplots(
        figsize=(2.2 * len(attacks) + 6, 0.7 * len(model_names) + 3))
    sns.heatmap(mat, annot=False, cmap="Greys",
                xticklabels=[ATTACK_RU.get(a, a) for a in attacks],
                yticklabels=model_names,
                vmin=0, vmax=1, ax=ax,
                cbar_kws={"label": "Полнота Recall, отн. ед.\n"
                                   "(Recall, rel. units)"},
                linewidths=0.8, linecolor="black")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if np.isnan(v):
                continue
            color = "white" if v > 0.55 else "black"
            ax.text(j + 0.5, i + 0.5, f"{v:.2f}",
                    ha="center", va="center", color=color, fontsize=11)
    ax.set_title("Рисунок 6 — Полнота обнаружения по типам атак\n"
                 "(Per-attack recall by model)")
    ax.set_xlabel("Тип атаки (Attack type)")
    ax.set_ylabel("Модель (Model)")
    fig.tight_layout()
    _save(fig, out_dir, "06_per_attack_recall", formats)


# ======================================================================
# Рис. 7. Важность признаков
# ======================================================================
def plot_feature_importance(importances, feature_cols, out_dir, formats,
                            top=20):
    if importances is None or len(importances) == 0:
        return
    df = pd.DataFrame({"feature": feature_cols, "importance": importances})
    df = df.sort_values("importance", ascending=False).head(top)
    fig, ax = plt.subplots(figsize=(12, 0.5 * top + 2.5))
    ax.barh(df["feature"][::-1], df["importance"][::-1],
            color="white", edgecolor="black", linewidth=1.0, hatch="///")
    ax.set_xlabel("Важность признака, отн. ед.\n"
                  "(Feature importance, rel. units)")
    ax.set_ylabel("Признак (Feature)")
    for i, v in enumerate(df["importance"][::-1]):
        ax.text(v, i, f" {v:.3f}", va="center", fontsize=11)
    fig.tight_layout()
    _save(fig, out_dir, "07_feature_importance", formats)


# ======================================================================
# Рис. 8. Распределения оценок (топ-3)
# ======================================================================
def plot_score_distributions(results_list, y_test, out_dir, formats, k=3):
    top = [r for r in _top_models_by_f1(results_list, k=k)
           if len(np.asarray(r["scores"])) == len(y_test)]
    if not top:
        return
    fig, axes = plt.subplots(1, len(top), figsize=(8.5 * len(top), 6.5))
    if len(top) == 1:
        axes = [axes]
    letters = ["а", "б", "в", "г"]
    for i, (ax, r) in enumerate(zip(axes, top)):
        s = np.asarray(r["scores"])
        thr = r.get("threshold", 0.5)
        ax.hist(s[y_test == 0], bins=50, alpha=0.85, color="white",
                edgecolor="black", linewidth=1.0, hatch="///",
                label="норма (normal)", density=True)
        ax.hist(s[y_test == 1], bins=50, alpha=0.85, color="#888888",
                edgecolor="black", linewidth=1.0, hatch="xxx",
                label="аномалия (anomaly)", density=True)
        ax.axvline(thr, color="black", linestyle="--", linewidth=2.4,
                   label=f"порог (threshold) = {thr:.3f}")
        ax.set_title(f"({letters[i]}) {_ru(r['name'])}; "
                     f"F1 = {r.get('f1',0):.4f}",
                     fontsize=14)
        ax.set_xlabel("Оценка аномальности, отн. ед.\n"
                      "(Anomaly score, rel. units)")
        ax.set_ylabel("Плотность распределения, отн. ед.\n"
                      "(Probability density, rel. units)")
        ax.legend(fontsize=11, edgecolor="black", loc="upper center")
        ax.tick_params(axis="both", labelsize=12)
    fig.suptitle(
        "Рисунок 8 — Распределения оценок аномальности для топ-моделей\n"
        "(Score distributions — top models)", fontsize=15, y=1.03)
    fig.tight_layout()
    _save(fig, out_dir, "08_score_distributions", formats)


# ======================================================================
# Рис. 9. Анализ каскадного детектора
# ======================================================================
def plot_cascade_analysis(results_list, cascade_s1, cascade_s2,
                          cascade_comb, y_test, out_dir, formats):
    r = _find(results_list, "Cascade")
    if r is None and cascade_comb is None:
        return
    thr = r["threshold"] if r is not None else 0.5
    scores = np.asarray(r["scores"]) if r is not None \
        else np.asarray(cascade_comb)
    if len(scores) != len(y_test):
        return
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))

    axes[0].hist(scores[y_test == 0], bins=50, alpha=0.85, color="white",
                 edgecolor="black", linewidth=1.0, hatch="///",
                 label="норма (normal)", density=True)
    axes[0].hist(scores[y_test == 1], bins=50, alpha=0.85, color="#888888",
                 edgecolor="black", linewidth=1.0, hatch="xxx",
                 label="аномалия (anomaly)", density=True)
    axes[0].axvline(thr, color="black", linestyle="--", linewidth=2.4,
                    label=f"порог (threshold) = {thr:.3f}")
    axes[0].set_title("(а) Распределение оценок каскадного детектора\n"
                      "(Cascade detector score distribution)")
    axes[0].set_xlabel("Оценка аномальности, отн. ед.\n"
                       "(Anomaly score, rel. units)")
    axes[0].set_ylabel("Плотность распределения, отн. ед.\n"
                       "(Probability density, rel. units)")
    axes[0].legend(edgecolor="black")

    grid = np.linspace(scores.min(), scores.max(), 250)
    f1s = []
    for t in grid:
        yp = (scores >= t).astype(int)
        tp = ((yp == 1) & (y_test == 1)).sum()
        fp = ((yp == 1) & (y_test == 0)).sum()
        fn = ((yp == 0) & (y_test == 1)).sum()
        p = tp / (tp + fp) if (tp + fp) else 0
        rec = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * p * rec / (p + rec) if (p + rec) else 0
        f1s.append(f1)
    axes[1].plot(grid, f1s, color="black", linewidth=2.2,
                 marker="o", markevery=20, markerfacecolor="white",
                 markeredgecolor="black", markersize=7)
    axes[1].axvline(thr, color="black", linestyle="--", linewidth=2.4,
                    label=f"выбранный порог (selected threshold) = "
                          f"{thr:.3f}")
    axes[1].set_xlabel("Порог, отн. ед.\n(Threshold, rel. units)")
    axes[1].set_ylabel("F1-мера, отн. ед.\n(F1-score, rel. units)")
    axes[1].set_title("(б) Зависимость F1-меры от порога\n"
                      "(F1-score vs. threshold)")
    axes[1].legend(edgecolor="black")

    fig.suptitle(
        "Рисунок 9 — Анализ каскадного детектора\n"
        "(Cascade detector analysis)",
        fontsize=15, y=1.02)
    fig.tight_layout()
    _save(fig, out_dir, "09_cascade_analysis", formats)


# ======================================================================
# Рис. 10. Сравнение по обнаружению компрометации прокси
# ======================================================================
def plot_compromise_comparison(results_list, attack_types_test, y_test,
                               out_dir, formats):
    mask = (np.asarray(attack_types_test) == "proxy_compromise")
    if mask.sum() == 0:
        return
    rows = []
    for r in results_list:
        if r.get("scores") is None:
            continue
        if len(np.asarray(r["scores"])) != len(attack_types_test):
            continue
        if r.get("f1", 0) < 0.4:
            continue
        thr = r.get("threshold", 0.5)
        yp = (np.asarray(r["scores"]) >= thr).astype(int)
        st = _style(r["name"])
        rows.append({"model": _ru_block(r["name"]),
                     "raw_name": r["name"],
                     "hatch": st.get("h", "///"),
                     "compromise_recall": yp[mask].mean(),
                     "f1": r.get("f1", 0)})
    if not rows:
        return
    df = pd.DataFrame(rows).sort_values("compromise_recall", ascending=True)
    fig, ax = plt.subplots(figsize=(12, 0.6 * len(df) + 2.5))
    bars = ax.barh(df["model"], df["compromise_recall"],
                   color="white", edgecolor="black", linewidth=1.0)
    for bar, hatch in zip(bars, df["hatch"]):
        bar.set_hatch(hatch)
    ax.set_xlabel("Полнота обнаружения компрометации, отн. ед.\n"
                  "(Compromise detection recall, rel. units)")
    ax.set_ylabel("Модель / блок ансамбля\n(Model / ensemble block)")
    ax.set_xlim(0, 1.05)
    for i, v in enumerate(df["compromise_recall"]):
        ax.text(v, i, f" {v:.3f}", va="center", fontsize=11)
    ax.set_title("Рисунок 10 — Сравнение моделей по обнаружению "
                 "компрометации прокси (F1 ≥ 0,4)\n"
                 "(Proxy compromise recall comparison; F1 ≥ 0.4)")
    fig.tight_layout()
    _save(fig, out_dir, "10_compromise_comparison", formats)


# ======================================================================
# Рис. 11. Тепловая карта блоков ансамбля
# ======================================================================
def plot_ensemble_blocks(ens_results, attack_types_test, out_dir, formats):
    if not ens_results:
        return
    attacks = sorted(a for a in pd.Series(attack_types_test).unique()
                     if a != "none")
    rows, names = [], []
    for r in ens_results:
        if r.get("scores") is None:
            continue
        if len(np.asarray(r["scores"])) != len(attack_types_test):
            continue
        thr = r.get("threshold", 0.5)
        yp = (np.asarray(r["scores"]) >= thr).astype(int)
        row = []
        for a in attacks:
            m = (np.asarray(attack_types_test) == a)
            row.append(yp[m].mean() if m.sum() else np.nan)
        rows.append(row)
        raw = r["name"]
        if raw.startswith("Ens:"):
            block_key = raw.replace("Ens:", "")
        elif raw.startswith("Ensemble:"):
            block_key = raw.replace("Ensemble:", "")
        else:
            block_key = raw
        names.append(BLOCK_RU.get(block_key, block_key))
    if not rows:
        return
    mat = np.array(rows)
    fig, ax = plt.subplots(
        figsize=(2.2 * len(attacks) + 6, 0.65 * len(names) + 3))
    sns.heatmap(mat, annot=False, cmap="Greys",
                xticklabels=[ATTACK_RU.get(a, a) for a in attacks],
                yticklabels=names,
                vmin=0, vmax=1, ax=ax,
                cbar_kws={"label": "Полнота Recall, отн. ед.\n"
                                   "(Recall, rel. units)"},
                linewidths=0.8, linecolor="black")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if np.isnan(v):
                continue
            color = "white" if v > 0.55 else "black"
            ax.text(j + 0.5, i + 0.5, f"{v:.2f}",
                    ha="center", va="center", color=color, fontsize=11)
    ax.set_title("Рисунок 11 — Полнота специализированных блоков ансамбля "
                 "по типам атак\n"
                 "(Specialized ensemble blocks — recall per attack type)")
    ax.set_xlabel("Тип атаки (Attack type)")
    ax.set_ylabel("Блок ансамбля (Ensemble block)")
    fig.tight_layout()
    _save(fig, out_dir, "11_ensemble_blocks_heatmap", formats)


# ======================================================================
# Рис. 12. Аблационное исследование (наборы признаков / каскад)
# ======================================================================
def plot_ablation(ablation_results, out_dir, formats):
    if not ablation_results:
        return
    df = pd.DataFrame(ablation_results)

    def _translate_config(cfg: str) -> str:
        if cfg.startswith("Full"):
            return cfg.replace("Full", "Полный набор\nпризнаков ")
        if cfg.startswith("No MS"):
            return cfg.replace("No MS", "Без MS-\nпризнаков ")
        if cfg.startswith("No temporal"):
            return cfg.replace("No temporal", "Без временных\nпризнаков ")
        if cfg.startswith("No txn"):
            return cfg.replace("No txn", "Без транзакц.\nпризнаков ")
        if "Stage-1" in cfg:
            return "Каскад:\nтолько этап 1"
        if "Both" in cfg:
            return "Каскад:\nоба этапа"
        return cfg

    if "config" in df.columns:
        df["config"] = df["config"].map(_translate_config)

    if "f1" in df.columns and "config" in df.columns:
        df = df.sort_values("f1")
        fig, axes = plt.subplots(1, 2, figsize=(17, max(4.5, 0.8 * len(df))))
        axes[0].barh(df["config"], df["f1"],
                     color="white", edgecolor="black",
                     linewidth=1.0, hatch="///")
        axes[0].set_xlabel("F1-мера, отн. ед.\n"
                           "(F1-score, rel. units)")
        axes[0].set_ylabel("Конфигурация (Configuration)")
        axes[0].set_title("(а) F1-мера (F1-score)")
        axes[0].set_xlim(0, 1.05)
        for i, v in enumerate(df["f1"]):
            axes[0].text(v, i, f" {v:.4f}", va="center", fontsize=11)
        if "compromise_rate" in df.columns:
            axes[1].barh(df["config"], df["compromise_rate"],
                         color="white", edgecolor="black",
                         linewidth=1.0, hatch="xxx")
            axes[1].set_xlabel("Полнота обнаружения компрометации, %\n"
                               "(Compromise detection recall, %)")
            axes[1].set_ylabel("Конфигурация (Configuration)")
            axes[1].set_title("(б) Полнота на атаках компрометации\n"
                              "(Recall on compromise attacks)")
            axes[1].set_xlim(0, 105)
            for i, v in enumerate(df["compromise_rate"]):
                axes[1].text(v, i, f" {v:.1f}%", va="center", fontsize=11)
        fig.suptitle("Рисунок 12 — Аблационное исследование\n"
                     "(Ablation study)", fontsize=15, y=1.02)
        fig.tight_layout()
        _save(fig, out_dir, "12_ablation", formats)


# ======================================================================
# Рис. 13. F1 от порога
# ======================================================================
def plot_threshold_curves(results_list, y_test, out_dir, formats):
    top = [r for r in _top_models_by_f1(results_list, k=4)
           if len(np.asarray(r["scores"])) == len(y_test)]
    if not top:
        return
    fig, ax = plt.subplots(figsize=(12, 7.5))
    letters = ["а", "б", "в", "г"]
    for i, r in enumerate(top):
        s = np.asarray(r["scores"])
        grid = np.linspace(s.min(), s.max(), 200)
        f1s = []
        for t in grid:
            yp = (s >= t).astype(int)
            tp = ((yp == 1) & (y_test == 1)).sum()
            fp = ((yp == 1) & (y_test == 0)).sum()
            fn = ((yp == 0) & (y_test == 1)).sum()
            p = tp / (tp + fp) if (tp + fp) else 0
            rec = tp / (tp + fn) if (tp + fn) else 0
            f1 = 2 * p * rec / (p + rec) if (p + rec) else 0
            f1s.append(f1)
        st = _style(r["name"])
        ax.plot(grid, f1s,
                label=f"{letters[i]} — {_ru(r['name'])}",
                color=st["c"], linestyle=st["ls"], linewidth=st["lw"],
                marker=st["m"], markevery=20, markersize=8,
                markerfacecolor="white", markeredgecolor="black")
    ax.set_xlabel("Порог, отн. ед.\n(Threshold, rel. units)")
    ax.set_ylabel("F1-мера, отн. ед.\n(F1-score, rel. units)")
    ax.set_title("Рисунок 13 — Зависимость F1-меры от порога для топ-моделей\n"
                 "(F1-score vs. threshold — top models)")
    ax.legend(edgecolor="black", loc="best")
    fig.tight_layout()
    _save(fig, out_dir, "13_threshold_curves", formats)


# ======================================================================
# Рис. 14. Калибровка
# ======================================================================
def plot_calibration(results_list, y_test, out_dir, formats):
    top = [r for r in _top_models_by_f1(results_list, k=4)
           if len(np.asarray(r["scores"])) == len(y_test)]
    if not top:
        return
    fig, ax = plt.subplots(figsize=(10, 8))
    curves = []
    for r in top:
        s = np.asarray(r["scores"], dtype=float)
        if s.max() > s.min():
            s = (s - s.min()) / (s.max() - s.min())
        bins = np.linspace(0, 1, 11)
        idx = np.clip(np.digitize(s, bins) - 1, 0, 9)
        mean_pred, frac_pos = [], []
        for b in range(10):
            m = (idx == b)
            if m.sum() > 5:
                mean_pred.append(s[m].mean())
                frac_pos.append(np.asarray(y_test)[m].mean())
        curves.append({"name": r["name"],
                       "x": np.array(mean_pred),
                       "y": np.array(frac_pos)})

    ideal_models = []
    for c in curves:
        if len(c["x"]) == 0:
            continue
        dev = np.max(np.abs(c["y"] - c["x"]))
        c["ideal"] = bool(dev < 0.05)
        if c["ideal"]:
            ideal_models.append(_ru(c["name"]))

    jitter_amounts = {c["name"]: 0.0 for c in curves}
    if len(ideal_models) > 1:
        rng = np.linspace(-0.02, 0.02, len(ideal_models))
        ideal_curves = [cc for cc in curves if cc["ideal"]]
        for i, c in enumerate(ideal_curves):
            jitter_amounts[c["name"]] = float(rng[i])

    letters = ["а", "б", "в", "г"]
    for i, c in enumerate(curves):
        st = _style(c["name"])
        j = jitter_amounts.get(c["name"], 0.0)
        label = f"{letters[i]} — {_ru(c['name'])}"
        if c.get("ideal"):
            label += " (совп. с диагональю / on diagonal)"
        ax.plot(c["x"], c["y"] + j,
                marker=st["m"], markersize=12,
                label=label, color=st["c"], linestyle=st["ls"],
                linewidth=st["lw"], markeredgecolor="black",
                markerfacecolor="white", markeredgewidth=1.0)

    ax.plot([0, 1], [0, 1], "k:", alpha=0.7, linewidth=1.5,
            label="идеальная калибровка (perfect calibration)")
    ax.set_xlabel("Средняя предсказанная оценка (нормированная), отн. ед.\n"
                  "(Mean predicted score, normalized, rel. units)")
    ax.set_ylabel("Доля истинно положительных, отн. ед.\n"
                  "(Fraction of positives, rel. units)")
    ax.set_xlim(-0.03, 1.05)
    ax.set_ylim(-0.05, 1.07)
    ax.set_title("Рисунок 14 — Диаграмма надёжности (калибровка)\n"
                 "(Reliability diagram)")
    ax.legend(edgecolor="black", loc="upper left")
    fig.tight_layout()
    _save(fig, out_dir, "14_calibration", formats)


# ======================================================================
# Рис. 15. F1 vs FPR
# ======================================================================
def plot_sip_specific(results_list, out_dir, formats, y_test=None):
    rows = []
    for r in results_list:
        if r.get("scores") is None:
            continue
        f1_used = r.get("f1", 0)
        fpr_used = r.get("fpr", 0)
        scores = r.get("scores")
        if (y_test is not None and scores is not None
                and len(scores) == len(y_test) and fpr_used > 0.5):
            th_new, _ = _maxf1_threshold(scores, y_test)
            f1_used, fpr_used = _recompute_metrics(scores, y_test, th_new)
        rows.append({
            "model": _ru_block(r["name"]),
            "F1": f1_used,
            "FPR": fpr_used,
            "AUC-PR": r.get("auc_pr", 0),
        })
    if not rows:
        return
    df = pd.DataFrame(rows).sort_values("F1", ascending=True)
    fig, ax = plt.subplots(figsize=(13, 9))
    sc = ax.scatter(df["FPR"], df["F1"], c=df["AUC-PR"],
                    cmap="Greys", s=180, edgecolor="black", linewidth=1.4,
                    vmin=0, vmax=1)
    _spread_labels(ax, df["FPR"].values, df["F1"].values,
                   df["model"].values, min_dist=0.035)
    ax.set_xlabel("Доля ложных срабатываний FPR, отн. ед.\n"
                  "(False Positive Rate, rel. units)")
    ax.set_ylabel("F1-мера, отн. ед.\n(F1-score, rel. units)")
    ax.set_xlim(-0.02, max(df["FPR"].max() * 1.15, 0.1))
    ax.set_ylim(-0.05, 1.07)
    ax.set_title("Рисунок 15 — Соотношение F1-меры и FPR по моделям;\n"
                 "оттенок маркера — значение AUC-PR\n"
                 "(F1 vs. FPR; marker shade denotes AUC-PR)")
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label("AUC-PR, отн. ед.\n(rel. units)")
    fig.tight_layout()
    _save(fig, out_dir, "15_sip_specific", formats)


# ======================================================================
# Рис. 16. История обучения — фикс. сетка 2x2
# ======================================================================
def plot_training_history(trainers_history, out_dir, formats):
    if not trainers_history:
        return
    names = list(trainers_history.keys())

    # if len(names) > 4:
    #     print(f"[plot_training_history] предупреждение: получено "
    #           f"{len(names)} моделей, отображаются первые 4 "
    #           f"(сетка 2×2 фиксирована).")
    #     names = names[:4]

    priority = ["Autoencoder", "VAE", "LSTM-AE", "ZCAD"]
    names = [n for n in priority if n in trainers_history]
    names += [n for n in trainers_history if n not in names]
    names = names[:4]


    rows_n, cols_n = 2, 2
    fig, axes = plt.subplots(rows_n, cols_n, figsize=(13, 9.5))
    letters = ["а", "б", "в", "г"]

    for idx in range(rows_n * cols_n):
        ax = axes[idx // cols_n, idx % cols_n]
        if idx >= len(names):
            ax.axis("off")
            continue

        name = names[idx]
        h = trainers_history[name]
        tl = np.asarray(h.get("train_losses", []), dtype=float)
        vl = np.asarray(h.get("val_losses", []), dtype=float)

        all_losses = np.concatenate([tl, vl]) if (len(tl) and len(vl)) else \
                     (tl if len(tl) else vl)
        if len(all_losses) > 5:
            ymax = np.percentile(all_losses, 98) * 1.1
            ymin = max(0, np.percentile(all_losses, 1) * 0.9)
        else:
            ymax = float(all_losses.max()) if len(all_losses) else 1.0
            ymin = 0.0

        use_log = False
        if len(all_losses) > 5 and all_losses[all_losses > 0].size > 0:
            pos = all_losses[all_losses > 0]
            if pos.max() / (pos.min() + 1e-12) > 100:
                use_log = True

        if len(tl):
            ax.plot(tl, color="black", linestyle="-", linewidth=1.8,
                    marker="o", markevery=max(1, len(tl) // 8),
                    markerfacecolor="white", markeredgecolor="black",
                    markersize=6, label="обучающая (train)")
        if len(vl):
            ax.plot(vl, color="#555555", linestyle="--", linewidth=1.8,
                    marker="s", markevery=max(1, len(vl) // 8),
                    markerfacecolor="white", markeredgecolor="black",
                    markersize=6, label="валидационная (validation)")
        if use_log:
            ax.set_yscale("log")
            ax.set_ylabel("Функция потерь (лог. шкала), отн. ед.\n"
                          "(Loss, log scale, rel. units)")
        else:
            ax.set_ylim(ymin, ymax)
            ax.set_ylabel("Функция потерь, отн. ед.\n"
                          "(Loss, rel. units)")
        ax.set_title(f"({letters[idx]}) {name}", fontsize=13)
        ax.set_xlabel("Номер эпохи (Epoch)")
        ax.legend(edgecolor="black")
        ax.grid(True, alpha=0.35, linestyle=":")
    fig.tight_layout()
    _save(fig, out_dir, "16_training_history", formats)


# ======================================================================
# Рис. 17. Компонентная абляция ZCAD
# ======================================================================
def plot_zc_msd_ablation(zc_ablation_results, out_dir, formats):
    if not zc_ablation_results:
        return
    df = pd.DataFrame(zc_ablation_results)
    if "config" not in df.columns or "f1" not in df.columns:
        print("[plot_zc_msd_ablation] пропуск: нет полей config/f1.")
        return

    def _translate_zc(cfg: str) -> str:
        c = str(cfg).lower().replace("\n", " ").strip()
        if "полн" in c or "full" in c:
            return "Полная ZCAD\n(zone-attn + contrastive\n+ stability)"
        if "contrastive" in c:
            return "Без contrastive-\nфункции потерь"
        if "stability" in c:
            return "Без stability-\nагрегации"
        if "attn" in c or "внимани" in c or "zone-attn" in c:
            return "Только zone-attention\n(без contrastive/stability)"
        return str(cfg)

    df["config_ru"] = df["config"].map(_translate_zc)

    recall_col = None
    for cand in ("compromise_recall", "compromise_rate", "recall"):
        if cand in df.columns:
            recall_col = cand
            break
    if recall_col is not None:
        rec_vals = df[recall_col].astype(float).values
        if np.nanmax(rec_vals) <= 1.0 + 1e-9:
            df["_recall_pct"] = rec_vals * 100.0
        else:
            df["_recall_pct"] = rec_vals

    df = df.sort_values("f1")

    has_recall = "_recall_pct" in df.columns
    ncols = 2 if has_recall else 1
    fig, axes = plt.subplots(
        1, ncols, figsize=(8.5 * ncols, max(4.5, 0.95 * len(df) + 1.5)))
    if ncols == 1:
        axes = [axes]

    axes[0].barh(df["config_ru"], df["f1"],
                 color="white", edgecolor="black",
                 linewidth=1.0, hatch="\\|/")
    axes[0].set_xlabel("F1-мера, отн. ед.\n(F1-score, rel. units)")
    axes[0].set_ylabel("Конфигурация модели (Configuration)")
    axes[0].set_title("(а) F1-мера (F1-score)")
    axes[0].set_xlim(0, 1.05)
    for i, v in enumerate(df["f1"]):
        axes[0].text(v, i, f" {v:.4f}", va="center", fontsize=11)

    if has_recall:
        axes[1].barh(df["config_ru"], df["_recall_pct"],
                     color="white", edgecolor="black",
                     linewidth=1.0, hatch="xxx")
        axes[1].set_xlabel("Полнота обнаружения компрометации, %\n"
                           "(Proxy-compromise recall, %)")
        axes[1].set_ylabel("Конфигурация модели (Configuration)")
        axes[1].set_title("(б) Полнота на компрометации прокси\n"
                          "(Recall on proxy-compromise)")
        axes[1].set_xlim(0, 105)
        for i, v in enumerate(df["_recall_pct"]):
            axes[1].text(v, i, f" {v:.1f}%", va="center", fontsize=11)

    fig.suptitle(
        "Рисунок 17 — Компонентная абляция модели ZCAD\n"
        "(вклад zone-attention, contrastive-функции потерь и "
        "stability-агрегации)\n"
        "(ZCAD component ablation)",
        fontsize=15, y=1.03)
    fig.tight_layout()
    _save(fig, out_dir, "17_zcad_ablation", formats)


# ======================================================================
# Рис. 18. Межгрупповая корреляционная матрица (8×8) — ЧИТАЕМАЯ
# ======================================================================
def plot_group_correlation_matrix(dataset_path, feature_cols, out_dir,
                                   formats):
    """
    Рис. 18. Средняя абсолютная корреляция Пирсона МЕЖДУ функциональными
    группами признаков (агрегированная матрица размера G×G).

    По диагонали — средняя |корреляция| внутри группы (внутренняя
    связность), вне диагонали — между группами. Низкие внедиагональные
    значения доказывают, что группы (в т.ч. транзакционная целостность)
    несут НЕЗАВИСИМУЮ информацию, а не дублируют друг друга.
    """
    if dataset_path is None or not os.path.exists(dataset_path):
        print(f"[plot_group_correlation_matrix] пропуск: датасет не найден "
              f"({dataset_path}).")
        return
    if not feature_cols:
        return

    df = pd.read_csv(dataset_path)
    cols = [c for c in feature_cols if c in df.columns]
    if len(cols) < 2:
        return

    group_of = {c: _feature_group(c) for c in cols}
    groups = [g for g in FEATURE_GROUP_ORDER
              if any(group_of[c] == g for c in cols)]

    # Полная матрица корреляций (по абсолютной величине).
    corr = df[cols].corr(method="pearson").abs().fillna(0.0)

    G = len(groups)
    mat = np.zeros((G, G))
    for i, gi in enumerate(groups):
        cols_i = [c for c in cols if group_of[c] == gi]
        for j, gj in enumerate(groups):
            cols_j = [c for c in cols if group_of[c] == gj]
            sub = corr.loc[cols_i, cols_j].values
            if i == j:
                # средняя |корр| внутри группы, исключая диагональ (=1)
                if len(cols_i) > 1:
                    off = sub[~np.eye(len(cols_i), dtype=bool)]
                    mat[i, j] = float(off.mean())
                else:
                    mat[i, j] = 1.0
            else:
                mat[i, j] = float(sub.mean())

    labels = [FEATURE_GROUP_RU.get(g, g) for g in groups]

    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(mat, cmap="Greys", vmin=0, vmax=1, aspect="equal")

    ax.set_xticks(range(G))
    ax.set_yticks(range(G))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=11)
    ax.set_yticklabels(labels, fontsize=11)

    # Значения в клетках крупным шрифтом.
    for i in range(G):
        for j in range(G):
            v = mat[i, j]
            color = "white" if v > 0.5 else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=13, fontweight="bold", color=color)

    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Средняя |корреляция Пирсона|, отн. ед.\n"
                 "(Mean |Pearson correlation|, rel. units)")

    ax.set_title("Рисунок 18 — Средняя корреляция между функциональными\n"
                 "группами признаков (диагональ — связность внутри группы)\n"
                 "(Mean inter-group feature correlation)",
                 fontsize=14, pad=16)
    fig.tight_layout()
    _save(fig, out_dir, "18_group_correlation_matrix", formats)


# ======================================================================
# Рис. 18-прил. Полная матрица признаков (для ПРИЛОЖЕНИЯ) — крупная
# ======================================================================
def plot_feature_correlation_full(dataset_path, feature_cols, out_dir,
                                   formats, max_features=28):
    """
    Полностраничная матрица корреляций отдельных признаков для приложения.
    Читаемость обеспечивается: (1) жёстким ограничением до max_features
    (top по дисперсии внутри групп), (2) увеличенным размером клетки,
    (3) подписями признаков ТОЛЬКО по нижней оси, (4) групповыми
    блоками-скобками слева вместо мелких подписей на каждой строке.
    """
    if dataset_path is None or not os.path.exists(dataset_path):
        print("[plot_feature_correlation_full] пропуск: датасет не найден.")
        return
    if not feature_cols:
        return

    df = pd.read_csv(dataset_path)
    cols = [c for c in feature_cols if c in df.columns]
    if len(cols) < 2:
        return

    ordered, group_of, bounds = _grouped_feature_order(cols)

    # Отбор наиболее вариативных признаков в каждой группе.
    if len(ordered) > max_features:
        keep = []
        per_group = max(2, max_features // max(1, len(bounds)))
        for g, s, e in bounds:
            grp_cols = ordered[s:e]
            var = df[grp_cols].var(numeric_only=True).sort_values(
                ascending=False)
            keep.extend(list(var.index[:per_group]))
        ordered = [c for c in ordered if c in keep]
        ordered, group_of, bounds = _grouped_feature_order(ordered)

    corr = df[ordered].corr(method="pearson").fillna(0.0).values
    n = len(ordered)

    # Крупная фигура: ~0.55 дюйма на клетку.
    fig, ax = plt.subplots(figsize=(0.55 * n + 6, 0.55 * n + 5))
    im = ax.imshow(corr, cmap="RdGy_r", vmin=-1, vmax=1, aspect="equal")

    # Подписи признаков только снизу (по X), крупнее.
    ax.set_xticks(range(n))
    ax.set_xticklabels(
        [c.replace("obs_", "").replace("_", " ") for c in ordered],
        rotation=90, fontsize=9)
    ax.set_yticks(range(n))
    ax.set_yticklabels(
        [c.replace("obs_", "").replace("_", " ") for c in ordered],
        fontsize=9)

    # Разделители групп + подписи групп слева (крупные, вертикально).
    for g, s, e in bounds:
        ax.axhline(e - 0.5, color="black", linewidth=1.6)
        ax.axvline(e - 0.5, color="black", linewidth=1.6)
        center = (s + e - 1) / 2.0
        ax.text(-3.5, center, FEATURE_GROUP_RU.get(g, g),
                ha="right", va="center", fontsize=11, fontweight="bold",
                rotation=90)

    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)

    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Коэффициент корреляции Пирсона, отн. ед.\n"
                 "(Pearson correlation, rel. units)")

    ax.set_title("Приложение — Полная матрица корреляций признаков\n"
                 "(Full feature correlation matrix)",
                 fontsize=14, pad=20)
    fig.tight_layout()
    _save(fig, out_dir, "18b_feature_correlation_full", formats)


# ======================================================================
# Рис. 19. Важность признаков как heatmap по группам — НОВОЕ
# ======================================================================
def plot_group_importance_heatmap(importances, feature_cols, out_dir,
                                  formats, top_per_group=6):
    """
    Рис. 19. Тепловая карта важности признаков (Random Forest),
    сгруппированных по функциональным группам. Строки — группы,
    столбцы — top_per_group наиболее важных признаков внутри группы;
    значения — нормированная важность (доля от максимума). Также
    строится боковая панель суммарной важности каждой группы.
    """
    if importances is None or len(importances) == 0 or not feature_cols:
        print("[plot_group_importance_heatmap] пропуск: нет важностей.")
        return
    imp = np.asarray(importances, dtype=float)
    if len(imp) != len(feature_cols):
        print("[plot_group_importance_heatmap] пропуск: длины важностей и "
              "feature_cols не совпадают.")
        return

    df = pd.DataFrame({"feature": feature_cols, "importance": imp})
    df["group"] = df["feature"].map(_feature_group)

    # Порядок групп по суммарной важности (по убыванию).
    group_sum = df.groupby("group")["importance"].sum().to_dict()
    groups_present = [g for g in FEATURE_GROUP_ORDER
                      if g in group_sum]
    # добавим нестандартные группы (если есть)
    for g in df["group"].unique():
        if g not in groups_present:
            groups_present.append(g)
    groups_present.sort(key=lambda g: group_sum.get(g, 0.0), reverse=True)

    # Матрица: строки — группы, столбцы — top_per_group признаков.
    heat = np.full((len(groups_present), top_per_group), np.nan)
    cell_labels = [["" for _ in range(top_per_group)]
                   for _ in range(len(groups_present))]
    global_max = df["importance"].max() + 1e-12

    for gi, g in enumerate(groups_present):
        sub = df[df["group"] == g].sort_values("importance",
                                               ascending=False)
        for j in range(min(top_per_group, len(sub))):
            row = sub.iloc[j]
            heat[gi, j] = row["importance"] / global_max
            # короткая подпись признака внутри ячейки
            short = row["feature"].replace("obs_", "").replace("_", " ")
            if len(short) > 16:
                short = short[:15] + "…"
            cell_labels[gi][j] = f"{short}\n{row['importance']:.3f}"

    fig, (ax, axb) = plt.subplots(
        1, 2, figsize=(3.0 * top_per_group + 5, 0.9 * len(groups_present) + 3),
        gridspec_kw={"width_ratios": [top_per_group, 1.4]})

    im = ax.imshow(np.nan_to_num(heat, nan=0.0), cmap="Greys",
                   vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(top_per_group))
    ax.set_xticklabels([f"#{j+1}" for j in range(top_per_group)])
    ax.set_yticks(range(len(groups_present)))
    ax.set_yticklabels([FEATURE_GROUP_RU.get(g, g).replace("\n", " ")
                        for g in groups_present])
    for gi in range(len(groups_present)):
        for j in range(top_per_group):
            if not cell_labels[gi][j]:
                continue
            val = heat[gi, j]
            color = "white" if (val is not None and val > 0.55) else "black"
            ax.text(j, gi, cell_labels[gi][j], ha="center", va="center",
                    fontsize=8, color=color)
    ax.set_xlabel("Ранг признака внутри группы\n"
                  "(Feature rank within group)")
    ax.set_ylabel("Функциональная группа\n(Functional group)")
    ax.set_title("(а) Важность top-признаков по группам\n"
                 "(Top-feature importance per group)", fontsize=13)
    cb = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("Важность (норм. к макс.), отн. ед.\n"
                 "(Importance, normalized, rel. units)")

    # Боковая панель — суммарная важность групп (в процентах).
    sums = np.array([group_sum.get(g, 0.0) for g in groups_present])
    total = sums.sum() + 1e-12
    sums_pct = 100.0 * sums / total
    ypos = range(len(groups_present))
    axb.barh(list(ypos), sums_pct, color="white", edgecolor="black",
             linewidth=1.0, hatch="///")
    axb.set_yticks(list(ypos))
    axb.set_yticklabels([])
    axb.invert_yaxis()
    axb.set_xlabel("Суммарная важность группы, %\n"
                   "(Group total importance, %)")
    axb.set_title("(б) Вклад группы\n(Group share)", fontsize=13)
    axb.set_xlim(0, max(sums_pct.max() * 1.25, 5))
    for i, v in enumerate(sums_pct):
        axb.text(v, i, f" {v:.1f}%", va="center", fontsize=10)

    # выравниваем оси y обеих панелей
    ax.set_ylim(len(groups_present) - 0.5, -0.5)

    fig.suptitle(
        "Рисунок 19 — Важность признаков по функциональным группам "
        "(Random Forest)\n"
        "(Feature importance heatmap by functional groups)",
        fontsize=15, y=1.02)
    fig.tight_layout()
    _save(fig, out_dir, "19_group_importance_heatmap", formats)


# ----------------------------------------------------------------------
# Вспомогательная: упорядочивание признаков по функциональным группам
# (используется Рис. 18-прил. и др.)
# ----------------------------------------------------------------------
def _grouped_feature_order(feature_cols):
    """
    Возвращает (ordered_cols, group_of_col, group_bounds), где признаки
    упорядочены по функциональным группам согласно FEATURE_GROUP_ORDER.
    group_bounds — список (group_key, start_idx, end_idx) для разделителей.
    """
    group_of = {c: _feature_group(c) for c in feature_cols}
    ordered = []
    bounds = []
    for g in FEATURE_GROUP_ORDER:
        cols_g = [c for c in feature_cols if group_of[c] == g]
        if not cols_g:
            continue
        start = len(ordered)
        ordered.extend(cols_g)
        bounds.append((g, start, len(ordered)))
    # Признаки без назначенной (нестандартной) группы — в конец.
    leftover = [c for c in feature_cols if c not in ordered]
    if leftover:
        start = len(ordered)
        ordered.extend(leftover)
        bounds.append(("other", start, len(ordered)))
    return ordered, group_of, bounds


# ======================================================================
# Главная точка входа — структура как в Diameter-версии
# ======================================================================
def render_all_plots(artifacts_path,
                     plots_dir="results_sip_v1/plots_print",
                     formats=("png", "svg", "pdf"),
                     dataset_path=None):
    with open(artifacts_path, "rb") as f:
        art = pickle.load(f)

    results_list      = art["results_list"]
    ens_results       = art.get("ens_results", [])
    ablation_results  = art.get("ablation_results", [])
    y_test            = np.asarray(art["y_test"])
    attack_types_test = np.asarray(art["attack_types_test"])
    importances       = art.get("importances", np.array([]))
    feature_cols      = art.get("feature_cols", [])
    cascade_s1        = art.get("cascade_s1")
    cascade_s2        = art.get("cascade_s2")
    cascade_comb      = art.get("cascade_comb")
    trainers_history  = art.get("trainers_history", {})
    zc_ablation_results = (art.get("zc_ablation")
                           or art.get("zc_ablation_results")
                           or art.get("zc_msd_ablation")
                           or [])

    # Путь к исходному датасету для корреляционной матрицы (Рис. 18).
    # По умолчанию ищем рядом с plot_artifacts.pkl, затем — стандартный путь.
    if dataset_path is None:
        cand = [
            os.path.join(os.path.dirname(artifacts_path),
                         "sip_dataset_v3.csv"),
            "project_300626_1439_imp_zc-msd/sip_dataset_v3.csv",
        ]
        dataset_path = next((p for p in cand if os.path.exists(p)), None)

    _ensure_dir(plots_dir)

    plot_roc_curves(results_list, y_test, plots_dir, formats)
    plot_roc_zoom(results_list, y_test, plots_dir, formats)
    plot_pr_curves(results_list, y_test, plots_dir, formats)
    plot_metric_bars(results_list, y_test, plots_dir, formats)
    plot_confusion_matrices(results_list, y_test, plots_dir, formats, k=4)
    plot_per_attack_recall(results_list, attack_types_test, y_test,
                           plots_dir, formats)
    plot_feature_importance(importances, feature_cols, plots_dir, formats)
    plot_score_distributions(results_list, y_test, plots_dir, formats, k=3)
    plot_cascade_analysis(results_list, cascade_s1, cascade_s2, cascade_comb,
                          y_test, plots_dir, formats)
    plot_compromise_comparison(results_list, attack_types_test, y_test,
                               plots_dir, formats)
    plot_ensemble_blocks(ens_results, attack_types_test, plots_dir, formats)
    plot_ablation(ablation_results, plots_dir, formats)
    plot_threshold_curves(results_list, y_test, plots_dir, formats)
    plot_calibration(results_list, y_test, plots_dir, formats)
    plot_sip_specific(results_list, plots_dir, formats, y_test=y_test)
    plot_training_history(trainers_history, plots_dir, formats)
    plot_zc_msd_ablation(zc_ablation_results, plots_dir, formats)
    # ── НОВЫЕ ГРАФИКИ ──
        # ── НОВЫЕ ГРАФИКИ ──
    plot_group_correlation_matrix(dataset_path, feature_cols,
                                  plots_dir, formats)          # Рис. 18 (8×8)
    plot_feature_correlation_full(dataset_path, feature_cols,
                                  plots_dir, formats)          # приложение
    plot_group_importance_heatmap(importances, feature_cols, plots_dir,
                                  formats)                     # Рис. 19


    print(f"[plot_results_print_sip] печатные графики сохранены: "
          f"{plots_dir}")


if __name__ == "__main__":
    import sys
    art_path = sys.argv[1] if len(sys.argv) > 1 else \
        "project_300626_1439_imp_zc-msd/results_sip_v3.1/plot_artifacts.pkl"
    out_dir  = sys.argv[2] if len(sys.argv) > 2 else \
        "project_300626_1439_imp_zc-msd/results_sip_v3.1/plots_print"
    ds_path  = sys.argv[3] if len(sys.argv) > 3 else None
    render_all_plots(art_path, plots_dir=out_dir, dataset_path=ds_path)
