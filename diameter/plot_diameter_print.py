"""
plot_diameter_print.py — печатная графика Diameter в едином стиле.
==================================================================
Ничего не обучает. Читает артефакты, сохранённые ms_multiseed_core
(run_one_seed → artifacts_seed*.pkl, run_seeds → all_seed_metrics.csv,
significance.csv), и строит все фигуры в одном grayscale-стиле для печати
(600 dpi, sans-serif, штриховки вместо цвета, PNG + PDF + SVG).

Фигуры 01-14 — набор, соответствующий ResultsVisualizer диаметр-пайплайна.
Фигуры 15-18 — мультисид: среднее ± СКО, разброс по сидам, Уилкоксон.
Плюс LaTeX-таблицы table_mean_std.tex и table_wilcoxon.tex.

Заголовки фигур по умолчанию ОТКЛЮЧЕНЫ (подпись даётся в тексте статьи);
включить обратно: --titles

Использование:
    python plot_diameter_print.py --run diameter_seed_runs/seed_42
    python plot_diameter_print.py --seeds-dir diameter_seed_runs
    python plot_diameter_print.py --seeds-dir diameter_seed_runs \
                                  --data diameter_dataset_v1.csv
    python plot_diameter_print.py --run diameter_seed_runs/seed_42 \
                                  --out figs --formats png pdf --titles
"""

import os
import glob
import pickle
import argparse
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, PercentFormatter
from matplotlib.patches import Patch

from sklearn.metrics import roc_curve, precision_recall_curve, auc

import warnings
warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════
#  1. ЕДИНЫЙ ПЕЧАТНЫЙ СТИЛЬ (идентичен SS7-версии)
# ══════════════════════════════════════════════════════════════════════

DPI = 600
FORMATS = ["png", "pdf", "svg"]
W1, W2 = 3.45, 7.10          # ширина колонки / страницы, дюймы
ROC_YMAX = 1.02              # >1.0, чтобы кривые с AUC≈1 не сливались с рамкой

TITLES = False               # переключается ключом --titles


def apply_style():
    plt.rcParams.update({
        # шрифт по умолчанию (обычный, без засечек)
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Liberation Sans",
                            "Helvetica"],
        "mathtext.fontset": "dejavusans",
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.labelsize": 9.0,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.0,
        "figure.titlesize": 10.5,
        "axes.linewidth": 0.7,
        "axes.edgecolor": "#000000",
        "axes.facecolor": "white",
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "#c8c8c8",
        "grid.linewidth": 0.4,
        "grid.linestyle": ":",
        "lines.linewidth": 1.3,
        "lines.markersize": 3.6,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "legend.frameon": True,
        "legend.framealpha": 1.0,
        "legend.edgecolor": "#444444",
        "legend.fancybox": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "hatch.linewidth": 0.55,
    })


MODEL_STYLE = {
    "Isolation Forest":       dict(c="#111111", ls="-",               m="o", h="////"),
    "One-Class SVM":          dict(c="#2a2a2a", ls="--",              m="s", h="\\\\\\\\"),
    "Autoencoder":            dict(c="#3f3f3f", ls="-.",              m="^", h="...."),
    "VAE":                    dict(c="#555555", ls=":",               m="D", h="xxxx"),
    "LSTM-AE":                dict(c="#6b6b6b", ls=(0, (5, 1)),       m="v", h="++++"),
    "Temporal-Attention-AE":  dict(c="#808080", ls=(0, (7, 2)),       m="<", h="||||"),
    "Random Forest":          dict(c="#949494", ls="-",               m=">", h="----"),
    "Gradient Boosting":      dict(c="#aaaaaa", ls="--",              m="P", h="oooo"),
    "Cascade":                dict(c="#000000", ls="-",               m="*", h="**"),
    "Cascade-Stage1":         dict(c="#5f5f5f", ls=(0, (3, 1, 1, 1)), m="1", h="//"),
    "Cascade-Stage2":         dict(c="#8c8c8c", ls=(0, (3, 1, 1, 1)), m="2", h="\\\\"),
    "Meta-Ensemble":          dict(c="#202020", ls=(0, (4, 1, 1, 1)), m="X", h="xx"),
    "RF (no MS)":             dict(c="#b5b5b5", ls=":",               m="h", h=".."),
    "RF (no temporal)":       dict(c="#c9c9c9", ls=":",               m="p", h="oo"),
}
_FB_C = ["#1a1a1a", "#4d4d4d", "#7a7a7a", "#a3a3a3", "#c4c4c4"]
_FB_LS = ["-", "--", "-.", ":", (0, (6, 2))]
_FB_H = ["////", "....", "xxxx", "++++", "||||"]


def st(name, i=0):
    if name in MODEL_STYLE:
        return MODEL_STYLE[name]
    return dict(c=_FB_C[i % 5], ls=_FB_LS[i % 5], m="o", h=_FB_H[i % 5])


MAIN_ORDER = ["Isolation Forest", "One-Class SVM", "Autoencoder", "VAE",
              "LSTM-AE", "Temporal-Attention-AE", "Random Forest",
              "Gradient Boosting", "Cascade", "Meta-Ensemble"]

MODEL_LABEL = {
    "Isolation Forest": "Isolation Forest",
    "One-Class SVM": "One-Class SVM",
    "Autoencoder": "Автокодировщик",
    "VAE": "VAE",
    "LSTM-AE": "LSTM-АК",
    "Temporal-Attention-AE": "Temporal-Attn-АК",
    "Random Forest": "Случайный лес",
    "Gradient Boosting": "Град. бустинг",
    "Cascade": "Каскад",
    "Cascade-Stage1": "Каскад, ур. 1",
    "Cascade-Stage2": "Каскад, ур. 2",
    "Meta-Ensemble": "Мета-ансамбль",
    "RF (no MS)": "СЛ без MS-призн.",
    "RF (no temporal)": "СЛ без temporal",
}

METRIC_LABEL = {
    "auc_roc": "AUC-ROC", "auc_pr": "AUC-PR", "f1": "F1-мера", "fpr": "FPR",
    "compromise_recall": "Полнота, компрометация узла, %",
    "recall_at_1pct_fpr": "Полнота при FPR = 1 %, %",
}


# ══════════════════════════════════════════════════════════════════════
#  2. СПЕЦИФИКА DIAMETER (атаки определяются автоматически)
# ══════════════════════════════════════════════════════════════════════

COMPROMISE = "node_compromise"

ATTACK_ORDER = ["location_tracking", "sms_data_interception",
                "signaling_dos", "fraud_profile", "node_compromise"]

ATTACK_LABEL = {
    "location_tracking":      "Слежение\nза локацией",
    "sms_data_interception":  "Перехват SMS\nи данных",
    "signaling_dos":          "Сигнальный\nDoS",
    "fraud_profile":          "Мошеннический\nпрофиль",
    "node_compromise":        "Компрометация\nузла",
    "location_track":         "Слежение\nза локацией",
    "sms_intercept":          "Перехват SMS",
    "subscriber_dos":         "DoS абонента",
    "irsf":                   "IRSF",
    "fraud":                  "Мошенничество",
}
ATTACK_SHORT = {
    "location_tracking": "Локация", "sms_data_interception": "Перехват",
    "signaling_dos": "DoS", "fraud_profile": "Фрод",
    "node_compromise": "Компромет.", "location_track": "Локация",
    "sms_intercept": "Перехват", "subscriber_dos": "DoS абон.",
    "irsf": "IRSF", "fraud": "Фрод",
}
ATTACK_GRAY = ["#3d3d3d", "#5c5c5c", "#7a7a7a", "#9a9a9a", "#111111",
               "#b8b8b8", "#282828"]
ATTACK_HATCH = ["////", "..", "xx", "\\\\", "++", "||", "oo"]


def alabel(a):
    return ATTACK_LABEL.get(a, str(a).replace("_", "\n"))


def ashort(a):
    return ATTACK_SHORT.get(a, str(a).replace("_", " ")[:11])


def agray(a, attacks):
    if a == COMPROMISE:
        return "#111111"
    return ATTACK_GRAY[attacks.index(a) % len(ATTACK_GRAY)] \
        if a in attacks else "#8c8c8c"


def detect_attacks(df):
    """Типы атак из колонок rec_* с сохранением осмысленного порядка."""
    found = [c[4:] for c in df.columns if str(c).startswith("rec_")]
    ordered = [a for a in ATTACK_ORDER if a in found]
    ordered += sorted(a for a in found if a not in ordered)
    if COMPROMISE in ordered:
        ordered = [a for a in ordered if a != COMPROMISE] + [COMPROMISE]
    return ordered


MS_FEATURES_FALLBACK = [
    "cu_delivered", "cu_rtt_delay", "cu_req_delay", "cu_resp_delay",
    "cu_combined_loss_prob", "integrity_check", "consecutive_cu_losses",
    "cu_processing_delay", "cu_response_time_jitter", "cu_staleness",
    "cu_consistency_error", "integrity_fail_rate",
]


def ms_features():
    """MS-признаки из diameter_training_v1, если модуль импортируется."""
    for mod, attr in [("diameter_training_v1", "MS_FEATURES"),
                      ("diameter_training_v1", "MASTER_SLAVE_FEATURES")]:
        try:
            m = __import__(mod)
            if hasattr(m, attr):
                return list(getattr(m, attr))
        except Exception:
            pass
    return MS_FEATURES_FALLBACK


def first_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


class Out:
    """Сохранение фигуры во все форматы + журнал."""

    def __init__(self, outdir, formats=FORMATS):
        self.dir, self.formats, self.saved = outdir, formats, []
        os.makedirs(outdir, exist_ok=True)

    def __call__(self, fig, name):
        for f in self.formats:
            fig.savefig(os.path.join(self.dir, f"{name}.{f}"),
                        format=f, dpi=DPI)
        plt.close(fig)
        self.saved.append(name)
        print(f"  ✓ {name} [{', '.join(self.formats)}]")


# ── заголовки: главный заголовок фигуры подавляется, панельный — остаётся ──

def _title(ax, text, **kw):
    """Главный заголовок одиночной оси (подавляется по умолчанию)."""
    if TITLES:
        ax.set_title(text, **kw)


def _suptitle(fig, text, **kw):
    if TITLES:
        fig.suptitle(text, **kw)


def _panel(ax, text, fs=8.5):
    """Подпись панели (имя модели/этапа) — нужна всегда для читаемости."""
    ax.set_title(text, fontsize=fs)


def _layout(fig, top=0.955, **kw):
    """tight_layout с запасом сверху только когда заголовок включён."""
    fig.tight_layout(rect=(0, 0, 1, top if TITLES else 1.0), **kw)


def _panel_tag(ax, tag, dx=-0.13, dy=1.06):
    ax.text(dx, dy, tag, transform=ax.transAxes, fontsize=9.5,
            fontweight="bold", va="top", ha="left")


def _bar_labels(ax, bars, fmt="{:.3f}", dy=0.008, fs=6.2, rot=0):
    for b in bars:
        h = b.get_height()
        if np.isfinite(h):
            ax.text(b.get_x() + b.get_width() / 2, h + dy, fmt.format(h),
                    ha="center", va="bottom", fontsize=fs, rotation=rot)


# ══════════════════════════════════════════════════════════════════════
#  3. ЗАГРУЗКА АРТЕФАКТОВ
# ══════════════════════════════════════════════════════════════════════

def load_run(run_dir):
    hits = sorted(glob.glob(os.path.join(run_dir, "artifacts_seed*.pkl")))
    if not hits:
        raise FileNotFoundError(f"нет artifacts_seed*.pkl в {run_dir}")
    with open(hits[0], "rb") as f:
        a = pickle.load(f)
    a["results_df"] = pd.DataFrame(a["results"])
    a["attacks"] = detect_attacks(a["results_df"])
    print(f"[load] {hits[0]}: сид {a['seed']}, "
          f"{len(a['results'])} моделей, атаки: {', '.join(a['attacks'])}")
    return a


def load_seeds(seeds_dir):
    csv = os.path.join(seeds_dir, "all_seed_metrics.csv")
    if os.path.exists(csv):
        alld = pd.read_csv(csv)
    else:
        frames = [pd.read_csv(p) for p in sorted(glob.glob(
            os.path.join(seeds_dir, "seed_*", "metrics_seed*.csv")))]
        if not frames:
            raise FileNotFoundError(f"нет метрик по сидам в {seeds_dir}")
        alld = pd.concat(frames, ignore_index=True)
    sig_p = os.path.join(seeds_dir, "significance.csv")
    sig = pd.read_csv(sig_p) if os.path.exists(sig_p) else pd.DataFrame()
    print(f"[load] {seeds_dir}: {alld['seed'].nunique()} сидов, "
          f"{alld['name'].nunique()} моделей, {len(sig)} сравнений")
    return alld, sig


def present(a, order=MAIN_ORDER):
    return [m for m in order if m in a["model_scores"]]


def rows_by_name(df):
    return {r["name"]: r for _, r in df.iterrows()}


# ══════════════════════════════════════════════════════════════════════
#  4. ФИГУРЫ 01-14
# ══════════════════════════════════════════════════════════════════════

def fig01_roc(a, save):
    y = a["y_test"]
    fig, ax = plt.subplots(figsize=(W1 * 1.35, W1 * 1.05))
    for i, name in enumerate(present(a)):
        fpr, tpr, _ = roc_curve(y, a["model_scores"][name])
        k = st(name, i)
        ax.plot(fpr, tpr, color=k["c"], linestyle=k["ls"],
                lw=1.5 if name == "Cascade" else 1.1,
                label=f"{MODEL_LABEL.get(name, name)} ({auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], color="#999999", lw=0.7, ls=(0, (2, 3)),
            label="Случайный выбор (0.500)")
    ax.set_xlabel("Доля ложных срабатываний (FPR)")
    ax.set_ylabel("Доля верных обнаружений (TPR)")
    ax.set_xlim(0, 1); ax.set_ylim(0, ROC_YMAX)
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    _title(ax, "ROC-кривые детекторов аномалий Diameter")
    ax.legend(loc="lower right")
    _layout(fig)
    save(fig, "01_roc_curves")


def fig02_roc_zoom(a, save):
    y = a["y_test"]
    fig, ax = plt.subplots(figsize=(W1 * 1.35, W1 * 1.05))
    for i, name in enumerate(present(a)):
        fpr, tpr, _ = roc_curve(y, a["model_scores"][name])
        k = st(name, i)
        ax.plot(fpr, tpr, color=k["c"], linestyle=k["ls"],
                lw=1.5 if name == "Cascade" else 1.1,
                label=MODEL_LABEL.get(name, name))
    ax.axvline(0.01, color="#000000", lw=0.8, ls=(0, (1, 2)))
    # подпись отодвинута от линии и приподнята, чтобы не задевать кривые
    ax.text(0.0135, 0.10, "FPR = 1 %", fontsize=6.8, rotation=90,
            va="bottom", ha="left")
    ax.set_xlim(0, 0.10); ax.set_ylim(0, ROC_YMAX)
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax.set_xlabel("FPR (область низких ложных срабатываний)")
    ax.set_ylabel("TPR")
    _title(ax, "ROC-кривые, увеличенная область FPR < 10 %")
    ax.legend(loc="lower right")
    _layout(fig)
    save(fig, "02_roc_zoom")


def fig03_pr(a, save):
    y = a["y_test"]
    fig, ax = plt.subplots(figsize=(W1 * 1.35, W1 * 1.05))
    for i, name in enumerate(present(a)):
        pr, rc, _ = precision_recall_curve(y, a["model_scores"][name])
        k = st(name, i)
        ax.plot(rc, pr, color=k["c"], linestyle=k["ls"],
                lw=1.5 if name == "Cascade" else 1.1,
                label=MODEL_LABEL.get(name, name))
    base = float(np.mean(y))
    ax.axhline(base, color="#999999", lw=0.7, ls=(0, (2, 3)),
               label=f"Базовый уровень ({base:.3f})")
    ax.set_xlabel("Полнота (Recall)")
    ax.set_ylabel("Точность (Precision)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    _title(ax, "Кривые «точность–полнота»")
    ax.legend(loc="lower left")
    _layout(fig)
    save(fig, "03_precision_recall")


def _recall_matrix(df, models, attacks):
    M = np.full((len(models), len(attacks)), np.nan)
    r = rows_by_name(df)
    for i, m in enumerate(models):
        if m not in r:
            continue
        for j, atk in enumerate(attacks):
            v = r[m].get(f"rec_{atk}", np.nan)
            M[i, j] = v if np.isfinite(v) else np.nan
    return M


def _heatmap(ax, M, rowlabels, collabels, vmax=100, fmt="{:.1f}"):
    im = ax.imshow(M, cmap=plt.cm.Greys, vmin=0, vmax=vmax, aspect="auto")
    ax.set_xticks(range(M.shape[1]))
    ax.set_xticklabels(collabels, fontsize=7.2)
    ax.set_yticks(range(M.shape[0]))
    ax.set_yticklabels(rowlabels, fontsize=7.2)
    ax.tick_params(axis="both", length=0, pad=2)
    ax.grid(False)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if not np.isfinite(M[i, j]):
                ax.text(j, i, "—", ha="center", va="center", fontsize=6.5)
            else:
                ax.text(j, i, fmt.format(M[i, j]), ha="center", va="center",
                        fontsize=6.4,
                        color="white" if M[i, j] > 0.58 * vmax else "black")
    for sp in ax.spines.values():
        sp.set_linewidth(0.7)
    return im


def fig04_attack_heatmap(a, save):
    df, atks = a["results_df"], a["attacks"]
    models = [m for m in MAIN_ORDER if m in set(df["name"])]
    M = _recall_matrix(df, models, atks)
    fig, ax = plt.subplots(figsize=(W2 * 0.8, 0.34 * len(models) + 1.5))
    im = _heatmap(ax, M, [MODEL_LABEL.get(m, m) for m in models],
                  [alabel(x) for x in atks])
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("Полнота обнаружения, %", fontsize=8)
    cb.outline.set_linewidth(0.7)
    _title(ax, "Полнота обнаружения по типам атак Diameter")
    _layout(fig)
    save(fig, "04_attack_heatmap")


def fig05_summary_bars(a, save):
    df = a["results_df"]; r = rows_by_name(df)
    models = [m for m in MAIN_ORDER if m in set(df["name"])]
    metrics = [("auc_roc", "////"), ("auc_pr", "...."), ("f1", "xxxx")]
    grays = ["#2b2b2b", "#777777", "#bcbcbc"]
    x = np.arange(len(models)); w = 0.26
    fig, ax = plt.subplots(figsize=(W2, 2.9))
    for k, (met, hatch) in enumerate(metrics):
        b = ax.bar(x + (k - 1) * w, [r[m].get(met, np.nan) for m in models],
                   w, color=grays[k], hatch=hatch, edgecolor="black",
                   linewidth=0.6, label=METRIC_LABEL[met])
        _bar_labels(ax, b, "{:.3f}", dy=0.01, fs=5.6, rot=90)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABEL.get(m, m) for m in models],
                       rotation=30, ha="right")
    ax.set_ylabel("Значение метрики"); ax.set_ylim(0, 1.16)
    _title(ax, "Сводное сравнение качества детекторов")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.0))
    _layout(fig)
    save(fig, "05_summary_bars")


def fig06_feature_importance(a, save, top_n=20):
    imp = np.asarray(a.get("rf_importances", []), float)
    cols = list(a.get("feature_cols", []))
    if imp.size == 0 or len(cols) != imp.size:
        print("  [skip] 06: нет важностей признаков")
        return
    msf = set(ms_features())
    idx = np.argsort(imp)[::-1][:top_n][::-1]
    names, vals = [cols[i] for i in idx], imp[idx]
    is_ms = [n in msf for n in names]
    fig, ax = plt.subplots(figsize=(W1 * 1.4, 0.19 * len(names) + 1.0))
    ax.barh(np.arange(len(names)), vals,
            color=["#3a3a3a" if f else "#c2c2c2" for f in is_ms],
            hatch=["////" if f else "" for f in is_ms],
            edgecolor="black", linewidth=0.5)
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels([n.replace("_", " ") for n in names], fontsize=6.4)
    ax.set_xlabel("Важность по Джини")
    _title(ax, f"Топ-{top_n} признаков случайного леса (Diameter)")
    ax.legend(handles=[Patch(facecolor="#3a3a3a", hatch="////",
                             edgecolor="black", label="MS-признаки"),
                       Patch(facecolor="#c2c2c2", edgecolor="black",
                             label="Прочие признаки")], loc="lower right")
    share = 100 * vals[np.array(is_ms)].sum() / max(vals.sum(), 1e-12)
    ax.text(0.98, 0.02, f"Доля MS в топ-{top_n}: {share:.1f} %",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.6)
    _layout(fig)
    save(fig, "06_feature_importance")


def fig07_ablation(a, save):
    r = rows_by_name(a["results_df"])
    plan = [("Полный набор", "Random Forest"),
            ("Без MS-призн.", "RF (no MS)"),
            ("Без temporal", "RF (no temporal)"),
            ("Каскад: только ур. 1", "Cascade-Stage1"),
            ("Каскад: оба уровня", "Cascade")]
    plan = [(l, k) for l, k in plan if k in r]
    if len(plan) < 2:
        print("  [skip] 07: недостаточно аблаций")
        return
    labs = [p[0] for p in plan]
    f1 = [r[k].get("f1", np.nan) for _, k in plan]
    cr = [r[k].get("compromise_recall", np.nan) for _, k in plan]
    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.9))
    for ax, v, ttl, fmt, ymax in [
            (axes[0], f1, "Влияние на F1-меру", "{:.3f}", 1.16),
            (axes[1], cr, "Влияние на обнаружение компрометации",
             "{:.1f}", 118)]:
        b = ax.bar(range(len(v)), v, 0.62,
                   color=["#2b2b2b"] + ["#a8a8a8"] * (len(v) - 1),
                   hatch=["////"] + [".."] * (len(v) - 1),
                   edgecolor="black", linewidth=0.6)
        if np.isfinite(v[0]):
            ax.axhline(v[0], color="#000000", lw=0.7, ls=(0, (2, 2)))
        _bar_labels(ax, b, fmt, dy=0.01 * ymax, fs=6.4)
        ax.set_xticks(range(len(v)))
        # небольшой наклон вместо переноса строк — подписи не наезжают
        ax.set_xticklabels(labs, fontsize=6.8, rotation=18, ha="right")
        ax.set_ylim(0, ymax)
        _title(ax, ttl)
    axes[0].set_ylabel("F1-мера"); axes[1].set_ylabel("Полнота, %")
    _panel_tag(axes[0], "(а)"); _panel_tag(axes[1], "(б)")
    _layout(fig)
    save(fig, "07_ablation")


def fig08_score_distributions(a, save):
    y, at, atks = a["y_test"], np.asarray(a["attack_types"]), a["attacks"]
    want = [m for m in ["VAE", "Random Forest", "Cascade"]
            if m in a["model_scores"]]
    if not want:
        print("  [skip] 08: нет оценок моделей")
        return
    show = [x for x in [COMPROMISE] + atks if x in atks][:3]
    fig, axes = plt.subplots(1, len(want), figsize=(W2, 2.5))
    axes = np.atleast_1d(axes)
    for ax, name in zip(axes, want):
        s = np.nan_to_num(a["model_scores"][name], nan=0.0)
        bins = np.linspace(np.percentile(s, 0.5), np.percentile(s, 99.5), 45)
        ax.hist(s[y == 0], bins=bins, density=True, color="#d9d9d9",
                edgecolor="#4a4a4a", linewidth=0.4, label="Норма")
        for atk in show:
            m = (at == atk)
            if m.sum() < 10:
                continue
            ax.hist(s[m], bins=bins, density=True, histtype="step", lw=1.2,
                    color=agray(atk, atks),
                    linestyle="-" if atk == COMPROMISE else "--",
                    label=ashort(atk))
        _panel(ax, MODEL_LABEL.get(name, name), fs=8.8)
        ax.set_xlabel("Оценка аномальности"); ax.set_yscale("log")
    axes[0].set_ylabel("Плотность (лог. шкала)")
    axes[0].legend(loc="upper center", ncol=2, fontsize=6.2)
    for ax, t in zip(axes, ["(а)", "(б)", "(в)"]):
        _panel_tag(ax, t)
    _suptitle(fig, "Распределения оценок аномальности по классам трафика")
    _layout(fig)
    save(fig, "08_score_distributions")


def fig09_confusion(a, save):
    """Матрицы ошибок: подписи истинных классов повёрнуты на 90°."""
    df = a["results_df"]; r = rows_by_name(df)
    models = [m for m in ["Autoencoder", "VAE", "Random Forest", "Cascade",
                          "Meta-Ensemble"] if m in set(df["name"])][:4]
    if not models:
        print("  [skip] 09: нет моделей для матриц ошибок")
        return
    fig, axes = plt.subplots(1, len(models), figsize=(W2, 2.35))
    axes = np.atleast_1d(axes)
    for ax, name in zip(axes, models):
        row = r[name]
        C = np.array([[row["tn"], row["fp"]], [row["fn"], row["tp"]]], float)
        Cn = C / np.maximum(C.sum(1, keepdims=True), 1)
        ax.imshow(Cn, cmap=plt.cm.Greys, vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{int(C[i, j])}\n({100*Cn[i, j]:.1f} %)",
                        ha="center", va="center", fontsize=6.2,
                        color="white" if Cn[i, j] > 0.55 else "black")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Норма", "Атака"], fontsize=7)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Норма", "Атака"], fontsize=7,
                           rotation=90, va="center")
        ax.tick_params(axis="both", length=0, pad=2)
        _panel(ax, MODEL_LABEL.get(name, name), fs=8.5)
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_linewidth(0.7)
    axes[0].set_ylabel("Истинный класс", labelpad=4)
    fig.supxlabel("Предсказанный класс", fontsize=9)
    _suptitle(fig, "Матрицы ошибок лучших детекторов")
    # ручные отступы: вертикальные подписи требуют места слева
    fig.subplots_adjust(left=0.085, right=0.995, wspace=0.30,
                        bottom=0.20, top=0.86 if TITLES else 0.90)
    save(fig, "09_confusion_matrices")


def fig10_compromise(a, save):
    df = a["results_df"]; r = rows_by_name(df)
    models = [m for m in MAIN_ORDER + ["Cascade-Stage1", "Cascade-Stage2"]
              if m in set(df["name"])]
    v = [r[m].get("compromise_recall", np.nan) for m in models]
    order = np.argsort([-(x if np.isfinite(x) else -1) for x in v])
    models = [models[i] for i in order]; v = [v[i] for i in order]
    fig, ax = plt.subplots(figsize=(W2 * 0.78, 2.7))
    b = ax.bar(range(len(v)), v, 0.64,
               color=[st(m, i)["c"] for i, m in enumerate(models)],
               hatch=[st(m, i)["h"] for i, m in enumerate(models)],
               edgecolor="black", linewidth=0.6)
    _bar_labels(ax, b, "{:.1f}", dy=1.2, fs=6.2)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([MODEL_LABEL.get(m, m) for m in models],
                       rotation=32, ha="right")
    ax.set_ylabel("Полнота, %"); ax.set_ylim(0, 110)
    _title(ax, "Обнаружение компрометации узла Diameter")
    _layout(fig)
    save(fig, "10_compromise_detection")


def fig11_training(a, save):
    hist = [("Автокодировщик", a.get("ae_train_hist", [])),
            ("VAE", a.get("vae_train_hist", []))]
    hist = [(n, np.asarray(h, float)) for n, h in hist if len(h) > 1]
    if not hist:
        print("  [skip] 11: нет истории обучения")
        return
    fig, axes = plt.subplots(1, len(hist),
                             figsize=(W1 * len(hist) * 1.15, 2.3))
    axes = np.atleast_1d(axes)
    for ax, (n, h) in zip(axes, hist):
        ax.plot(np.arange(1, len(h) + 1), h, color="#1a1a1a", lw=1.2,
                label="Обучающая выборка")
        ax.set_yscale("log"); ax.set_xlabel("Эпоха")
        _panel(ax, n, fs=8.8); ax.legend(loc="upper right")
    axes[0].set_ylabel("Функция потерь (лог. шкала)")
    _suptitle(fig, "Кривые обучения нейросетевых моделей")
    _layout(fig)
    save(fig, "11_training_curves")


def fig12_ensemble(a, save):
    df, atks = a["results_df"], a["attacks"]
    ens = [n for n in df["name"] if str(n).startswith("Ens:")]
    if not ens:
        print("  [skip] 12: специализированный ансамбль не запускался")
        return
    M = _recall_matrix(df, ens, atks)
    fig, ax = plt.subplots(figsize=(W2 * 0.78, 0.34 * len(ens) + 1.4))
    im = _heatmap(ax, M,
                  [n.replace("Ens:", "").replace("_", " ") for n in ens],
                  [alabel(x) for x in atks])
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("Полнота, %", fontsize=8); cb.outline.set_linewidth(0.7)
    _title(ax, "Специализация детекторов ансамбля по группам признаков")
    _layout(fig)
    save(fig, "12_ensemble_specialization")


def fig13_dataset(data_csv, save, atks=None):
    if not data_csv or not os.path.exists(data_csv):
        print("  [skip] 13: датасет не указан (--data)")
        return
    df = pd.read_csv(data_csv)
    acol = first_col(df, ["attack_type", "attack", "label_type"])
    if acol is None:
        print("  [skip] 13: нет колонки attack_type")
        return
    atks = atks or [a for a in ATTACK_ORDER if a in set(df[acol])] or \
        sorted(set(df[acol]) - {"none", "normal"})
    fig, axes = plt.subplots(2, 2, figsize=(W2, 4.9))

    # (а) распределение классов
    ax = axes[0, 0]
    cnt = df[acol].value_counts()
    norm_key = "none" if "none" in cnt.index else "normal"
    keys = [norm_key] + [k for k in atks if k in cnt.index]
    v = [cnt.get(k, 0) for k in keys]
    b = ax.bar(range(len(v)), v, 0.62,
               color=["#d9d9d9"] + [agray(k, atks) for k in keys[1:]],
               hatch=[""] + ATTACK_HATCH[:len(keys) - 1],
               edgecolor="black", linewidth=0.6)
    for bb, x in zip(b, v):
        ax.text(bb.get_x() + bb.get_width() / 2, x, f"{x}", ha="center",
                va="bottom", fontsize=6.2)
    ax.set_xticks(range(len(v)))
    ax.set_xticklabels(["Норма"] + [ashort(k) for k in keys[1:]],
                       fontsize=6.8, rotation=20, ha="right")
    ax.set_ylabel("Число записей"); ax.set_yscale("log")
    _panel(ax, "Распределение классов", fs=8.8)
    _panel_tag(ax, "(а)")

    # (б) временная карта атак
    ax = axes[0, 1]
    tcol = first_col(df, ["interval", "time_interval", "window", "t"])
    if tcol:
        for i, atk in enumerate([k for k in atks if k in set(df[acol])]):
            s = df[df[acol] == atk].groupby(tcol).size()
            ax.plot(s.index, s.values, color=agray(atk, atks),
                    ls=_FB_LS[i % 5], lw=1.0, label=ashort(atk))
        ax.set_xlabel("Интервал наблюдения")
        ax.set_ylabel("Число аномальных записей")
        ax.legend(fontsize=6.0, ncol=2)
    _panel(ax, "Временная карта атак", fs=8.8)
    _panel_tag(ax, "(б)")

    # (в) контраст признаков: норма vs компрометация узла
    ax = axes[1, 0]
    cand = ["obs_inbound_outbound_ratio", "integrity_fail_rate",
            "var_obs_total_messages", "cu_consistency_error",
            "obs_destinations_cv", "obs_answer_result_error_rate",
            "obs_s6a_dominance_ratio", "obs_total_messages"]
    feats = [f for f in cand if f in df.columns][:5]
    if not feats:                      # запасной вариант: по дисперсии
        num = df.select_dtypes(include=[np.number])
        feats = list(num.var().sort_values(ascending=False).index[:5])
    comp_key = COMPROMISE if COMPROMISE in set(df[acol]) else (atks[-1]
                                                               if atks else None)
    if feats and comp_key:
        norm = df[df[acol] == norm_key]; comp = df[df[acol] == comp_key]
        mn = [norm[f].median() for f in feats]
        mc = [comp[f].median() for f in feats]
        sc = [max(abs(x), abs(z), 1e-9) for x, z in zip(mn, mc)]
        x = np.arange(len(feats)); w = 0.36
        ax.bar(x - w / 2, [p / s for p, s in zip(mn, sc)], w, color="#d9d9d9",
               edgecolor="black", linewidth=0.6, label="Норма")
        ax.bar(x + w / 2, [p / s for p, s in zip(mc, sc)], w, color="#2b2b2b",
               hatch="////", edgecolor="black", linewidth=0.6,
               label="Компрометация")
        ax.set_xticks(x)
        ax.set_xticklabels([f.replace("_", "\n") for f in feats], fontsize=5.6)
        ax.set_ylabel("Медиана (норм.)"); ax.legend(fontsize=6.2)
    _panel(ax, "Контраст признаков", fs=8.8)
    _panel_tag(ax, "(в)")

    # (г) надёжность канала ведущий-ведомый
    ax = axes[1, 1]
    ccol = first_col(df, ["cu_delivered", "cu_delivery", "integrity_check"])
    if ccol:
        dr = 100 * df.groupby(acol)[ccol].mean()
        keys = [k for k in [norm_key] + atks if k in dr.index]
        b = ax.bar(range(len(keys)), [dr[k] for k in keys], 0.62,
                   color="#8c8c8c", hatch="..", edgecolor="black",
                   linewidth=0.6)
        _bar_labels(ax, b, "{:.1f}", dy=0.8, fs=6.2)
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels(["Норма"] + [ashort(k) for k in keys[1:]],
                           fontsize=6.8, rotation=20, ha="right")
        ax.set_ylabel(f"{ccol.replace('_', ' ')}, %"); ax.set_ylim(0, 110)
    _panel(ax, "Надёжность канала ведущий–ведомый", fs=8.8)
    _panel_tag(ax, "(г)")

    _suptitle(fig, "Характеристики синтетического датасета Diameter")
    _layout(fig, top=0.96)
    save(fig, "13_dataset_overview")


def fig14_cascade(a, save):
    s1, s2, sc = a.get("cascade_s1"), a.get("cascade_s2"), a.get("cascade_comb")
    if s1 is None or s2 is None or sc is None:
        print("  [skip] 14: нет оценок этапов каскада")
        return
    y, at = a["y_test"], np.asarray(a["attack_types"])
    panels = [("Уровень 1:\nобщий детектор", s1),
              ("Уровень 2:\nдетектор компрометации", s2),
              ("Каскад:\nкомбинированная оценка", sc)]
    fig, axes = plt.subplots(1, 3, figsize=(W2, 2.6))
    bins = np.linspace(0, 1, 41)
    for ax, (ttl, s) in zip(axes, panels):
        s = np.nan_to_num(np.asarray(s, float))
        ax.hist(s[y == 0], bins=bins, density=True, color="#d9d9d9",
                edgecolor="#4a4a4a", linewidth=0.4, label="Норма")
        m_oth = (y == 1) & (at != COMPROMISE)
        if m_oth.sum() > 5:
            ax.hist(s[m_oth], bins=bins, density=True, histtype="step",
                    lw=1.1, ls="--", color="#6a6a6a", label="Прочие атаки")
        m_c = (at == COMPROMISE)
        if m_c.sum() > 5:
            ax.hist(s[m_c], bins=bins, density=True, histtype="step", lw=1.4,
                    color="#000000", label="Компрометация")
        ax.axvline(0.5, color="#000000", lw=0.7, ls=(0, (1, 2)))
        ax.set_yscale("log"); ax.set_xlabel("Оценка")
        _panel(ax, ttl, fs=8.0)
    axes[0].set_ylabel("Плотность (лог. шкала)")
    axes[0].legend(loc="upper center", fontsize=6.0)
    for ax, t in zip(axes, ["(а)", "(б)", "(в)"]):
        _panel_tag(ax, t)
    _suptitle(fig, "Анализ уровней каскадного детектора")
    _layout(fig)
    save(fig, "14_cascade_analysis")


# ══════════════════════════════════════════════════════════════════════
#  5. МУЛЬТИСИД-ФИГУРЫ 15-18
# ══════════════════════════════════════════════════════════════════════

def fig15_multiseed_bars(alld, save):
    models = [m for m in MAIN_ORDER if m in set(alld["name"])]
    mets = [m for m in ["auc_pr", "f1", "compromise_recall",
                        "recall_at_1pct_fpr"] if m in alld.columns]
    fig, axes = plt.subplots(2, 2, figsize=(W2, 5.0))
    for ax, met in zip(axes.ravel(), mets):
        g = alld.groupby("name")[met]
        mu = [g.mean().get(m, np.nan) for m in models]
        sd = [g.std().get(m, np.nan) for m in models]
        ax.bar(range(len(models)), mu, 0.62, yerr=np.nan_to_num(sd),
               capsize=2.2,
               color=[st(m, i)["c"] for i, m in enumerate(models)],
               hatch=[st(m, i)["h"] for i, m in enumerate(models)],
               edgecolor="black", linewidth=0.6,
               error_kw=dict(elinewidth=0.7, ecolor="#000000"))
        for i, m in enumerate(models):     # точки отдельных сидов
            pts = alld.loc[alld["name"] == m, met].dropna().to_numpy()
            if len(pts):
                ax.plot(i + np.linspace(-0.13, 0.13, len(pts)), pts,
                        ls="none", marker="o", ms=1.8, mfc="white",
                        mec="#000000", mew=0.4, zorder=5)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([MODEL_LABEL.get(m, m) for m in models],
                           rotation=32, ha="right", fontsize=6.4)
        ax.set_ylabel(METRIC_LABEL.get(met, met), fontsize=8)
        top = np.nanmax([x for x in mu if np.isfinite(x)] or [1])
        ax.set_ylim(0, top * 1.22)
    for ax, t in zip(axes.ravel(), ["(а)", "(б)", "(в)", "(г)"]):
        _panel_tag(ax, t, dx=-0.10)
    _suptitle(fig, f"Устойчивость результатов Diameter по "
                   f"{alld['seed'].nunique()} прогонам (среднее ± СКО)")
    _layout(fig)
    save(fig, "15_multiseed_mean_std")


def fig16_seed_spread(alld, save):
    met = "compromise_recall"
    if met not in alld.columns:
        print("  [skip] 16: нет compromise_recall")
        return
    models = [m for m in MAIN_ORDER + ["Cascade-Stage1", "RF (no MS)",
                                       "RF (no temporal)"]
              if m in set(alld["name"])]
    data = [alld.loc[alld["name"] == m, met].dropna().to_numpy()
            for m in models]
    keep = [i for i, d in enumerate(data) if len(d) >= 2]
    models = [models[i] for i in keep]; data = [data[i] for i in keep]
    if not data:
        print("  [skip] 16: мало данных")
        return
    fig, ax = plt.subplots(figsize=(W2 * 0.82, 3.0))
    bp = ax.boxplot(data, widths=0.55, patch_artist=True, showmeans=True,
                    meanprops=dict(marker="D", ms=2.6, mfc="white",
                                   mec="black", mew=0.6),
                    medianprops=dict(color="black", lw=1.0),
                    whiskerprops=dict(lw=0.7), capprops=dict(lw=0.7),
                    flierprops=dict(marker="o", ms=2.0, mfc="white",
                                    mec="black", mew=0.4))
    for i, (patch, m) in enumerate(zip(bp["boxes"], models)):
        patch.set_facecolor(st(m, i)["c"]); patch.set_edgecolor("black")
        patch.set_linewidth(0.6); patch.set_alpha(0.55)
    ax.set_xticklabels([MODEL_LABEL.get(m, m) for m in models],
                       rotation=32, ha="right", fontsize=6.6)
    ax.set_ylabel(METRIC_LABEL[met])
    _title(ax, "Разброс полноты обнаружения компрометации по сидам")
    _layout(fig)
    save(fig, "16_seed_spread")


def fig17_wilcoxon(sig, save):
    if sig is None or sig.empty:
        print("  [skip] 17: нет significance.csv")
        return
    d = sig.copy()
    d["label"] = [f"{MODEL_LABEL.get(x, x)} vs {MODEL_LABEL.get(y, y)}\n"
                  f"[{METRIC_LABEL.get(m, m).split(',')[0]}]"
                  for x, y, m in zip(d["a"], d["b"], d["metric"])]
    scale = np.where(d["metric"].isin(["compromise_recall",
                                       "recall_at_1pct_fpr"]), 100.0, 1.0)
    d["delta_rel"] = 100 * d["delta_median"] / np.where(
        np.abs(d["median_b"]) > 1e-9, np.abs(d["median_b"]), scale)
    d = d.sort_values("delta_rel")
    pcol = "p_two_holm" if "p_two_holm" in d.columns else "p_two_sided"
    sigm = d[pcol] < 0.05
    fig, ax = plt.subplots(figsize=(W2 * 0.85, 0.30 * len(d) + 1.5))
    ax.barh(np.arange(len(d)), d["delta_rel"], 0.6,
            color=["#2b2b2b" if s else "#c4c4c4" for s in sigm],
            hatch=["////" if s else "" for s in sigm],
            edgecolor="black", linewidth=0.6)
    ax.axvline(0, color="black", lw=0.8)
    for i, (_, r) in enumerate(d.iterrows()):
        txt = (f"p={r[pcol]:.1e}" if np.isfinite(r[pcol]) else "p=—") + \
              f"  ({int(r['a_better'])}/{int(r['n_seeds'])})"
        pos = r["delta_rel"] >= 0
        ax.text(r["delta_rel"] + (0.6 if pos else -0.6), i, txt, fontsize=5.9,
                va="center", ha="left" if pos else "right")
    ax.set_yticks(np.arange(len(d)))
    ax.set_yticklabels(d["label"], fontsize=6.0)
    ax.set_xlabel("Относительный прирост медианы, % (A относительно B)")
    lim = np.nanmax(np.abs(d["delta_rel"])) * 1.45 + 1
    ax.set_xlim(-lim, lim)
    _title(ax, "Парный критерий Уилкоксона, поправка Холма (Diameter)")
    ax.legend(handles=[Patch(facecolor="#2b2b2b", hatch="////",
                             edgecolor="black",
                             label="значимо, $p_{Holm}<0.05$"),
                       Patch(facecolor="#c4c4c4", edgecolor="black",
                             label="незначимо")], loc="lower right")
    _layout(fig)
    save(fig, "17_wilcoxon_significance")


def fig18_attack_heatmap_multiseed(alld, save):
    models = [m for m in MAIN_ORDER if m in set(alld["name"])]
    atks = detect_attacks(alld)
    cols = [f"rec_{a}" for a in atks if f"rec_{a}" in alld.columns]
    if not cols:
        print("  [skip] 18: нет per-attack метрик")
        return
    mu = alld.groupby("name")[cols].mean(); sd = alld.groupby("name")[cols].std()
    M = np.array([[mu.loc[m, c] if m in mu.index else np.nan for c in cols]
                  for m in models], float)
    S = np.array([[sd.loc[m, c] if m in sd.index else np.nan for c in cols]
                  for m in models], float)
    fig, ax = plt.subplots(figsize=(W2 * 0.82, 0.34 * len(models) + 1.5))
    im = ax.imshow(M, cmap=plt.cm.Greys, vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([alabel(c[4:]) for c in cols], fontsize=7.0)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([MODEL_LABEL.get(m, m) for m in models], fontsize=7.0)
    ax.tick_params(axis="both", length=0, pad=2)
    ax.grid(False)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if not np.isfinite(M[i, j]):
                continue
            s = 0.0 if not np.isfinite(S[i, j]) else S[i, j]
            ax.text(j, i, f"{M[i, j]:.1f}\n$\\pm${s:.1f}", ha="center",
                    va="center", fontsize=5.8,
                    color="white" if M[i, j] > 58 else "black")
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("Полнота, %", fontsize=8); cb.outline.set_linewidth(0.7)
    _title(ax, f"Полнота по типам атак Diameter, "
               f"{alld['seed'].nunique()} сидов (среднее ± СКО)")
    _layout(fig)
    save(fig, "18_attack_heatmap_multiseed")


def export_latex(alld, sig, outdir):
    mets = [m for m in ["auc_roc", "auc_pr", "f1", "fpr",
                        "compromise_recall", "recall_at_1pct_fpr"]
            if m in alld.columns]
    models = [m for m in MAIN_ORDER if m in set(alld["name"])]
    L = [r"\begin{tabular}{l" + "c" * len(mets) + "}", r"\hline",
         "Модель & " + " & ".join(METRIC_LABEL.get(m, m).split(",")[0]
                                  for m in mets) + r" \\", r"\hline"]
    for m in models:
        sub = alld[alld["name"] == m]; cells = []
        for met in mets:
            mu, sd = sub[met].mean(), sub[met].std()
            dg = 4 if met in ("auc_roc", "auc_pr", "f1", "fpr") else 1
            cells.append("—" if not np.isfinite(mu) else
                         f"${mu:.{dg}f} \\pm "
                         f"{0 if not np.isfinite(sd) else sd:.{dg}f}$")
        L.append(f"{MODEL_LABEL.get(m, m)} & " + " & ".join(cells) + r" \\")
    L += [r"\hline", r"\end{tabular}"]
    with open(os.path.join(outdir, "table_mean_std.tex"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(L))
    print("  ✓ table_mean_std.tex")

    if sig is not None and not sig.empty:
        pcol = "p_two_holm" if "p_two_holm" in sig.columns else "p_two_sided"
        T = [r"\begin{tabular}{llccc}", r"\hline",
             r"Сравнение & Метрика & Медианы & $\Delta$ & "
             r"$p_{\mathrm{Holm}}$ \\", r"\hline"]
        for _, r in sig.iterrows():
            star = r"$^{*}$" if r[pcol] < 0.05 else ""
            T.append(f"{MODEL_LABEL.get(r['a'], r['a'])} vs "
                     f"{MODEL_LABEL.get(r['b'], r['b'])} & "
                     f"{METRIC_LABEL.get(r['metric'], r['metric']).split(',')[0]}"
                     f" & {r['median_a']:.4f} / {r['median_b']:.4f} & "
                     f"{r['delta_median']:+.4f} & {r[pcol]:.3g}{star} \\\\")
        T += [r"\hline", r"\end{tabular}"]
        with open(os.path.join(outdir, "table_wilcoxon.tex"), "w",
                  encoding="utf-8") as f:
            f.write("\n".join(T))
        print("  ✓ table_wilcoxon.tex")


# ══════════════════════════════════════════════════════════════════════
#  6. CLI
# ══════════════════════════════════════════════════════════════════════

def main():
    global TITLES
    p = argparse.ArgumentParser(
        description="Печатная графика Diameter в едином стиле")
    p.add_argument("--run", default=None,
                   help="каталог одного сида (diameter_seed_runs/seed_42)")
    p.add_argument("--seeds-dir", default=None,
                   help="каталог мультисид-прогона (diameter_seed_runs)")
    p.add_argument("--data", default=None, help="csv датасета для фигуры 13")
    p.add_argument("--out", default="diameter_figs_print",
                   help="каталог для фигур")
    p.add_argument("--formats", nargs="+", default=FORMATS,
                   choices=["png", "pdf", "svg", "eps"])
    p.add_argument("--titles", action="store_true",
                   help="вернуть заголовки фигур (по умолчанию отключены)")
    args = p.parse_args()
    if not args.run and not args.seeds_dir:
        p.error("укажите --run и/или --seeds-dir")

    TITLES = args.titles
    apply_style()
    save = Out(args.out, args.formats)
    print(f"\nПечатная графика Diameter → {args.out}/ "
          f"({DPI} dpi, {', '.join(args.formats)}, "
          f"заголовки: {'вкл' if TITLES else 'выкл'})\n")

    run_dir = args.run
    if run_dir is None:
        c = sorted(glob.glob(os.path.join(args.seeds_dir, "seed_*")))
        run_dir = c[0] if c else None

    atks = None
    if run_dir:
        print("— фигуры по одному прогону —")
        a = load_run(run_dir)
        atks = a["attacks"]
        fig01_roc(a, save)
        fig02_roc_zoom(a, save)
        fig03_pr(a, save)
        fig04_attack_heatmap(a, save)
        fig05_summary_bars(a, save)
        fig06_feature_importance(a, save)
        fig07_ablation(a, save)
        fig08_score_distributions(a, save)
        fig09_confusion(a, save)
        fig10_compromise(a, save)
        fig11_training(a, save)
        fig12_ensemble(a, save)
        fig13_dataset(args.data, save, atks)
        fig14_cascade(a, save)
    elif args.data:
        fig13_dataset(args.data, save)

    if args.seeds_dir:
        print("\n— мультисид-фигуры —")
        alld, sig = load_seeds(args.seeds_dir)
        fig15_multiseed_bars(alld, save)
        fig16_seed_spread(alld, save)
        fig17_wilcoxon(sig, save)
        fig18_attack_heatmap_multiseed(alld, save)
        print("\n— таблицы LaTeX —")
        export_latex(alld, sig, args.out)

    print(f"\nГотово: {len(save.saved)} фигур в {args.out}/")


if __name__ == "__main__":
    main()
