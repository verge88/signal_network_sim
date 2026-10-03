from __future__ import annotations

import math

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from .model import ChartSpec
from .tables import filter_dataframe

CHART_TYPES = {
    "Столбчатая": "bar",
    "Линейная": "line",
    "Точки": "scatter",
    "Boxplot": "box",
    "Гистограмма": "hist",
}
AGGREGATIONS = {
    "Среднее": "mean",
    "Медиана": "median",
    "Сумма": "sum",
    "Без агрегации": "none",
}
ERROR_TYPES = {"Нет": "none", "Std": "std", "95% CI": "ci95"}


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _error_value(values: pd.Series, mode: str) -> float:
    s = _numeric(values).dropna()
    if len(s) <= 1:
        return 0.0
    if mode == "std":
        return float(s.std(ddof=1))
    if mode == "ci95":
        return float(1.96 * s.std(ddof=1) / math.sqrt(len(s)))
    return 0.0


def _aggregate(df: pd.DataFrame, spec: ChartSpec) -> pd.DataFrame:
    if spec.y not in df.columns:
        raise ValueError(f"Нет столбца Y: {spec.y}")
    work = df.copy()
    work[spec.y] = _numeric(work[spec.y])
    work = work.dropna(subset=[spec.y])
    if spec.aggregation == "none":
        return work

    group_cols = [c for c in (spec.x, spec.group) if c]
    if not group_cols:
        raise ValueError("Для агрегации выберите X или Group.")
    agg_name = spec.aggregation if spec.aggregation in ("mean", "median", "sum") else "mean"
    grouped = work.groupby(group_cols, dropna=False)[spec.y]
    values = getattr(grouped, agg_name)().rename("_value").reset_index()
    if spec.error != "none":
        err = grouped.apply(lambda s: _error_value(s, spec.error)).rename("_error").reset_index()
        values = values.merge(err, on=group_cols, how="left")
    else:
        values["_error"] = 0.0
    return values


def _new_figure(title: str) -> tuple[Figure, object]:
    fig = Figure(figsize=(9.0, 5.8), dpi=100, constrained_layout=True)
    ax = fig.add_subplot(111)
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.25)
    return fig, ax


def build_figure(df: pd.DataFrame, spec: ChartSpec) -> Figure:
    work = filter_dataframe(df, spec.filter_column, spec.filter_value)
    if work.empty:
        raise ValueError("После фильтрации нет данных.")
    title = spec.title or f"{spec.y} — {spec.x}"
    fig, ax = _new_figure(title)

    if spec.chart_type in ("bar", "line"):
        if not spec.x or spec.x not in work.columns:
            raise ValueError("Для bar/line выберите столбец X.")
        data = _aggregate(work, spec)
        y_col = spec.y if spec.aggregation == "none" else "_value"
        err_col = None if spec.aggregation == "none" else "_error"

        if spec.group and spec.group in data.columns:
            groups = list(data.groupby(spec.group, dropna=False))
            categories = list(dict.fromkeys(data[spec.x].astype(str).tolist()))
            pos = np.arange(len(categories), dtype=float)
            if spec.chart_type == "bar":
                width = 0.8 / max(1, len(groups))
                for gi, (group_value, part) in enumerate(groups):
                    keyed = {str(row[spec.x]): row for _, row in part.iterrows()}
                    ys = [float(keyed[c][y_col]) if c in keyed else np.nan for c in categories]
                    es = [float(keyed[c][err_col]) if c in keyed and err_col else 0.0 for c in categories]
                    offset = (gi - (len(groups) - 1) / 2.0) * width
                    ax.bar(pos + offset, ys, width=width, yerr=es if spec.error != "none" else None,
                           capsize=3 if spec.error != "none" else 0, alpha=0.85, label=str(group_value))
                ax.set_xticks(pos)
                ax.set_xticklabels(categories, rotation=30, ha="right")
            else:
                for group_value, part in groups:
                    part = part.copy()
                    part["_xstr"] = part[spec.x].astype(str)
                    order = {c: i for i, c in enumerate(categories)}
                    part["_ord"] = part["_xstr"].map(order)
                    part = part.sort_values("_ord")
                    ys = _numeric(part[y_col]).to_numpy()
                    es = _numeric(part[err_col]).fillna(0).to_numpy() if err_col else None
                    ax.errorbar(part["_xstr"], ys, yerr=es if spec.error != "none" else None,
                                marker="o", capsize=3 if spec.error != "none" else 0,
                                label=str(group_value))
            ax.legend()
        else:
            data = data.copy()
            x_vals = data[spec.x].astype(str).tolist()
            y_vals = _numeric(data[y_col]).to_numpy()
            y_err = _numeric(data[err_col]).fillna(0).to_numpy() if err_col else None
            if spec.chart_type == "bar":
                ax.bar(x_vals, y_vals, yerr=y_err if spec.error != "none" else None,
                       capsize=3 if spec.error != "none" else 0, alpha=0.85)
                ax.tick_params(axis="x", labelrotation=30)
            else:
                ax.errorbar(x_vals, y_vals, yerr=y_err if spec.error != "none" else None,
                            marker="o", capsize=3 if spec.error != "none" else 0)

    elif spec.chart_type == "scatter":
        if not spec.x or spec.x not in work.columns or not spec.y or spec.y not in work.columns:
            raise ValueError("Для scatter выберите X и Y.")
        x = _numeric(work[spec.x]); y = _numeric(work[spec.y])
        valid = x.notna() & y.notna(); work = work.loc[valid]; x = x.loc[valid]; y = y.loc[valid]
        if spec.group and spec.group in work.columns:
            for group_value, idx in work.groupby(spec.group).groups.items():
                ax.scatter(x.loc[idx], y.loc[idx], alpha=0.75, label=str(group_value))
            ax.legend()
        else:
            ax.scatter(x, y, alpha=0.75)

    elif spec.chart_type == "box":
        if not spec.x or spec.x not in work.columns or not spec.y or spec.y not in work.columns:
            raise ValueError("Для boxplot выберите категорию X и числовой Y.")
        labels, arrays = [], []
        for key, part in work.groupby(spec.x, dropna=False):
            values = _numeric(part[spec.y]).dropna().to_numpy()
            if len(values):
                labels.append(str(key)); arrays.append(values)
        if not arrays:
            raise ValueError("Нет числовых данных для boxplot.")
        ax.boxplot(arrays, tick_labels=labels, showmeans=True)
        ax.tick_params(axis="x", labelrotation=30)

    elif spec.chart_type == "hist":
        target = spec.y if spec.y in work.columns else spec.x
        if not target:
            raise ValueError("Для гистограммы выберите X или Y.")
        values = _numeric(work[target]).dropna()
        if values.empty:
            raise ValueError("Нет числовых данных для гистограммы.")
        bins = min(30, max(8, int(math.sqrt(len(values)))))
        ax.hist(values, bins=bins, alpha=0.85)
        ax.set_xlabel(spec.x_label or target)
    else:
        raise ValueError(f"Неизвестный тип графика: {spec.chart_type}")

    if spec.chart_type != "hist":
        ax.set_xlabel(spec.x_label or spec.x)
    ax.set_ylabel(spec.y_label or spec.y)
    return fig


def suggested_spec(df: pd.DataFrame, table_name: str = "") -> ChartSpec:
    categorical = [str(c) for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    numeric = [str(c) for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    preferred_x = ""
    for candidate in ("model", "name", "sophistication", "feature_set", "attack_type", "seed", "_protocol", "_source_protocol"):
        if candidate in df.columns:
            preferred_x = candidate; break
    if not preferred_x:
        preferred_x = categorical[0] if categorical else (numeric[0] if numeric else "")
    preferred_y = ""
    for candidate in ("episode_recall", "compromise_recall", "recall", "auc", "auc_roc", "auc_pr", "ap", "f1", "window_fpr", "fpr"):
        if candidate in df.columns:
            preferred_y = candidate; break
    if not preferred_y:
        preferred_y = numeric[0] if numeric else ""
    group = ""
    for candidate in ("family", "_protocol", "_source_protocol"):
        if candidate in df.columns and candidate != preferred_x:
            group = candidate; break
    return ChartSpec(chart_type="bar", table_name=table_name, x=preferred_x, y=preferred_y,
                     group=group, aggregation="mean", error="ci95")
