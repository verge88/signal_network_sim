from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd


def numeric_columns(df: pd.DataFrame) -> list[str]:
    return [str(c) for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def categorical_columns(df: pd.DataFrame, max_unique: int = 100) -> list[str]:
    out: list[str] = []
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s):
            continue
        try:
            n = s.nunique(dropna=True)
        except Exception:
            continue
        if n <= max_unique:
            out.append(str(c))
    return out


def ci95_half_width(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    n = len(values)
    if n <= 1:
        return 0.0
    return float(1.96 * values.std(ddof=1) / math.sqrt(n))


def aggregate_table(df: pd.DataFrame, group_by: Iterable[str], metrics: Iterable[str]) -> pd.DataFrame:
    groups = [c for c in group_by if c in df.columns]
    nums = [c for c in metrics if c in df.columns]
    if not nums:
        raise ValueError("Не выбраны числовые метрики.")

    work = df.copy()
    for c in nums:
        work[c] = pd.to_numeric(work[c], errors="coerce")

    if not groups:
        row: dict[str, float | int] = {}
        for metric in nums:
            s = work[metric].dropna()
            row[f"{metric}_mean"] = float(s.mean()) if len(s) else np.nan
            row[f"{metric}_std"] = float(s.std(ddof=1)) if len(s) > 1 else 0.0
            row[f"{metric}_count"] = int(s.count())
            row[f"{metric}_ci95"] = ci95_half_width(s)
        return pd.DataFrame([row])

    rows: list[dict] = []
    for key, part in work.groupby(groups, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(groups, key))
        for metric in nums:
            s = part[metric].dropna()
            row[f"{metric}_mean"] = float(s.mean()) if len(s) else np.nan
            row[f"{metric}_std"] = float(s.std(ddof=1)) if len(s) > 1 else 0.0
            row[f"{metric}_count"] = int(s.count())
            row[f"{metric}_ci95"] = ci95_half_width(s)
        rows.append(row)
    return pd.DataFrame(rows)


def descriptive_table(df: pd.DataFrame) -> pd.DataFrame:
    nums = numeric_columns(df)
    if not nums:
        return pd.DataFrame()
    d = df[nums].describe().T.reset_index().rename(columns={"index": "metric"})
    if "std" in d.columns and "count" in d.columns:
        d["ci95"] = 1.96 * d["std"] / np.sqrt(d["count"].clip(lower=1))
    return d


def filter_dataframe(df: pd.DataFrame, column: str = "", value: str = "") -> pd.DataFrame:
    if not column or column not in df.columns or value == "":
        return df
    s = df[column]
    text = str(value)
    if pd.api.types.is_numeric_dtype(s):
        try:
            num = float(text)
        except ValueError:
            return df[s.astype(str).str.contains(text, case=False, na=False)]
        return df[pd.to_numeric(s, errors="coerce") == num]
    return df[s.astype(str) == text]


def display_value(value) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        if abs(value) >= 10000:
            return f"{value:.3g}"
        return f"{value:.6g}"
    return str(value)
