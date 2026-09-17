from __future__ import annotations

import pandas as pd

LABEL_HINTS = ("label", "is_anomal", "is_compromised", "attack", "y_true", "target")
GROUP_HINTS = ("operator", "master", "node_id", "day", "date", "epoch", "window_day")


def dataset_report(csv_path: str, min_positive: int = 50, emit=print) -> dict:
    """Report positive windows and positive groups before starting training."""
    dataframe = pd.read_csv(csv_path)
    columns = list(dataframe.columns)
    labels = [column for column in columns
              if any(hint in column.lower() for hint in LABEL_HINTS)]
    groups = [column for column in columns
              if any(hint in column.lower() for hint in GROUP_HINTS)]
    emit(f"окон: {len(dataframe)}, столбцов: {len(columns)}")
    emit(f"кандидаты в метку: {labels or '—'}; кандидаты в группировку: {groups or '—'}")

    report = {"rows": len(dataframe), "labels": labels, "units": {}}
    for label in labels:
        numeric = pd.to_numeric(dataframe[label], errors="coerce").fillna(0)
        positive = int(numeric.astype(bool).sum())
        emit(f"  {label}: положительных окон {positive} ({positive / max(len(dataframe), 1):.5f})")
        report["units"][label] = {"windows": positive}
        for group in groups:
            count = int(dataframe.loc[numeric > 0, group].nunique())
            mark = "OK" if count > min_positive else "МАЛО"
            emit(f"    единиц «{group}» с положительной меткой: {count} [{mark}]")
            report["units"][label][group] = count
    return report
