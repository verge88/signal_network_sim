from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
from matplotlib.figure import Figure

from .model import ExperimentResult


def safe_name(text: str) -> str:
    text = re.sub(r"[^\w.\-]+", "_", str(text), flags=re.UNICODE)
    return text.strip("_") or "result"


def save_figure(fig: Figure, path: str | Path, dpi: int = 300) -> Path:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() not in (".png", ".svg", ".pdf"):
        raise ValueError("Поддерживаются PNG, SVG и PDF.")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


def save_table(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df.to_csv(path, index=False, encoding="utf-8-sig")
    elif suffix == ".xlsx":
        try:
            df.to_excel(path, index=False)
        except ImportError as exc:
            raise RuntimeError("Для XLSX установите openpyxl: pip install openpyxl") from exc
    elif suffix in (".tex", ".latex"):
        path.write_text(df.to_latex(index=False), encoding="utf-8")
    elif suffix in (".md", ".markdown"):
        try:
            text = df.to_markdown(index=False)
        except ImportError as exc:
            raise RuntimeError("Для Markdown установите tabulate: pip install tabulate") from exc
        path.write_text(text, encoding="utf-8")
    else:
        raise ValueError("Поддерживаются CSV, XLSX, TEX и MD.")
    return path


def export_workbook(result: ExperimentResult, path: str | Path) -> Path:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    try:
        writer = pd.ExcelWriter(path, engine="openpyxl")
    except ImportError as exc:
        raise RuntimeError("Для XLSX установите openpyxl: pip install openpyxl") from exc
    used = set()
    with writer:
        for name, df in result.tables.items():
            sheet = safe_name(name)[-31:] or "table"
            base = sheet; i = 2
            while sheet in used:
                suffix = f"_{i}"; sheet = base[:31-len(suffix)] + suffix; i += 1
            used.add(sheet)
            df.to_excel(writer, sheet_name=sheet, index=False)
    return path


def export_result_package(result: ExperimentResult, destination: str | Path) -> Path:
    destination = Path(destination); destination.mkdir(parents=True, exist_ok=True)
    manifest_dir = destination / "manifests"; table_dir = destination / "tables"
    manifest_dir.mkdir(exist_ok=True); table_dir.mkdir(exist_ok=True)
    for name, doc in result.manifests.items():
        (manifest_dir / f"{safe_name(name)}.json").write_text(
            json.dumps(doc, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    for name, doc in result.json_documents.items():
        if name in result.manifests:
            continue
        if name.endswith("null_test.json"):
            (manifest_dir / f"{safe_name(name)}.json").write_text(
                json.dumps(doc, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    for name, df in result.tables.items():
        df.to_csv(table_dir / f"{safe_name(name)}.csv", index=False, encoding="utf-8-sig")
    summary = {
        "experiment_id": result.experiment_id,
        "protocol": result.protocol,
        "source_root": str(result.root),
        "summary": result.summary(),
        "warnings": result.warnings,
        "tables": sorted(result.tables),
    }
    (destination / "result_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return destination
