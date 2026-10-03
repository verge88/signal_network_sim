from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class ChartSpec:
    chart_type: str = "bar"
    table_name: str = ""
    x: str = ""
    y: str = ""
    group: str = ""
    aggregation: str = "mean"
    error: str = "none"
    title: str = ""
    x_label: str = ""
    y_label: str = ""
    filter_column: str = ""
    filter_value: str = ""


@dataclass
class ExperimentResult:
    root: Path
    experiment_id: str
    protocol: str = ""
    manifests: dict[str, dict[str, Any]] = field(default_factory=dict)
    json_documents: dict[str, Any] = field(default_factory=dict)
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    table_sources: dict[str, Path] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def table_names(self) -> list[str]:
        return sorted(self.tables)

    def get_table(self, name: str) -> pd.DataFrame:
        if name not in self.tables:
            raise KeyError(f"Неизвестная таблица: {name}")
        return self.tables[name]

    def primary_manifest(self) -> dict[str, Any]:
        if not self.manifests:
            return {}
        preferred = ["manifest.json", "index.json"]
        for suffix in preferred:
            for key, value in self.manifests.items():
                if key.endswith(suffix):
                    return value
        return next(iter(self.manifests.values()))

    def null_test(self) -> dict[str, Any]:
        for key, value in self.json_documents.items():
            if key.endswith("null_test.json") and isinstance(value, dict):
                return value
        return {}

    def summary(self) -> dict[str, Any]:
        manifest = self.primary_manifest()
        cfg = manifest.get("config", {}) if isinstance(manifest, dict) else {}
        git = manifest.get("git", {}) if isinstance(manifest, dict) else {}
        experiment = manifest.get("experiment", {}) if isinstance(manifest, dict) else {}
        seeds = experiment.get("seeds") or cfg.get("seeds") or manifest.get("seeds") or ""

        null = self.null_test()
        null_status = ""
        if null:
            if null.get("passed") is True:
                null_status = "PASSED"
            elif null.get("passed") is False:
                null_status = "FAILED"
            elif null.get("null_test_passed") is True:
                null_status = "PASSED"

        return {
            "Эксперимент": self.experiment_id,
            "Протокол": self.protocol or manifest.get("protocol", ""),
            "Seeds": seeds,
            "Git commit": git.get("commit", manifest.get("git_commit", "")),
            "Config hash": manifest.get("config_hash", manifest.get("fingerprint", "")),
            "Статус": manifest.get("status", ""),
            "Null-test": null_status,
            "Каталог": str(self.root),
            "Таблиц": len(self.tables),
        }
