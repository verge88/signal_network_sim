from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from .model import ExperimentResult

KNOWN_CSV_KIND = {
    "all_seed_metrics.csv": "raw_metrics",
    "multiseed_raw.csv": "raw_metrics",
    "raw_metrics.csv": "raw_metrics",
    "metrics.csv": "raw_metrics",
    "models.csv": "models",
    "aggregated.csv": "aggregate_metrics",
    "multiseed_agg.csv": "aggregate_metrics",
    "aggregate_metrics.csv": "aggregate_metrics",
    "table_mean_std.csv": "aggregate_display",
    "significance.csv": "statistical_tests",
    "multiseed_tests.csv": "statistical_tests",
    "statistical_tests.csv": "statistical_tests",
    "ablation_feature_sets.csv": "ablation_feature_sets",
    "ablation_temporal.csv": "ablation_temporal",
    "lomfo.csv": "lomfo",
    "detectability_curve.csv": "detectability_curve",
    "sophistication_sweep.csv": "sophistication_sweep",
}

SKIP_DIR_NAMES = {"__pycache__", ".git", ".venv", "venv", "data", "dataset", "datasets"}
SKIP_FILE_PARTS = ("_dataset_", "dataset.csv")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _infer_protocol_from_path(path: Path) -> str:
    lowered = [part.lower() for part in path.parts]
    for proto in ("ss7", "diameter", "sip", "sba", "5g"):
        if proto in lowered:
            return "sba" if proto == "5g" else proto
    return ""


def _manifest_protocol(doc: Any) -> str:
    if not isinstance(doc, dict):
        return ""
    direct = doc.get("protocol")
    if isinstance(direct, str) and direct:
        return direct.lower()
    for key in ("config", "experiment"):
        nested = doc.get(key)
        if isinstance(nested, dict):
            value = nested.get("protocol")
            if isinstance(value, str) and value:
                return value.lower()
    return ""


def _nearest_protocol(csv_path: Path, root: Path, manifest_by_dir: dict[Path, str]) -> str:
    cur = csv_path.parent
    while True:
        if cur in manifest_by_dir and manifest_by_dir[cur]:
            return manifest_by_dir[cur]
        if cur == root or cur.parent == cur:
            break
        cur = cur.parent
    return _infer_protocol_from_path(csv_path)


def _table_key(path: Path, root: Path) -> str:
    rel = path.relative_to(root)
    parent = str(rel.parent).replace("\\", "/")
    return f"{parent}/{path.stem}" if parent not in ("", ".") else path.stem


def _is_candidate_csv(path: Path) -> bool:
    if any(part in SKIP_DIR_NAMES for part in path.parts):
        return False
    name = path.name.lower()
    if any(fragment in name for fragment in SKIP_FILE_PARTS):
        return False
    size = path.stat().st_size
    if size > 80 * 1024 * 1024:
        return False
    if name in KNOWN_CSV_KIND:
        return True
    return size <= 20 * 1024 * 1024


def _load_metrics_json(doc: Any) -> pd.DataFrame | None:
    if not isinstance(doc, dict):
        return None
    rows = doc.get("rows")
    if isinstance(rows, list) and rows:
        try:
            return pd.DataFrame(rows)
        except Exception:
            return None
    return None


def _add_combined_tables(result: ExperimentResult, kinds: dict[str, list[str]]) -> None:
    for kind, names in kinds.items():
        if len(names) < 2:
            continue
        frames = []
        for name in names:
            df = result.tables[name].copy()
            source = result.table_sources.get(name)
            if "_source_table" not in df.columns:
                df.insert(0, "_source_table", name)
            if "_source_protocol" not in df.columns:
                proto = _infer_protocol_from_path(source or result.root)
                df.insert(0, "_source_protocol", proto)
            frames.append(df)
        try:
            combined = pd.concat(frames, ignore_index=True, sort=False)
        except Exception as exc:
            result.warnings.append(f"Не удалось объединить таблицы {kind}: {exc}")
            continue
        name = f"combined/{kind}"
        result.tables[name] = combined
        result.table_sources[name] = result.root


def load_experiment(path: str | Path) -> ExperimentResult:
    root = Path(path).expanduser().resolve()
    if root.is_file():
        root = root.parent
    if not root.exists():
        raise FileNotFoundError(root)

    result = ExperimentResult(root=root, experiment_id=root.name)
    manifest_by_dir: dict[Path, str] = {}

    for json_path in sorted(root.rglob("*.json")):
        if any(part in SKIP_DIR_NAMES for part in json_path.parts):
            continue
        try:
            doc = _read_json(json_path)
        except Exception as exc:
            result.warnings.append(f"{json_path}: JSON не прочитан: {exc}")
            continue

        rel = str(json_path.relative_to(root)).replace("\\", "/")
        result.json_documents[rel] = doc

        if json_path.name in ("manifest.json", "index.json"):
            result.manifests[rel] = doc
            proto = _manifest_protocol(doc)
            if proto:
                manifest_by_dir[json_path.parent] = proto
                if not result.protocol:
                    result.protocol = proto

        if json_path.name == "metrics.json":
            df = _load_metrics_json(doc)
            if df is not None:
                parent = str(json_path.parent.relative_to(root)).replace("\\", "/")
                key = f"{parent}/metrics_json".lstrip("./")
                result.tables[key] = df
                result.table_sources[key] = json_path

    kinds: dict[str, list[str]] = defaultdict(list)

    for csv_path in sorted(root.rglob("*.csv")):
        try:
            if not _is_candidate_csv(csv_path):
                continue
        except OSError:
            continue
        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:
            result.warnings.append(f"{csv_path}: CSV не прочитан: {exc}")
            continue

        key = _table_key(csv_path, root)
        proto = _nearest_protocol(csv_path, root, manifest_by_dir)
        if proto and "_protocol" not in df.columns:
            df.insert(0, "_protocol", proto)
        result.tables[key] = df
        result.table_sources[key] = csv_path

        kind = KNOWN_CSV_KIND.get(csv_path.name.lower())
        if kind:
            kinds[kind].append(key)
        if not result.protocol and proto:
            result.protocol = proto

    _add_combined_tables(result, kinds)
    if not result.protocol:
        result.protocol = _infer_protocol_from_path(root)
    if not result.tables:
        result.warnings.append("В каталоге не найдены CSV/metrics.json с результатами.")
    return result
