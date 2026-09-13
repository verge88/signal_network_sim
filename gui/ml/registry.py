"""Experiment registry, manifests and transfer matrices."""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from typing import Optional

import pandas as pd


def new_run_dir(root: str, protocol: str, tag: str = "") -> str:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    safe = "".join(char if char.isalnum() or char in "-_" else "_" for char in tag)[:40]
    path = os.path.join(root, f"{stamp}_{protocol}{'_' + safe if safe else ''}")
    os.makedirs(path, exist_ok=True)
    return path


def environment() -> dict:
    def version(module: str) -> Optional[str]:
        try:
            return __import__(module).__version__
        except Exception:  # noqa: BLE001
            return None
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True, check=False).stdout.strip() or None
        dirty = bool(subprocess.run(["git", "status", "--porcelain"],
                                    capture_output=True, text=True, check=False).stdout.strip())
    except OSError:
        commit, dirty = None, False
    return {"python": sys.version.split()[0], "platform": platform.platform(),
            "numpy": version("numpy"), "pandas": version("pandas"),
            "sklearn": version("sklearn"), "torch": version("torch"),
            "git_commit": commit, "git_dirty": dirty}


def write_manifest(run_dir: str, *, protocol: str, seed: int, tag: str,
                   topology_path: str, topology_snapshot: Optional[dict],
                   overrides: dict, notes: str = "") -> None:
    manifest = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "protocol": protocol,
                "seed": seed, "tag": tag, "topology_path": topology_path,
                "overrides": overrides, "notes": notes, "env": environment()}
    with open(os.path.join(run_dir, "manifest.json"), "w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
    if topology_snapshot is not None:
        with open(os.path.join(run_dir, "topology.json"), "w", encoding="utf-8") as stream:
            json.dump(topology_snapshot, stream, ensure_ascii=False, indent=2)


def _read_json(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as stream:
            return json.load(stream)
    except (OSError, ValueError):
        return {}


def metrics_frame(run_dir: str) -> Optional[pd.DataFrame]:
    candidates = ("results/metrics.csv", "results/results.csv",
                  "results/model_comparison.csv", "transfer_metrics.csv")
    for relative in candidates:
        path = os.path.join(run_dir, relative)
        if os.path.exists(path):
            try:
                return pd.read_csv(path)
            except Exception:  # noqa: BLE001
                pass
    results_dir = os.path.join(run_dir, "results")
    if os.path.isdir(results_dir):
        for name in os.listdir(results_dir):
            if name.endswith(".csv"):
                try:
                    return pd.read_csv(os.path.join(results_dir, name))
                except Exception:  # noqa: BLE001
                    pass
    return None


def best_metrics(run_dir: str) -> dict:
    frame = metrics_frame(run_dir)
    if frame is None or frame.empty:
        return {}
    metric = next((name for name in ("roc_auc", "auc", "ROC-AUC", "auroc") if name in frame), None)
    if metric is None:
        return {}
    index = frame[metric].astype(float).idxmax()
    model_col = next((name for name in ("model", "name", "Model") if name in frame), None)
    result = {"best_model": str(frame.loc[index, model_col]) if model_col else "?",
              "best_roc_auc": round(float(frame.loc[index, metric]), 4)}
    for extra in ("pr_auc", "f1", "recall_at_1pct_fpr"):
        if extra in frame:
            result[extra] = round(float(frame.loc[index, extra]), 4)
    return result


def list_runs(root: str) -> list[dict]:
    if not os.path.isdir(root):
        return []
    rows = []
    for name in sorted(os.listdir(root), reverse=True):
        directory = os.path.join(root, name)
        if not os.path.isdir(directory):
            continue
        manifest = _read_json(os.path.join(directory, "manifest.json"))
        summary = _read_json(os.path.join(directory, "summary.json"))
        row = {"run": name, "dir": directory,
               "protocol": manifest.get("protocol", summary.get("protocol", "?")),
               "seed": manifest.get("seed", summary.get("seed")),
               "tag": manifest.get("tag", ""), "status": summary.get("status", "-"),
               "commit": (manifest.get("env") or {}).get("git_commit", ""),
               "created": manifest.get("created", "")}
        row.update(best_metrics(directory))
        rows.append(row)
    return rows


def aggregate_seeds(run_dirs: list[str]) -> pd.DataFrame:
    frames = []
    for directory in run_dirs:
        frame = metrics_frame(directory)
        if frame is None or frame.empty:
            continue
        frame = frame.copy()
        frame["__run"] = os.path.basename(directory)
        frame["__seed"] = _read_json(os.path.join(directory, "manifest.json")).get("seed")
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    model_col = next((name for name in ("model", "name", "Model") if name in combined), None)
    if model_col is None:
        return combined
    numeric = combined.select_dtypes("number").columns.difference(["__seed"])
    result = combined.groupby(model_col)[list(numeric)].agg(["mean", "std", "count"])
    result.columns = [f"{left}_{right}" for left, right in result.columns]
    return result.reset_index()


def transfer_matrix(runs_root: str, metric: str = "roc_auc") -> pd.DataFrame:
    rows = []
    if not os.path.isdir(runs_root):
        return pd.DataFrame()
    for name in os.listdir(runs_root):
        directory = os.path.join(runs_root, name)
        path = os.path.join(directory, "transfer_metrics.csv")
        if not os.path.exists(path):
            continue
        manifest = _read_json(os.path.join(directory, "manifest.json"))
        try:
            frame = pd.read_csv(path)
        except Exception:  # noqa: BLE001
            continue
        for _, row in frame.iterrows():
            if metric in row:
                rows.append({"train": manifest.get("tag") or name,
                             "test": row.get("dataset", "?"),
                             "model": row.get("model", "?"), metric: row[metric]})
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).pivot_table(index="train", columns="test",
                                          values=metric, aggfunc="max")
