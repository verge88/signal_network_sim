"""Independent evaluation of captured detectors on another dataset."""
from __future__ import annotations

import json
import os
import pickle
from typing import Any, Optional

import numpy as np
import pandas as pd

LABEL_COL = "is_anomaly"
ATTACK_COL = "attack_type"


def recall_at_fpr(y: np.ndarray, scores: np.ndarray, fpr_target: float = 0.01) -> float:
    negatives = scores[y == 0]
    if negatives.size == 0:
        return float("nan")
    threshold = np.quantile(negatives, 1.0 - fpr_target)
    positives = scores[y == 1]
    return float((positives >= threshold).mean()) if positives.size else float("nan")


def per_attack_recall(frame: pd.DataFrame, scores: np.ndarray, threshold: float) -> dict[str, float]:
    if ATTACK_COL not in frame or LABEL_COL not in frame:
        return {}
    predicted = scores >= threshold
    result: dict[str, float] = {}
    for attack, group in frame.groupby(ATTACK_COL):
        if str(attack).lower() in ("none", "normal", "nan", ""):
            continue
        indices = group.index.to_numpy()
        mask = np.isin(frame.index.to_numpy(), indices)
        positives = frame.loc[indices, LABEL_COL].to_numpy().astype(bool)
        if positives.sum():
            result[str(attack)] = float(predicted[mask][positives].mean())
    return result


def score_metrics(y: np.ndarray, scores: np.ndarray, frame: pd.DataFrame,
                  threshold: Optional[float] = None) -> dict:
    from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
    threshold = float(np.quantile(scores, 0.95) if threshold is None else threshold)
    predicted = (scores >= threshold).astype(int)
    result = {"threshold": threshold, "n": int(len(y)),
              "anomaly_rate": float(np.mean(y)), "roc_auc": float("nan"),
              "pr_auc": float("nan"), "f1": float("nan"),
              "recall_at_1pct_fpr": recall_at_fpr(y, scores)}
    if len(np.unique(y)) > 1:
        result.update(roc_auc=float(roc_auc_score(y, scores)),
                     pr_auc=float(average_precision_score(y, scores)),
                     f1=float(f1_score(y, predicted, zero_division=0)))
    result["per_attack_recall"] = per_attack_recall(frame, scores, threshold)
    return result


class LoadedRun:
    def __init__(self, run_dir: str):
        self.run_dir = run_dir
        path = os.path.join(run_dir, "artifacts", "sklearn_models.pkl")
        if not os.path.exists(path):
            raise FileNotFoundError(f"нет {path}: модели не сохранены")
        with open(path, "rb") as stream:
            self.objects: dict[str, Any] = pickle.load(stream)
        self.scaler = self._pick("StandardScaler")
        self.imputer = self._pick("SimpleImputer")
        self.features = self._features()

    def _pick(self, prefix: str) -> Any:
        return next((value for key, value in self.objects.items() if key.startswith(prefix)), None)

    def _features(self) -> list[str]:
        path = os.path.join(self.run_dir, "results", "feature_cols.json")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as stream:
                return list(json.load(stream))
        names = getattr(self.scaler, "feature_names_in_", None)
        if names is not None:
            return list(names)
        raise RuntimeError("не удалось определить список признаков прогона")

    def detectors(self) -> dict[str, Any]:
        prefixes = ("IsolationForest", "OneClassSVM", "RandomForestClassifier", "GradientBoostingClassifier")
        return {key: value for key, value in self.objects.items() if key.startswith(prefixes)}

    def prepare(self, frame: pd.DataFrame) -> np.ndarray:
        missing = [column for column in self.features if column not in frame]
        if missing:
            raise ValueError(f"в датасете нет признаков: {missing[:8]}")
        leaks = [column for column in self.features if column.startswith(("gt_", "true_", "label"))]
        if leaks:
            raise ValueError(f"утечка ground truth в признаках: {leaks}")
        values = frame[self.features].to_numpy(dtype=float)
        if self.imputer is not None:
            values = self.imputer.transform(values)
        if self.scaler is not None:
            values = self.scaler.transform(values)
        return values

    def score(self, name: str, values: np.ndarray) -> np.ndarray:
        model = self.objects[name]
        if hasattr(model, "predict_proba"):
            return model.predict_proba(values)[:, 1]
        if hasattr(model, "score_samples"):
            return -model.score_samples(values)
        if hasattr(model, "decision_function"):
            return -model.decision_function(values)
        raise TypeError(f"{name}: нет способа получить score")


def evaluate_run(run_dir: str, dataset_path: str) -> list[dict]:
    run = LoadedRun(run_dir)
    frame = pd.read_csv(dataset_path).reset_index(drop=True)
    if LABEL_COL not in frame:
        raise ValueError(f"в {dataset_path} нет колонки {LABEL_COL}")
    values = run.prepare(frame)
    labels = frame[LABEL_COL].to_numpy().astype(int)
    rows = []
    for name in run.detectors():
        try:
            scores = run.score(name, values)
            metrics = score_metrics(labels, scores, frame)
        except Exception as exc:  # noqa: BLE001
            rows.append({"model": name, "error": str(exc)})
            continue
        metrics.update(model=name, dataset=os.path.basename(dataset_path), run=os.path.basename(run_dir))
        rows.append(metrics)
    output = os.path.join(run_dir, "transfer_metrics.csv")
    pd.DataFrame(rows).to_csv(output, mode="a", index=False,
                              header=not os.path.exists(output))
    return rows
