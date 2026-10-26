"""Scientific training adapter for the Open5GS-calibrated Diameter simulator.

This module reuses the v1 model zoo but replaces the row-random split with a
grouped holdout. Attack episodes stay intact; normal traffic is grouped in
contiguous time blocks shared by all nodes so zone/temporal neighbours do not
leak across train/validation/test.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder

import diameter_training_v1 as base

# Re-export the existing model/trainer API expected by ms_multiseed_core.
from diameter_training_v1 import *  # noqa: F401,F403,E402


NORMAL_ATTACK_VALUES = {"none", "normal", "", "nan", "None"}


@dataclass
class TrainingConfig(base.TrainingConfig):
    split_normal_block_intervals: int = 120  # 30 min at 15 s intervals
    split_max_tries: int = 256
    split_min_test_episodes: int = 2
    split_min_test_attack_types: int = 2


class DataPipeline(base.DataPipeline):
    """Diameter feature pipeline with episode/time-block-disjoint splits."""

    def __init__(self, config: TrainingConfig):
        super().__init__(config)
        self.split_report = {}

    @staticmethod
    def _is_attack(series: pd.Series) -> pd.Series:
        s = series.astype(str)
        return ~s.isin(NORMAL_ATTACK_VALUES)

    def _annotate_groups(self, df: pd.DataFrame) -> pd.DataFrame:
        if "interval" not in df.columns or "node_id" not in df.columns:
            raise ValueError("grouped split requires interval and node_id columns")

        out = df.copy()
        out["_row_order"] = np.arange(len(out))
        out["attack_episode_id"] = ""
        attack_mask = self._is_attack(out[self.ATTACK_COL])

        # An attack episode is contiguous in time for a concrete target node and
        # attack type. This prevents one episode from being sliced across splits.
        attacked = out.loc[attack_mask, [
            "interval", "node_id", self.ATTACK_COL
        ]].sort_values([self.ATTACK_COL, "node_id", "interval"])
        for (attack, node), idx in attacked.groupby(
                [self.ATTACK_COL, "node_id"], sort=False).groups.items():
            rows = out.loc[list(idx)].sort_values("interval")
            gaps = rows["interval"].diff().fillna(1).ne(1).cumsum()
            for run_id, run_idx in rows.groupby(gaps).groups.items():
                eid = f"A:{attack}:node{int(node)}:run{int(run_id)}"
                out.loc[list(run_idx), "attack_episode_id"] = eid

        block = int(self.config.split_normal_block_intervals)
        if block <= 0:
            raise ValueError("split_normal_block_intervals must be positive")
        normal_group = "N:" + (out["interval"].astype(int) // block).astype(str)
        out["split_group"] = np.where(
            attack_mask,
            out["attack_episode_id"],
            normal_group,
        )
        return out.sort_values("_row_order").drop(columns="_row_order")

    @staticmethod
    def _split_ok(train, val, test, cfg) -> bool:
        parts = (train, val, test)
        if any(len(p) == 0 for p in parts):
            return False
        if any(p["is_anomaly"].nunique() < 2 for p in parts):
            return False

        gs = [set(p["split_group"].astype(str)) for p in parts]
        if gs[0] & gs[1] or gs[0] & gs[2] or gs[1] & gs[2]:
            return False

        eps = test.loc[test["is_anomaly"] == 1, "attack_episode_id"]
        eps = {x for x in eps.astype(str) if x}
        if len(eps) < int(cfg.split_min_test_episodes):
            return False

        at = test.loc[test["is_anomaly"] == 1, "attack_type"].astype(str)
        if at.nunique() < int(cfg.split_min_test_attack_types):
            return False
        return True

    def _grouped_split(self, df: pd.DataFrame):
        groups = df["split_group"].astype(str).to_numpy()
        val_rel = self.config.val_size / max(
            1e-9, (1.0 - self.config.test_size)
        )

        for attempt in range(int(self.config.split_max_tries)):
            seed1 = int(self.config.seed) + 7919 * attempt
            gss1 = GroupShuffleSplit(
                n_splits=1,
                test_size=self.config.test_size,
                random_state=seed1,
            )
            trv_idx, te_idx = next(gss1.split(df, df[self.LABEL_COL], groups))
            train_val = df.iloc[trv_idx]
            test = df.iloc[te_idx]

            g2 = train_val["split_group"].astype(str).to_numpy()
            gss2 = GroupShuffleSplit(
                n_splits=1,
                test_size=val_rel,
                random_state=seed1 + 104729,
            )
            tr_idx, va_idx = next(
                gss2.split(train_val, train_val[self.LABEL_COL], g2)
            )
            train = train_val.iloc[tr_idx]
            val = train_val.iloc[va_idx]

            if self._split_ok(train, val, test, self.config):
                return (
                    train.sort_values(["interval", "node_id"]).copy(),
                    val.sort_values(["interval", "node_id"]).copy(),
                    test.sort_values(["interval", "node_id"]).copy(),
                    attempt,
                )

        raise RuntimeError(
            "could not construct episode/time-block-disjoint split satisfying "
            "class and test-episode constraints"
        )

    def load_and_prepare(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        df = pd.read_csv(self.config.data_path)
        print(
            f"Loaded {len(df)} records, anomalies: {df[self.LABEL_COL].sum()} "
            f"({100 * df[self.LABEL_COL].mean():.2f}%)"
        )
        print(f"Attack distribution:\n{df[self.ATTACK_COL].value_counts()}\n")

        if "node_type" in df.columns and "node_type_enc" not in df.columns:
            df["node_type_enc"] = LabelEncoder().fit_transform(df["node_type"])

        self._build_feature_cols(df)
        for c in self.feature_cols:
            if c.startswith("gt_"):
                raise ValueError(f"GT column '{c}' in feature set!")

        df = self._annotate_groups(df)
        train, val, test, attempt = self._grouped_split(df)

        def episodes(p):
            x = p.loc[p[self.LABEL_COL] == 1, "attack_episode_id"].astype(str)
            return sorted({v for v in x if v})

        group_sets = {
            "train": set(train["split_group"].astype(str)),
            "val": set(val["split_group"].astype(str)),
            "test": set(test["split_group"].astype(str)),
        }
        self.split_report = {
            "mode": "episode_and_time_block_group_holdout",
            "normal_block_intervals": int(
                self.config.split_normal_block_intervals
            ),
            "attempt": int(attempt),
            "group_overlap": bool(
                (group_sets["train"] & group_sets["val"])
                or (group_sets["train"] & group_sets["test"])
                or (group_sets["val"] & group_sets["test"])
            ),
            "rows": {
                "train": int(len(train)),
                "val": int(len(val)),
                "test": int(len(test)),
            },
            "anomaly_rate": {
                "train": float(train[self.LABEL_COL].mean()),
                "val": float(val[self.LABEL_COL].mean()),
                "test": float(test[self.LABEL_COL].mean()),
            },
            "episodes": {
                "train": episodes(train),
                "val": episodes(val),
                "test": episodes(test),
            },
            "test_attack_types": sorted(
                test.loc[test[self.LABEL_COL] == 1, self.ATTACK_COL]
                .astype(str).unique().tolist()
            ),
        }

        print(f"Observable features: {len(self.feature_cols)}")
        print(
            "Grouped split — "
            f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}"
        )
        print(
            "Attack episodes — "
            f"Train: {len(self.split_report['episodes']['train'])}, "
            f"Val: {len(self.split_report['episodes']['val'])}, "
            f"Test: {len(self.split_report['episodes']['test'])}"
        )
        print(
            f"Test attack types: {self.split_report['test_attack_types']}"
        )
        print(f"Group overlap: {self.split_report['group_overlap']}\n")
        return train, val, test
