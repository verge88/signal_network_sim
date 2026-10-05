from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np
import pandas as pd


EPS = 1e-9


@dataclass
class TemporalState:
    alpha: float = 0.04
    mean: Dict[Tuple[str, str], float] = field(default_factory=dict)
    var: Dict[Tuple[str, str], float] = field(default_factory=dict)

    def transform_value(self, nf_id: str, name: str, value: float, floor: float) -> float:
        key = (nf_id, name)
        if key not in self.mean:
            self.mean[key] = value
            self.var[key] = floor * floor
            return 0.0
        m = self.mean[key]
        v = max(self.var[key], floor * floor)
        z = abs(value - m) / np.sqrt(v)
        d = value - m
        self.mean[key] = m + self.alpha * d
        self.var[key] = (1.0 - self.alpha) * v + self.alpha * d * d
        return float(z)


def _robust_z(values: pd.Series, floor: float) -> pd.Series:
    med = float(values.median())
    mad = float((values - med).abs().median())
    scale = max(1.4826 * mad, floor)
    return (values - med).abs() / scale


def add_features(raw: pd.DataFrame, temporal: TemporalState | None = None) -> pd.DataFrame:
    if raw.empty:
        return raw.copy()
    temporal = temporal or TemporalState()
    df = raw.sort_values(["window_id", "nf_id"]).copy()

    df["req_norm"] = df["observed_rps"] / df["base_rps"].clip(lower=EPS)
    df["streams_norm"] = df["concurrent_streams"] / df["observed_rps"].clip(lower=1.0)

    df["report_observed_gap"] = (
        (df["report_rps"] - df["observed_rps"]).abs()
        / df["observed_rps"].clip(lower=1.0)
    )

    temp_req = []
    temp_rst = []
    temp_auth = []
    for row in df.itertuples(index=False):
        temp_req.append(temporal.transform_value(row.nf_id, "req_norm", float(row.req_norm), 0.035))
        temp_rst.append(temporal.transform_value(row.nf_id, "rst", float(row.http2_rst_rate), 0.0008))
        temp_auth.append(temporal.transform_value(row.nf_id, "403", float(row.authz_403_rate), 0.0006))
    df["temporal_req_z"] = temp_req
    df["temporal_rst_z"] = temp_rst
    df["temporal_auth_z"] = temp_auth

    group_cols = ["window_id", "zone_key"]
    df["zone_req_z"] = (
        df.groupby(group_cols, group_keys=False)["req_norm"]
          .transform(lambda s: _robust_z(s, 0.035))
    )
    df["zone_error_z"] = (
        df.groupby(group_cols, group_keys=False)["error_rate"]
          .transform(lambda s: _robust_z(s, 0.003))
    )
    df["zone_rst_z"] = (
        df.groupby(group_cols, group_keys=False)["http2_rst_rate"]
          .transform(lambda s: _robust_z(s, 0.0009))
    )
    df["zone_size"] = df.groupby(group_cols)["nf_id"].transform("count").astype(float)
    return df
