from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import FleetConfig


SERVICE_BY_TYPE = {
    "AMF": "namf-comm",
    "SMF": "nsmf-pdusession",
    "UDM": "nudm-sdm",
    "AUSF": "nausf-auth",
    "PCF": "npcf-smpolicycontrol",
}

BASE_RPS = {
    "AMF": 130.0,
    "SMF": 95.0,
    "UDM": 65.0,
    "AUSF": 45.0,
    "PCF": 55.0,
}


@dataclass(frozen=True)
class AttackSpec:
    name: str
    severity: float = 1.0


class FiveGCoreSimulator:
    """Synthetic 5G Core/SBA telemetry generator.

    The generator represents two observation channels:
      * independently observed SBI telemetry (e.g. SCP/tap),
      * NF self-report carried by the proposed KIE/control channel.

    NRF-derived metadata is represented by NFType, S-NSSAI, PLMN, locality,
    service name and lifecycle counters. Dynamic zone membership is defined by
    currently registered NFs sharing this tuple.
    """

    def __init__(self, cfg: FleetConfig, seed: int):
        self.cfg = cfg
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.window_id = 0
        self.fleet = self._build_fleet()
        self.down_until: Dict[str, int] = {}
        self._zone_index = self._make_zone_index()

    def _build_fleet(self) -> pd.DataFrame:
        rows: List[dict] = []
        for nf_type in self.cfg.nf_types:
            service = SERVICE_BY_TYPE[nf_type]
            base = BASE_RPS[nf_type]
            for snssai in self.cfg.slices:
                for locality in self.cfg.localities:
                    for replica in range(self.cfg.replicas_per_zone):
                        capacity = 0.92 + 0.08 * replica
                        nf_id = f"{nf_type.lower()}-{snssai}-{locality}-{replica}"
                        rows.append(
                            dict(
                                nf_id=nf_id,
                                nf_type=nf_type,
                                snssai=snssai,
                                plmn=self.cfg.plmn,
                                locality=locality,
                                service=service,
                                replica=replica,
                                base_rps=base * capacity,
                                capacity=capacity,
                            )
                        )
        return pd.DataFrame(rows)

    def _make_zone_index(self) -> Dict[Tuple[str, str, str, str, str], List[str]]:
        out: Dict[Tuple[str, str, str, str, str], List[str]] = {}
        for r in self.fleet.to_dict("records"):
            key = (r["nf_type"], r["snssai"], r["plmn"], r["locality"], r["service"])
            out.setdefault(key, []).append(r["nf_id"])
        return out

    @property
    def nf_ids(self) -> List[str]:
        return self.fleet["nf_id"].tolist()

    def _registered(self, nf_id: str) -> bool:
        return self.down_until.get(nf_id, -1) <= self.window_id

    def _maybe_churn(self) -> None:
        if self.rng.random() < self.cfg.churn_probability:
            candidates = [n for n in self.nf_ids if self._registered(n)]
            if candidates:
                nf = str(self.rng.choice(candidates))
                self.down_until[nf] = self.window_id + self.cfg.churn_hold_windows

    def _pick_targets(self, scenario: str) -> List[str]:
        z = (
            "AMF",
            self.cfg.slices[0],
            self.cfg.plmn,
            self.cfg.localities[0],
            SERVICE_BY_TYPE["AMF"],
        )
        peers = [n for n in self._zone_index[z] if self._registered(n)]
        if len(peers) < 3:
            peers = list(self._zone_index[z])
        if scenario == "coordinated_majority":
            return peers[:2]
        if scenario == "nrf_poisoning":
            z2 = (
                "SMF",
                self.cfg.slices[0],
                self.cfg.plmn,
                self.cfg.localities[0],
                SERVICE_BY_TYPE["SMF"],
            )
            return [self._zone_index[z2][0]]
        return [peers[0]]

    def _base_row(self, meta: dict, global_load: float, slice_load: float, locality_load: float) -> dict:
        rng = self.rng
        expected_rps = meta["base_rps"] * global_load * slice_load * locality_load
        observed_rps = max(0.1, expected_rps * (1.0 + rng.normal(0.0, 0.045)))
        error_rate = max(0.0, rng.normal(0.009, 0.0025))
        http2_rst_rate = max(0.0, rng.normal(0.0013, 0.00055))
        latency_ms = max(0.1, rng.normal(4.0, 0.45))
        concurrent_streams = max(1.0, observed_rps * rng.normal(0.36, 0.025))
        authz_403_rate = max(0.0, rng.normal(0.0008, 0.00035))
        token_req_rate = max(0.0, rng.normal(0.015, 0.003))
        discovery_rate = max(0.0, rng.normal(0.004, 0.001))
        profile_update_rate = max(0.0, rng.normal(0.0015, 0.0005))
        cross_slice_rate = max(0.0, rng.normal(0.00045, 0.00018))
        report_rps = observed_rps * (1.0 + rng.normal(0.0, 0.012))
        kie_rtt_ms = max(0.1, rng.normal(2.0, 0.22))
        kie_loss = 1.0 if rng.random() < 0.0015 else 0.0
        kie_auth_fail = 0.0
        heartbeat_age_s = max(0.1, rng.normal(self.cfg.window_s, 0.8))
        nrf_register_rate = max(0.0, rng.normal(0.0006, 0.00022))
        nrf_delete_rate = max(0.0, rng.normal(0.0003, 0.00012))
        scope_mismatch_rate = max(0.0, rng.normal(0.0004, 0.00018))

        return dict(
            **meta,
            window_id=self.window_id,
            registered=1,
            expected_rps=expected_rps,
            observed_rps=observed_rps,
            report_rps=report_rps,
            error_rate=error_rate,
            http2_rst_rate=http2_rst_rate,
            latency_ms=latency_ms,
            concurrent_streams=concurrent_streams,
            authz_403_rate=authz_403_rate,
            token_req_rate=token_req_rate,
            discovery_rate=discovery_rate,
            profile_update_rate=profile_update_rate,
            cross_slice_rate=cross_slice_rate,
            kie_rtt_ms=kie_rtt_ms,
            kie_loss=kie_loss,
            kie_auth_fail=kie_auth_fail,
            heartbeat_age_s=heartbeat_age_s,
            nrf_register_rate=nrf_register_rate,
            nrf_delete_rate=nrf_delete_rate,
            scope_mismatch_rate=scope_mismatch_rate,
            attack="normal",
            label=0,
        )

    def _apply_attack(self, row: dict, attack: str, severity: float, progress: float) -> None:
        rng = self.rng
        if attack == "compromised_lie":
            clean = row["observed_rps"]
            row["observed_rps"] *= 1.0 + 0.32 * severity
            row["report_rps"] = clean * (1.0 + rng.normal(0.0, 0.01))
        elif attack == "compromised_truth":
            row["observed_rps"] *= 1.0 + 0.32 * severity
            row["report_rps"] = row["observed_rps"] * (1.0 + rng.normal(0.0, 0.01))
        elif attack == "slow_drift_lie":
            clean = row["observed_rps"]
            drift = 0.02 + (0.18 * severity * progress)
            row["observed_rps"] *= 1.0 + drift
            row["report_rps"] = clean * (1.0 + rng.normal(0.0, 0.01))
        elif attack == "fake_nf":
            row["kie_auth_fail"] = 1.0
            row["kie_loss"] = max(row["kie_loss"], 0.5)
            row["heartbeat_age_s"] += 1.6 * self.cfg.window_s * severity
            row["nrf_register_rate"] += 0.08 * severity
            row["authz_403_rate"] += 0.09 * severity
        elif attack == "rapid_reset":
            row["http2_rst_rate"] += 0.025 * severity
            row["error_rate"] += 0.045 * severity
            row["latency_ms"] += 3.5 * severity
            row["concurrent_streams"] *= 1.0 + 0.75 * severity
        elif attack == "oauth_abuse":
            row["authz_403_rate"] += 0.09 * severity
            row["token_req_rate"] += 0.25 * severity
            row["scope_mismatch_rate"] += 0.12 * severity
        elif attack == "nrf_poisoning":
            row["nrf_register_rate"] += 0.12 * severity
            row["nrf_delete_rate"] += 0.05 * severity
            row["profile_update_rate"] += 0.16 * severity
            row["discovery_rate"] += 0.14 * severity
        elif attack == "cross_slice":
            row["cross_slice_rate"] += 0.07 * severity
            row["authz_403_rate"] += 0.018 * severity
        elif attack == "coordinated_majority":
            row["observed_rps"] *= 1.0 + 0.30 * severity
            row["report_rps"] = row["observed_rps"] * (1.0 + rng.normal(0.0, 0.01))
        else:
            raise ValueError(f"unknown attack: {attack}")
        row["attack"] = attack
        row["label"] = 1

    def generate(self, n_windows: int, attack: Optional[AttackSpec] = None) -> pd.DataFrame:
        rows: List[dict] = []
        scenario = attack.name if attack is not None else "normal"
        severity = attack.severity if attack is not None else 0.0
        targets = self._pick_targets(scenario) if attack is not None else []

        for k in range(n_windows):
            self._maybe_churn()
            phase = 2.0 * np.pi * ((self.window_id % 2880) / 2880.0)
            global_load = float(np.exp(self.rng.normal(0.0, 0.055)) * (1.0 + 0.15 * np.sin(phase)))
            slice_loads = {s: float(np.exp(self.rng.normal(0.0, 0.035))) for s in self.cfg.slices}
            locality_loads = {l: float(np.exp(self.rng.normal(0.0, 0.03))) for l in self.cfg.localities}

            for meta in self.fleet.to_dict("records"):
                if not self._registered(meta["nf_id"]):
                    continue
                row = self._base_row(
                    meta,
                    global_load=global_load,
                    slice_load=slice_loads[meta["snssai"]],
                    locality_load=locality_loads[meta["locality"]],
                )
                if attack is not None and meta["nf_id"] in targets:
                    progress = 0.0 if n_windows <= 1 else k / (n_windows - 1)
                    self._apply_attack(row, scenario, severity, progress)
                rows.append(row)

            self.window_id += 1

        frame = pd.DataFrame(rows)
        if frame.empty:
            return frame
        frame["zone_key"] = (
            frame["nf_type"].astype(str)
            + "|"
            + frame["snssai"].astype(str)
            + "|"
            + frame["plmn"].astype(str)
            + "|"
            + frame["locality"].astype(str)
            + "|"
            + frame["service"].astype(str)
        )
        return frame
