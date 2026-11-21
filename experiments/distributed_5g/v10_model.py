"""Separate V10 detector: frozen V9 temporal core + verified policy evidence.

This *does not* train on test labels, infer compromised credentials, or claim
to detect arbitrary OS takeover.  TG_V10 is a feature-only classifier for
each NF-window.  A second optional V10_SECURE decision combines TG_V10 with
an existing independently authenticated research-NF configuration report
from slot-4.  Security alarms and missing-observation states stay separate.

No S10 attack labels, episode timing, or variant names enter inference.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from experiments.integrated_5g.control import combine


class V10MultiThreatFusion:
    """Deterministic, explainable V9 + SBI rules; no unvalidated accuracy claim."""

    VERSION = "v10-node-compromise-multithreat-r1"
    RULE_COLUMNS = (
        "profile_policy_violation", "forbidden_operation", "route_mismatch",
        "report_mismatch", "auth_failure",
    )
    BURST_MIN_REQUESTS = 24
    BURST_MIN_ERROR_RATE = 0.25

    def traffic_margin(self, frame):
        frame = pd.DataFrame(frame)
        required = set(self.RULE_COLUMNS) | {"request_count", "error_rate"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"V10 missing independently measured features: {sorted(missing)}")
        evidence = frame[list(self.RULE_COLUMNS) + ["request_count", "error_rate"]]
        if not np.isfinite(evidence.to_numpy(dtype=float)).all():
            raise ValueError("V10 cannot classify missing/nonfinite security evidence")
        any_policy = (frame[list(self.RULE_COLUMNS)].to_numpy(dtype=float) > 0).any(axis=1)
        error_burst = (
            (frame["request_count"].to_numpy(dtype=float) >= self.BURST_MIN_REQUESTS)
            & (frame["error_rate"].to_numpy(dtype=float) >= self.BURST_MIN_ERROR_RATE)
        )
        # Positive = alarm; the independent frozen V9 output stays intact.
        return np.where(any_policy | error_burst, 1.0, -1.0)

    def combine_traffic(self, frozen_v9_margin, frame):
        v9 = np.asarray(frozen_v9_margin, dtype=float)
        policy = self.traffic_margin(frame)
        if v9.ndim != 1 or v9.shape != policy.shape or not np.isfinite(v9).all():
            raise ValueError("V9 and V10 windows are not aligned")
        return np.maximum(v9, policy)

    def assess_external(self, traffic_alarm, control_observation, trusted_key, decision_time):
        """Independent observation at *decision time*, never future data.

        Returns separate security alert and availability status. An unsigned
        edited delivered report is detected by HMAC, a signed genuine changed
        report by its measured configuration digest. Missing/unfresh evidence
        is UNKNOWN unless independently measured traffic already alarms.
        """
        if not isinstance(control_observation, dict):
            raise ValueError("Missing V10 external observation; cannot assume normal")
        state = combine(bool(traffic_alarm),
                        control_observation.get("delivered"),
                        control_observation["nonce"],
                        control_observation["expected"],
                        control_observation["instance"],
                        trusted_key, float(decision_time))["K"]
        if state not in ("normal", "alarm", "unknown"):
            raise ValueError("Unknown independent control state")
        return {
            "traffic_alarm": bool(traffic_alarm),
            "control_state": state,
            "security_alarm": state == "alarm",
            "availability_unknown": state == "unknown",
            "trusted_external_scope": "research-NF slot-4 only",
        }


def derive_from_v9(source_v9, output):
    """Derive NEW hash-pinned V10 bundle, preserving frozen V9 binary weights.

    Only the deterministic V10 wrapper is new. External control keys, training
    labels and attack-generation state are never included in model.joblib.
    """
    from experiments.live_5g.evaluate import load_frozen, sha256
    source_v9, output = Path(source_v9), Path(output)
    if output.exists():
        raise FileExistsError(f"Refuse to overwrite a frozen bundle: {output}")
    base = load_frozen(source_v9)
    old = base["manifest"]
    if old.get("ml_method") != "TG_V9" or "v9_pair_rescue" not in base:
        raise ValueError("V10 requires an authentic frozen TG_V9 model")
    if old.get("temporal_graph_v9", {}).get("heldout_v9_test_used_for_selection") is not False:
        raise ValueError("V9 development/heldout provenance is not authentic")
    model = copy.deepcopy(base)
    manifest = copy.deepcopy(old)
    model["v10_multithreat"] = V10MultiThreatFusion()
    model["ml_method"] = "TG_V10"
    manifest["ml_method"] = "TG_V10"
    manifest["detector"] = "temporal_graph_multithreat_v10"
    manifest["temporal_graph_v10"] = {
        "version": V10MultiThreatFusion.VERSION,
        "parent_method": "TG_V9",
        "parent_frozen_model_sha256": old["model_sha256"],
        "parent_train_runs": old.get("train_runs", []),
        "parent_calibration_runs": old.get("calibration_runs", []),
        "parent_development_runs": old.get("development_runs", []),
        "new_training": False,
        "new_calibration": False,
        "test_labels_used": False,
        "traffic": "max(original TG_V9 margin, policy/forbidden/route/report/auth flags, "
                   "24+ requests and >=25% error rate)",
        "external": "separate at-decision-time, HMAC-authenticated config witness on research NF slot-4",
        "unknown_is_not_security_alarm": True,
        "scope_limit": "No detection guarantee for stolen root/key or fully controlled external observer",
    }
    # Bundle external manifest will be replaced by load_frozen() on every read.
    model["manifest"] = manifest
    output.mkdir(parents=True)
    joblib.dump(model, output / "model.joblib")
    manifest["model_sha256"] = sha256(output / "model.joblib")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    check = load_frozen(output)
    assert check["manifest"]["model_sha256"] == manifest["model_sha256"]
    assert isinstance(check["v10_multithreat"], V10MultiThreatFusion)
    return manifest


def main():
    p = argparse.ArgumentParser(description="Create independent V10 model from frozen V9 checkpoint")
    p.add_argument("--source-v9", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    m = derive_from_v9(a.source_v9, a.out)
    print(json.dumps({
        "method": "TG_V10", "model_sha256": m["model_sha256"],
        "parent_model_sha256": m["temporal_graph_v10"]["parent_frozen_model_sha256"],
        "new_training": False, "new_calibration": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
