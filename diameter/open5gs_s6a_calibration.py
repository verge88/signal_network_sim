"""Open5GS S6a software-in-the-loop calibration for Diameter simulator.

The calibrated quantity is intentionally narrow and physically interpretable:
Diameter protocol bytes carried by one S6a transaction (request + answer).
The existing simulator's S6a counters represent transactions such as AIR and
ULR, so this replaces the legacy fixed 500-byte placeholder only for S6a
link-load accounting. TCP/IP framing is retained as capture evidence but is
not fitted because a reassembled Diameter message may span multiple segments.

Traffic-rate priors, command mix, attack priors, masking, topology capacities
and non-S6a traffic are not fitted to the Open5GS integration-test workload.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from .diameter_sim_v1 import SimulationConfig
except ImportError:
    from diameter_sim_v1 import SimulationConfig


SCHEMA = "diameter-open5gs-s6a-calibration-v1"
S6A_APP_ID = 16777251
COMMANDS = {
    316: "ULR",
    317: "CLR",
    318: "AIR",
    319: "IDR",
    320: "DSR",
    321: "PUR",
    322: "RSR",
    323: "NOR",
}
TRACE_COLUMNS = (
    "t",
    "frame_len",
    "ip_len",
    "diameter_len",
    "cmd_code",
    "is_request",
    "app_id",
    "hop_by_hop",
    "end_to_end",
)


def _parse_bool(value) -> Optional[bool]:
    if pd.isna(value):
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes"}:
        return True
    if text in {"0", "false", "no"}:
        return False
    try:
        return bool(int(float(text)))
    except (TypeError, ValueError):
        return None


def load_trace(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in TRACE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"trace is missing columns: {missing}")

    out = df.loc[:, TRACE_COLUMNS].copy()
    for col in TRACE_COLUMNS:
        if col == "is_request":
            continue
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["is_request"] = out["is_request"].map(_parse_bool)
    out = out.dropna(
        subset=[
            "t", "frame_len", "ip_len", "diameter_len", "cmd_code",
            "is_request", "app_id", "hop_by_hop", "end_to_end",
        ]
    )
    out = out[
        (out["app_id"].astype(int) == S6A_APP_ID)
        & out["cmd_code"].astype(int).isin(COMMANDS)
        & (out["ip_len"] > 0)
        & (out["diameter_len"] > 0)
    ].sort_values("t").reset_index(drop=True)

    if out.empty:
        raise ValueError("trace contains no usable S6a Diameter frames")
    return out


def pair_transactions(frames: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["cmd_code", "hop_by_hop", "end_to_end"]
    for key, grp in frames.groupby(keys, sort=False):
        grp = grp.sort_values("t")
        reqs = grp[grp["is_request"] == True]  # noqa: E712
        ans = grp[grp["is_request"] == False]  # noqa: E712
        if reqs.empty or ans.empty:
            continue
        # IDs should identify one transaction. If capture duplicates exist,
        # use the first request and first subsequent answer.
        req = reqs.iloc[0]
        after = ans[ans["t"] >= req["t"]]
        if after.empty:
            continue
        rsp = after.iloc[0]
        cmd_code = int(req["cmd_code"])
        req_ip = float(req["ip_len"])
        rsp_ip = float(rsp["ip_len"])
        req_d = float(req["diameter_len"])
        rsp_d = float(rsp["diameter_len"])
        total_ip = req_ip + rsp_ip
        total_d = req_d + rsp_d
        rows.append({
            "request_t": float(req["t"]),
            "answer_t": float(rsp["t"]),
            "command": COMMANDS[cmd_code],
            "cmd_code": cmd_code,
            "request_ip_bytes": req_ip,
            "answer_ip_bytes": rsp_ip,
            "transaction_ip_bytes": total_ip,
            "request_diameter_bytes": req_d,
            "answer_diameter_bytes": rsp_d,
            "transaction_diameter_bytes": total_d,
            "rtt_ms": float((rsp["t"] - req["t"]) * 1000.0),
        })
    tx = pd.DataFrame(rows)
    if tx.empty:
        raise ValueError("no complete S6a request/answer transactions found")
    return tx.sort_values("request_t").reset_index(drop=True)


def chronological_split(
    tx: pd.DataFrame,
    train_fraction: float = 0.60,
    *,
    min_train: int = 8,
    min_holdout: int = 8,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not 0.2 <= float(train_fraction) <= 0.8:
        raise ValueError("train_fraction must be in [0.2, 0.8]")
    n = len(tx)
    cut = int(np.floor(n * float(train_fraction)))
    if cut < min_train or n - cut < min_holdout:
        raise ValueError(
            f"need at least {min_train} fit and {min_holdout} holdout "
            f"transactions; got {cut}/{n-cut}"
        )
    return (
        tx.iloc[:cut].copy().reset_index(drop=True),
        tx.iloc[cut:].copy().reset_index(drop=True),
    )


def _quantiles(values: Iterable[float]) -> Dict[str, float]:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return {"p10": float("nan"), "p50": float("nan"), "p90": float("nan")}
    return {
        "p10": float(np.quantile(arr, 0.10)),
        "p50": float(np.quantile(arr, 0.50)),
        "p90": float(np.quantile(arr, 0.90)),
    }


def fit_profile(
    fit: pd.DataFrame,
    holdout: pd.DataFrame,
    *,
    min_command_fit: int = 3,
) -> Tuple[Dict, pd.DataFrame, pd.DataFrame]:
    overall = float(fit["transaction_diameter_bytes"].mean())
    if not np.isfinite(overall) or overall <= 0:
        raise ValueError("invalid fitted S6a transaction size")

    command_sizes: Dict[str, float] = {}
    command_fit_rows = []
    for command, grp in fit.groupby("command"):
        n = int(len(grp))
        mean_bytes = float(grp["transaction_diameter_bytes"].mean())
        command_fit_rows.append({
            "command": command,
            "fit_transactions": n,
            "fit_mean_diameter_bytes": mean_bytes,
            "fit_median_diameter_bytes": float(
                grp["transaction_diameter_bytes"].median()
            ),
            "fit_mean_rtt_ms": float(grp["rtt_ms"].mean()),
        })
        if n >= min_command_fit:
            command_sizes[str(command)] = mean_bytes

    pred = holdout.copy()
    pred["predicted_transaction_diameter_bytes"] = [
        command_sizes.get(str(cmd), overall) for cmd in pred["command"]
    ]
    pred["abs_error_bytes"] = (
        pred["predicted_transaction_diameter_bytes"]
        - pred["transaction_diameter_bytes"]
    ).abs()
    pred["relative_error"] = (
        pred["abs_error_bytes"]
        / pred["transaction_diameter_bytes"].clip(lower=1.0)
    )

    per_command = []
    for command, grp in pred.groupby("command"):
        per_command.append({
            "command": command,
            "holdout_transactions": int(len(grp)),
            "holdout_mean_diameter_bytes": float(
                grp["transaction_diameter_bytes"].mean()
            ),
            "predicted_diameter_bytes": float(
                grp["predicted_transaction_diameter_bytes"].iloc[0]
            ),
            "mae_bytes": float(grp["abs_error_bytes"].mean()),
            "median_relative_error": float(
                grp["relative_error"].median()
            ),
        })

    legacy = 500.0
    # Representative MME defaults: 65% S6a, 35% non-S6a. This multiplier is
    # diagnostic only; topology/traffic priors themselves are not fitted.
    mme_load_multiplier = (
        0.65 * overall + 0.35 * legacy
    ) / legacy

    fit_rtt = _quantiles(fit["rtt_ms"])
    holdout_rtt = _quantiles(holdout["rtt_ms"])
    fit_size = _quantiles(fit["transaction_diameter_bytes"])
    holdout_size = _quantiles(holdout["transaction_diameter_bytes"])

    profile = {
        "schema": SCHEMA,
        "scope": "Open5GS EPC S6a transaction wire-size calibration",
        "source": "Open5GS EPC attach integration tests (MME/HSS, freeDiameter)",
        "s6a_transaction_size_bytes": overall,
        "s6a_command_transaction_size_bytes": command_sizes,
        "fit_transactions": int(len(fit)),
        "holdout_transactions": int(len(holdout)),
        "fit_diameter_bytes_quantiles": fit_size,
        "holdout_diameter_bytes_quantiles": holdout_size,
        "fit_rtt_ms_quantiles": fit_rtt,
        "holdout_rtt_ms_quantiles": holdout_rtt,
        "holdout_mae_bytes": float(pred["abs_error_bytes"].mean()),
        "holdout_median_relative_error": float(
            pred["relative_error"].median()
        ),
        "holdout_p90_relative_error": float(
            np.quantile(pred["relative_error"], 0.90)
        ),
        "representative_mme_wire_load_multiplier_vs_legacy_500B": float(
            mme_load_multiplier
        ),
        "untouched_model_families": [
            "traffic-rate priors and time-of-day profile",
            "S6a command distribution",
            "Gx/Rx/S13/non-S6a message sizing",
            "attack priors and attack semantics",
            "masking/compromise behavior",
            "topology and configured link capacities",
            "CU polling and integrity parameters",
            "AR traffic dynamics",
        ],
        "diagnostic_only": [
            "S6a request-answer RTT is measured but is not used to overwrite "
            "link propagation delay because it includes MME/HSS processing "
            "and database time.",
            "Captured IP frame lengths are retained as evidence but are not "
            "used to infer per-transaction transport overhead because Diameter "
            "messages may be reassembled from multiple TCP segments."
        ],
    }
    return (
        profile,
        pred.reset_index(drop=True),
        pd.DataFrame(command_fit_rows).sort_values("command").reset_index(drop=True),
    )


def calibrated_config(base: SimulationConfig, profile: Dict) -> SimulationConfig:
    fit = profile.get("fit", profile)
    overall = float(fit["s6a_transaction_size_bytes"])
    if not np.isfinite(overall) or overall <= 0:
        raise ValueError("s6a_transaction_size_bytes must be finite and positive")
    command_sizes = {
        str(k).upper(): float(v)
        for k, v in fit.get("s6a_command_transaction_size_bytes", {}).items()
    }
    return replace(
        base,
        s6a_transaction_size_bytes=overall,
        s6a_command_transaction_size_bytes=command_sizes,
    )


def run_calibration(
    trace_path: str | Path,
    out_dir: str | Path,
    *,
    train_fraction: float = 0.60,
    min_command_fit: int = 3,
) -> Dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    frames = load_trace(trace_path)
    tx = pair_transactions(frames)
    fit_tx, hold_tx = chronological_split(tx, train_fraction)
    fit, hold_eval, cmd_fit = fit_profile(
        fit_tx, hold_tx, min_command_fit=min_command_fit
    )

    result = {
        "schema": SCHEMA,
        "method": {
            "design": "chronological fit/holdout on complete S6a transactions",
            "train_fraction": float(train_fraction),
            "pair_key": "command-code + hop-by-hop-id + end-to-end-id",
            "calibrated_quantity": (
                "request + answer Diameter protocol bytes per modeled S6a transaction"
            ),
        },
        "fit": fit,
        "claim_boundary": (
            "This is software-in-the-loop calibration against an independent "
            "Open5GS MME/HSS Diameter implementation. It calibrates S6a "
            "transaction wire size only. It is not operator traffic and does "
            "not calibrate workload rates, attack priors, or S6a command mix."
        ),
    }

    (out / "calibrated_profile.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    frames.to_csv(out / "s6a_frames.csv", index=False)
    tx.to_csv(out / "s6a_transactions.csv", index=False)
    fit_tx.to_csv(out / "fit_transactions.csv", index=False)
    hold_eval.to_csv(out / "holdout_validation.csv", index=False)
    cmd_fit.to_csv(out / "command_fit_summary.csv", index=False)

    md = f"""# Diameter / Open5GS S6a calibration

- Source: Open5GS EPC attach integration tests (MME/HSS + freeDiameter)
- Complete S6a transactions: {len(tx)}
- Fit transactions: {len(fit_tx)}
- Holdout transactions: {len(hold_tx)}
- Fitted mean S6a Diameter bytes / transaction: {fit['s6a_transaction_size_bytes']:.3f}
- Holdout MAE: {fit['holdout_mae_bytes']:.3f} bytes
- Holdout median relative error: {fit['holdout_median_relative_error']:.6f}
- Holdout p90 relative error: {fit['holdout_p90_relative_error']:.6f}
- Representative MME Diameter-load multiplier vs legacy 500 B placeholder: {fit['representative_mme_wire_load_multiplier_vs_legacy_500B']:.3f}x
- Calibrated commands: {', '.join(sorted(fit['s6a_command_transaction_size_bytes'])) or 'overall fallback only'}

**Claim boundary:** {result['claim_boundary']}
"""
    (out / "summary.md").write_text(md, encoding="utf-8")
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Calibrate Diameter S6a transaction sizes from Open5GS"
    )
    ap.add_argument("--trace", required=True)
    ap.add_argument("--out", default="diameter_calibration_open5gs")
    ap.add_argument("--train-fraction", type=float, default=0.60)
    ap.add_argument("--min-command-fit", type=int, default=3)
    args = ap.parse_args(argv)

    result = run_calibration(
        args.trace,
        args.out,
        train_fraction=args.train_fraction,
        min_command_fit=args.min_command_fit,
    )
    fit = result["fit"]
    print(json.dumps({
        "profile": str(Path(args.out) / "calibrated_profile.json"),
        "s6a_transaction_size_bytes": fit["s6a_transaction_size_bytes"],
        "s6a_command_transaction_size_bytes":
            fit["s6a_command_transaction_size_bytes"],
        "holdout_mae_bytes": fit["holdout_mae_bytes"],
        "holdout_median_relative_error":
            fit["holdout_median_relative_error"],
        "representative_mme_wire_load_multiplier":
            fit["representative_mme_wire_load_multiplier_vs_legacy_500B"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
