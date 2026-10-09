"""Live signature verification for prospective Open5GS KIE trial.

Must run AFTER stopping collectors while the original temporary key is present.
The key itself is never stored in uploaded artifacts.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from open5gs_kie_sidecar import verify_report


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def verify(reports_path: Path, truth_path: Path, secret: bytes, output: Path) -> dict:
    reports = load(reports_path)
    truth = load(truth_path)
    if not reports or not truth:
        raise ValueError("missing report/ground truth telemetry")
    identities = lambda records: [
        (str(x["nf_id"]).lower(), int(x["sequence"]), float(x["ts"]))
        for x in records
    ]
    r_ids, t_ids = identities(reports), identities(truth)
    if len(set(r_ids)) != len(r_ids) or len(set(t_ids)) != len(t_ids):
        raise ValueError("duplicated sidecar sample identity")
    if set(r_ids) != set(t_ids):
        raise ValueError("signed reports and collector truth do not pair exactly")
    invalid = sum(not verify_report(row, secret) for row in reports)
    result = {
        "report_count": len(reports),
        "invalid_signature_count": invalid,
        "all_signatures_valid": invalid == 0,
        "paired_collector_samples": len(truth),
        "all_reports_truth_paired": True,
        "verified_live_before_artifact_upload": True,
        "hmac_key_exported": False,
    }
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if invalid:
        raise ValueError(f"invalid signed reports: {invalid}")
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--reports", type=Path, required=True)
    p.add_argument("--truth", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    secret = os.environ.get("OPEN5GS_KIE_SECRET")
    if not secret:
        raise SystemExit("live HMAC key missing; offline bypass not permitted")
    print(json.dumps(verify(args.reports, args.truth, secret.encode(), args.output)))


if __name__ == "__main__":
    main()
