"""Paired retrospective S10 assessment, using recorded measurements only."""
import json
from pathlib import Path

import numpy as np

from .cross_scenario_detector import SLOTS, flags
from .cross_scenario_replay import aligned, read_csv, sha256, write_json
from .per_type_detector import CONTROL_NF
from .per_type_replay import Observations, load_model, confirm_stream
from .v9_cross_scenario_audit import KEYS, metrics
from experiments.integrated_5g.control import combine


def report():
    obs = Observations("data")
    cap = obs.capture
    model, meta = load_model("frozen")
    if cap.seed != 91401 or cap.windows != 3600 or cap.seed in meta["consumed_seeds"] or cap.run in model.fit_runs + model.calibration_runs:
        raise ValueError("Unexpected or overlapping archive")
    scores = model.score_panel(cap.values, cap.allowed)
    base = model.decide(cap.local, obs.loss, cap.allowed, *scores)
    confirm_stream(model, obs, base)
    fused = base.copy()
    measured = cap.original_alarm("TG_V10")
    reference = measured.copy()
    slot = SLOTS.index(CONTROL_NF)
    key = (cap.path / "lab_control_key.hex").read_text().strip()
    secret = bytes.fromhex(key)
    states = []
    with (cap.path / "control_events.jsonl").open() as controls, (cap.path / "v10_secure_decisions.jsonl").open() as records:
        for w, (c, r) in enumerate(zip(controls, records, strict=True)):
            c, r = json.loads(c), json.loads(r)
            if (w >= cap.windows or w != c["window_id"] or w != r["window_id"]
                    or r["run_id"] != cap.run or r["nf_id"] != CONTROL_NF):
                raise ValueError("Unaligned independent evidence")

            def verified(prior, at):
                return combine(bool(prior), c["delivered"], c["nonce"], c["expected"],
                               c["instance"], secret, at)["K"]

            if c["completed"] > r["decision_time"]:
                raise ValueError("Reference decision precedes receipt")
            historical = verified(measured[w, slot], r["decision_time"])
            if (historical != r["control_state"] or (historical == "alarm") != r["security_alarm"]
                    or (historical == "unknown") != r["availability_unknown"]):
                raise ValueError("Independent verification disagrees with historical reference")
            reference[w, slot] = historical == "alarm"
            state = verified(False, obs.times[w])
            states.append(state)
            fused[w, slot] |= state == "alarm"
    if len(states) != cap.windows:
        raise ValueError("Incomplete independent evidence")
    # Ground truth joins only after all four decision streams have been generated.
    truth, labels = cap.labels()
    frame = aligned(read_csv(cap.path / "labels.csv", [*KEYS, "variant", "episode_id"]), cap.index)
    shape = truth.shape
    variants = frame.variant.to_numpy().reshape(shape)
    episodes = frame.episode_id.to_numpy().reshape(shape)
    scope = truth & (labels == "S10")
    if scope.sum() != 90 or scope[:, slot].sum() != 90 or truth.sum() != 2600:
        raise ValueError("Unexpected exposures")
    methods = {"archived_traffic": measured, "archived_secure": reference,
               "frozen_per_type": base, "frozen_with_external": fused}
    totals = {m: metrics(truth, a) for m, a in methods.items()}
    normal_slot = ~truth[:, slot]
    per_variant = {}
    for v in ("node_config", "node_report_tamper", "node_sbi_abuse"):
        mask = scope[:, slot] & (variants[:, slot] == v)
        ws = np.flatnonzero(mask)
        if len(ws) != 30 or len(set(episodes[ws,slot])) != 1 or np.any(np.diff(ws) != 1):
            raise ValueError("Incomplete episode")
        hits = {}
        for m, a in methods.items():
            detected = ws[a[ws,slot]]
            hits[m] = {"hits": len(detected), "first_offset_windows": int(detected[0]-ws[0]) if len(detected) else None}
        per_variant[v] = {"windows": len(ws), "methods": hits,
                          "external_only": {s:sum(states[w] == s for w in ws) for s in ("alarm","normal","unknown")}}
    by_type = {}
    for name in sorted(set(labels[truth])):
        selected = truth & (labels == name)
        by_type[name] = {"n": int(selected.sum()), **{m:int(a[selected].sum()) for m,a in methods.items()}}
    if len(by_type) != 15:
        raise ValueError("Incomplete scenario coverage")
    out = dict(seed=cap.seed, model=meta["model_sha256"], windows=int(truth.size),
               normal_slot_windows=int(normal_slot.sum()), normal_slot_fp={m:int(a[normal_slot,slot].sum()) for m,a in methods.items()},
               reference_checks=len(states), totals=totals, by_type=by_type, S10=per_variant,
               archive_scope="retrospective; one acquired seed; no independent replicate")
    folder = Path("final-report")
    folder.mkdir()
    write_json(folder/"results.json", out)
    import csv
    with (folder/"s10-windows.csv").open("w", newline="") as stream:
        writer=csv.writer(stream)
        writer.writerow(["window","nf","variant","external_state",*methods])
        for w in np.flatnonzero(scope[:,slot]):
            writer.writerow([int(w),CONTROL_NF,variants[w,slot],states[w],*[int(a[w,slot]) for a in methods.values()]])
    print(json.dumps(out))


if __name__ == "__main__":
    report()
