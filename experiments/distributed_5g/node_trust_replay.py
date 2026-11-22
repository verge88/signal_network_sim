"""Replay ten archived scenarios and an independent, synthetic node-trust episode."""
import argparse
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from .cross_scenario_detector import SLOTS
from .per_type_detector import CONTROL_NF
from .per_type_replay import Observations, load_model
from .v9_cross_scenario_audit import SCENARIOS, metrics


def assess(approved, reported, witnessed):
    if witnessed is None:
        return "unknown"
    if witnessed != approved or reported != witnessed:
        return "alarm"
    return "consistent"


# An abstract single-node lifecycle. Each tuple gives phase, duration, trust
# state, approved configuration, actual configuration, local claim, visibility.
PHASES = (
    ("ordinary", 3, False, "A", "A", "A", True),
    ("authorized_update", 3, False, "B", "B", "B", True),
    ("latent", 2, True, "B", "B", "B", True),
    ("altered", 3, True, "B", "X", "X", True),
    ("concealed", 3, True, "B", "X", "B", True),
    ("observer_gap", 2, True, "B", "X", "B", False),
    ("restored", 3, False, "B", "B", "B", True),
)


def episode():
    window = 0
    for phase, duration, compromised, approved, actual, reported, visible in PHASES:
        for _ in range(duration):
            for nf in SLOTS:
                target = nf == CONTROL_NF
                witness = (actual if visible else None) if target else None
                claim = reported if target else "A"
                baseline = approved if target else "A"
                decision = assess(baseline, claim, witness) if target else "outside_scope"
                yield dict(window=window, nf=nf, approved=baseline, reported=claim,
                           witnessed=witness, decision=decision, phase=phase,
                           compromised=target and compromised)
            window += 1


def replay(archive, checkpoint, destination):
    model, manifest = load_model(checkpoint)
    obs = Observations(archive)
    cap = obs.capture
    if cap.seed in manifest["consumed_seeds"] or cap.run in model.fit_runs + model.calibration_runs:
        raise ValueError("Use a held-out archive")
    graph, single, group = model.score_panel(cap.values, cap.allowed)
    alarms = model.decide(cap.local, obs.loss, cap.allowed, graph, single, group)
    truth, labels = cap.labels()
    if set(labels[truth]) != set(SCENARIOS):
        raise ValueError("Expected all ten recorded scenarios")
    by_type = {}
    for name in SCENARIOS:
        selected = truth & (labels == name)
        by_type[name] = {"windows": int(selected.sum()), "detected": int(alarms[selected].sum())}
    rows = list(episode())
    target = [row for row in rows if row["nf"] == CONTROL_NF]
    result = {"archive_run": cap.run, "model_sha256": manifest["model_sha256"],
              "recorded": {"by_type": by_type, "totals": metrics(truth, alarms)},
              "synthetic_S10": {"target": CONTROL_NF, "node_windows": len(rows),
                                "compromised": sum(r["compromised"] for r in target),
                                "detected": sum(r["compromised"] and r["decision"] == "alarm" for r in target),
                                "unknown": sum(r["decision"] == "unknown" for r in target),
                                "false_alarms": sum(not r["compromised"] and r["decision"] == "alarm" for r in target)}}
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    with TemporaryDirectory(dir=destination.parent) as tmp:
        stage = Path(tmp) / "report"
        stage.mkdir()
        with (stage / "S10.csv").open("w", newline="", encoding="utf-8") as out:
            writer = csv.DictWriter(out, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        (stage / "report.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
        stage.rename(destination)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("archive", "checkpoint", "destination"):
        parser.add_argument(arg, type=Path)
    print(json.dumps(replay(**vars(parser.parse_args())), indent=2))
