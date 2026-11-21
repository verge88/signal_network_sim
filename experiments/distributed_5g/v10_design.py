"""V10 multi-attack schedule and safe node-compromise emulation for Open5GS.

The scenario generator is only imported by the test collector / auditors,
NEVER by the V10 detector. No exploit, shell, host privilege escalation or
compromise of native Open5GS binaries occurs. For S10 the already sandboxed
research NF configuration and optionally its local report delivery are
purposefully changed, while the independently keyed control remains trusted.
"""
from __future__ import annotations

import random

from experiments.robustness_5g.design import attack_schedule as base_attack_schedule
from experiments.robustness_5g.design import s5_members
from experiments.distributed_5g.v6_design import v6_members, v6_count
from experiments.distributed_5g.v7_design import v7_members, v7_count
from experiments.distributed_5g.v8_design import v8_members, v8_count
from experiments.distributed_5g.v9_design import v9_members, v9_count

V10_WINDOWS = 3600
V10_COMPROMISE_VARIANTS = (
    "node_config", "node_report_tamper", "node_sbi_abuse",
)
STRUCTURAL_SCENARIOS = ("S6", "S7", "S8", "S9")
ALL_SCENARIOS = frozenset((
    "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9",
    "C", "D", "F", "G", "R", "S10",
))
STRUCTURAL_MEMBERS = {
    "S6": v6_members, "S7": v7_members,
    "S8": v8_members, "S9": v9_members,
}
STRUCTURAL_COUNTS = {
    "S6": v6_count, "S7": v7_count,
    "S8": v8_count, "S9": v9_count,
}


def v10_members(entry):
    scenario = str(entry[2])
    if scenario == "S5":
        return tuple(s5_members(entry))
    if scenario in STRUCTURAL_MEMBERS:
        return tuple(STRUCTURAL_MEMBERS[scenario](entry))
    if scenario == "S10":
        if int(entry[3]) != 4 or entry[5] not in V10_COMPROMISE_VARIANTS:
            raise ValueError("S10 may only emulate the isolated research NF slot-4")
        return (4,)
    return (int(entry[3]),)


def v10_count(entry, window, index):
    """Only S6--S9 use the archived low-amplitude structural workload."""
    if entry[2] not in STRUCTURAL_COUNTS:
        raise ValueError("Not a structural V10 attack")
    return STRUCTURAL_COUNTS[entry[2]](entry, window, index)


def s10_control_mode(variant):
    """Choose the existing controlled research-NF fault boundary.

    'C': change the real lab NF's config file; the observer reports truth.
    'F': change the config AND corrupt an already authenticated local
         observer report in transit (its signature no longer verifies).
    This is **not** takeover of the independent signing process/key.
    """
    if variant == "node_report_tamper":
        return "F"
    if variant in ("node_config", "node_sbi_abuse"):
        return "C"
    raise ValueError(f"Unknown S10 compromise variant: {variant!r}")


def _find_slot(occupied, duration, *, earliest=70, latest=3520):
    for start in range(earliest, latest + 1, 10):
        # Ten normal windows of washout before/after every additional episode
        # (including original attack episodes).
        if all(window not in occupied for window in range(start - 10, start + duration + 10)):
            occupied.update(range(start, start + duration))
            return start
    raise ValueError("Not enough nonoverlapping 3600-window schedule capacity")


def v10_attack_schedule(windows: int, seed: int):
    """Original 40 legacy episodes PLUS 16 S6--S9 and 3 explicit S10 cases.

    Retain the precise old base generator (S1--S5,C,D,F,G,R), including
    seed-dependent timing and the 4 S5 episodes.  Extra scenarios are
    reserved in non-overlapping 30-window gaps, so no attack hides
    another attack's observation/label.
    """
    if int(windows) != V10_WINDOWS:
        raise ValueError("V10 full multi-threat scenario requires 3600 windows")
    existing = list(base_attack_schedule(int(windows), int(seed)))
    occupied = {w for e in existing for w in range(int(e[0]), int(e[1]))}
    rng = random.Random(int(seed) + 10_000_010)
    extras = []
    for scenario in STRUCTURAL_SCENARIOS:
        for k in (2, 3, 4, 5):
            zone = rng.randrange(4)
            members = sorted(rng.sample(list(range(zone * 8, (zone + 1) * 8)), k))
            variant = f"k{k}_members_" + "-".join(str(m) for m in members)
            extras.append((scenario, members[0], variant))
    for variant in V10_COMPROMISE_VARIANTS:
        extras.append(("S10", 4, variant))
    rng.shuffle(extras)
    for scenario, representative, variant in extras:
        start = _find_slot(occupied, 30)
        extras_entry = (start, start + 30, scenario, representative, 0, variant)
        # Fail closed against an incorrect membership encoding / zone.
        members = v10_members(extras_entry)
        if not members or (scenario in STRUCTURAL_SCENARIOS and len(set(m // 8 for m in members)) != 1):
            raise ValueError(f"Invalid group: {extras_entry}")
        existing.append(extras_entry)
    result = sorted(existing, key=lambda e: e[0])
    if set(e[2] for e in result) != ALL_SCENARIOS:
        raise ValueError("Incomplete V10 attack family coverage")
    for a, b in zip(result, result[1:]):
        if a[1] > b[0]:
            raise ValueError(f"Overlapping V10 episodes: {a} / {b}")
    return result
