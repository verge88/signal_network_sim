"""V11: causal, zone-local S9 staircase signature detector.

Scenario-specific synthetic S9 signature, not a general 5G classifier.
The streaming API consumes current and past request counts only. Labels, the
episode schedule and future windows cannot enter the predictor.
"""
from __future__ import annotations
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass

WAVE = (0, 1, 1, 2, 1, 0)
PERIOD = 3
EPISODE_LENGTH = 30

@dataclass
class _Pending:
    window: int
    baselines: dict[str, int]

@dataclass
class _Active:
    inferred_onset: int
    baselines: dict[str, int]

class CausalStaircaseDetector:
    """Per-run streaming detector, permutation-invariant within measured zones.

    zones maps NF names to known topology zones. Consecutive complete windows
    are required; gaps reset state. No labels or attack schedule are accepted.
    """
    def __init__(self, zones: Mapping[str, str | int]):
        self.zones = dict(zones)
        if len(self.zones) < 2:
            raise ValueError("At least two NF identifiers required")
        members = {}
        for nf, zone in self.zones.items():
            members.setdefault(zone, []).append(nf)
        self.members = {z: tuple(sorted(nfs)) for z, nfs in members.items()}
        self.reset()

    def reset(self) -> None:
        self._previous: deque[dict[str, int]] = deque(maxlen=2)
        self._last_window: int | None = None
        self._pending: dict[str | int, _Pending] = {}
        self._active: dict[str | int, _Active] = {}

    def step(self, window_id: int, request_counts: Mapping[str, int]) -> dict[str, int]:
        """Produce immutable decisions for this window without future data."""
        window_id = int(window_id)
        if set(request_counts) != set(self.zones):
            raise ValueError("Incomplete or unexpected NF coverage")
        counts = {}
        for nf, value in request_counts.items():
            if isinstance(value, bool) or int(value) != value or value < 0:
                raise ValueError(f"Non-integral request count for {nf}")
            counts[nf] = int(value)
        if self._last_window is not None and window_id != self._last_window + 1:
            self.reset()
        alarms: set[str] = set()
        for zone, zone_nfs in self.members.items():
            active = self._active.pop(zone, None)
            if active is not None:
                phase = window_id - active.inferred_onset
                if 0 <= phase < EPISODE_LENGTH:
                    offset = WAVE[(phase // PERIOD) % len(WAVE)]
                    good = {nf: base for nf, base in active.baselines.items()
                            if counts[nf] == base + offset}
                    if len(good) >= 2:
                        self._active[zone] = _Active(active.inferred_onset, good)
                        alarms.update(good)
                        continue
            pending = self._pending.pop(zone, None)
            if pending is not None and window_id == pending.window + 1:
                good = {nf: base for nf, base in pending.baselines.items()
                        if counts[nf] == base + 1}
                if len(good) >= 2:
                    self._active[zone] = _Active(pending.window - 2, good)
                    alarms.update(good)
                    continue
            if len(self._previous) != 2:
                continue
            prev2, prev1 = self._previous
            candidates = [nf for nf in zone_nfs
                          if 2 <= counts[nf] <= 4 and
                          counts[nf] == prev1[nf] == prev2[nf]]
            if len(candidates) >= 2 and len({counts[nf] for nf in candidates}) >= 2:
                self._pending[zone] = _Pending(
                    window_id, {nf: counts[nf] for nf in candidates})
                alarms.update(candidates)
        self._previous.append(counts)
        self._last_window = window_id
        return {nf: int(nf in alarms) for nf in self.zones}
