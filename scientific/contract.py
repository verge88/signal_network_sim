from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


def elapsed_since(start: Optional[int], now: int) -> int:
    """Return elapsed discrete intervals without treating start=0 as false."""
    if now < 0:
        raise ValueError("now must be >= 0")
    if start is None:
        return 0
    if start < 0:
        raise ValueError("start must be >= 0 or None")
    if now < start:
        raise ValueError(f"now={now} precedes start={start}")
    return now - start


class Capability(str, Enum):
    """Capabilities acquired after node compromise."""
    FORGE_LOCAL_REPORT = "forge_local_report"
    MODIFY_LOCAL_STATE = "modify_local_state"
    ORIGINATE_SIGNALLING = "originate_signalling"
    DROP_OR_REDIRECT = "drop_or_redirect"


@dataclass(frozen=True)
class CompromiseState:
    compromised_since: Optional[int] = None
    capabilities: frozenset[Capability] = frozenset()

    @property
    def compromised(self) -> bool:
        return self.compromised_since is not None

    def elapsed(self, now: int) -> int:
        return elapsed_since(self.compromised_since, now)

    def can(self, capability: Capability) -> bool:
        return capability in self.capabilities
