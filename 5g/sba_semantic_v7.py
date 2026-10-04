"""5G SBA semantic consistency v7: operational-state-aware semantic grace.

V7 freezes the v6 candidate configuration:

* semantic quorum: q3of5;
* global soft persistence: OFF;
* transport: Byzantine-resistant byz_3of4.

The new question is whether a detector can distinguish malicious semantic
inconsistency from short-lived, legitimate cross-NF disagreement.  V7 adds
explicit *operational markers* (refresh/synchronisation/recovery/reroute state)
and later recovery observations.  The detector may defer a matching semantic
quorum only inside a fact- and correlation-scoped grace interval.  If evidence
converges before the interval expires, the transient is suppressed; otherwise
an alert is emitted at grace expiry.  Thus grace is not a global persistence
requirement and does not apply to unrelated events/facts.

Operational markers are observable control-plane context, not attack labels.
The numerical grace/recovery times are synthetic proof-of-concept assumptions
and must be calibrated against implementation traces before production claims.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from enum import Enum
from typing import Dict, FrozenSet, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from sba_semantic_v6 import (
    AttackFamily,
    CompromiseMode,
    DetectionResult,
    DISTURBANCE_SPECS,
    EvidenceOrigin,
    EvidenceState,
    LegitimateDisturbance,
    OriginObservation,
    SemanticFact,
    SemanticQuorumDetector as V6SemanticQuorumDetector,
    SemanticSbaSimulator as V6SemanticSbaSimulator,
    Sophistication,
    TransportMode,
    V6Window,
    empirical_auc,
)


class OperationalState(str, Enum):
    NORMAL = "normal"
    TOKEN_REFRESH = "token_refresh"
    POLICY_SYNC = "policy_sync"
    SUBSCRIPTION_SYNC = "subscription_sync"
    NF_RECOVERY = "nf_recovery"
    ROUTE_TRANSITION = "route_transition"


@dataclass(frozen=True)
class OperationalGraceSpec:
    state: OperationalState
    fact: SemanticFact
    grace_s: float
    recovery_median_s: float


OPERATIONAL_GRACE: Mapping[LegitimateDisturbance, OperationalGraceSpec] = {
    LegitimateDisturbance.TOKEN_REFRESH_RACE: OperationalGraceSpec(
        OperationalState.TOKEN_REFRESH,
        SemanticFact.TOKEN,
        grace_s=2.0,
        recovery_median_s=0.55,
    ),
    LegitimateDisturbance.POLICY_PROPAGATION_DELAY: OperationalGraceSpec(
        OperationalState.POLICY_SYNC,
        SemanticFact.SLICE,
        grace_s=8.0,
        recovery_median_s=2.5,
    ),
    LegitimateDisturbance.SUBSCRIPTION_DESYNC: OperationalGraceSpec(
        OperationalState.SUBSCRIPTION_SYNC,
        SemanticFact.NOTIFICATION,
        grace_s=4.0,
        recovery_median_s=1.2,
    ),
    LegitimateDisturbance.NF_RESTART_RECOVERY: OperationalGraceSpec(
        OperationalState.NF_RECOVERY,
        SemanticFact.PROCEDURE,
        grace_s=15.0,
        recovery_median_s=4.5,
    ),
    LegitimateDisturbance.SCP_BENIGN_REROUTE: OperationalGraceSpec(
        OperationalState.ROUTE_TRANSITION,
        SemanticFact.ROUTE,
        grace_s=3.0,
        recovery_median_s=0.9,
    ),
}


# Used only by the grace-abuse stress experiment: an adaptive attack can occur
# during a legitimate-looking operational state.  No recovery observation is
# created for the attack, so a persistent inconsistency must still alert when
# the scoped grace interval expires.
ATTACK_OPERATIONAL_STATE: Mapping[
    AttackFamily, Tuple[OperationalState, SemanticFact, float]
] = {
    AttackFamily.SCP_BYPASS: (
        OperationalState.ROUTE_TRANSITION,
        SemanticFact.ROUTE,
        3.0,
    ),
    AttackFamily.NO_TOKEN: (
        OperationalState.TOKEN_REFRESH,
        SemanticFact.TOKEN,
        2.0,
    ),
    AttackFamily.NOTIFY_ABUSE: (
        OperationalState.SUBSCRIPTION_SYNC,
        SemanticFact.NOTIFICATION,
        4.0,
    ),
    AttackFamily.COARSE_SCOPE: (
        OperationalState.POLICY_SYNC,
        SemanticFact.SLICE,
        8.0,
    ),
    AttackFamily.PROCEDURE_SKIP: (
        OperationalState.NF_RECOVERY,
        SemanticFact.PROCEDURE,
        15.0,
    ),
    AttackFamily.CROSS_SLICE_CHAIN: (
        OperationalState.POLICY_SYNC,
        SemanticFact.SLICE,
        8.0,
    ),
}


@dataclass(frozen=True)
class OperationalMarker:
    state: OperationalState
    fact: SemanticFact
    correlation_id: str
    start_at: float
    expires_at: float
    source: str = "orchestrator"

    def applies(self, correlation_id: str, fact: SemanticFact, event_time: float) -> bool:
        return (
            self.fact == fact
            and self.correlation_id == correlation_id
            and self.start_at <= event_time <= self.expires_at
        )


@dataclass
class V7Window(V6Window):
    operational_markers: Tuple[OperationalMarker, ...] = ()


@dataclass
class V7DetectionResult(DetectionResult):
    grace_deferred_keys: int = 0
    grace_resolved_keys: int = 0
    grace_expired_alerts: int = 0


class SemanticSbaSimulator(V6SemanticSbaSimulator):
    """V6 worlds plus independent operational markers and recovery evidence."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Separate stream: enabling operational context must not perturb the v6
        # event/attack/transport world.
        self.operational_rng = np.random.default_rng(self.seed + 7_000_013)

    def _recovery_time(
        self,
        event_time: float,
        last_inconsistent_at: float,
        spec: OperationalGraceSpec,
    ) -> float:
        sigma = 0.30
        delay = spec.recovery_median_s * math.exp(
            self.operational_rng.normal(-0.5 * sigma * sigma, sigma)
        )
        # Recovery must be observable after the inconsistent reports but, for a
        # legitimate transient, before the declared grace interval expires.
        earliest = max(event_time + 0.02, last_inconsistent_at + 0.02)
        latest = event_time + 0.85 * spec.grace_s
        return float(min(max(event_time + delay, earliest), latest))

    def _operational_context(
        self,
        window: V6Window,
    ) -> Tuple[List[OriginObservation], Tuple[OperationalMarker, ...]]:
        if (
            window.disturbance == LegitimateDisturbance.NONE
            or not window.disturbance_event_ids
        ):
            return list(window.observations), ()

        spec = OPERATIONAL_GRACE[window.disturbance]
        by_event = {e.event_id: e for e in window.events}
        observations = list(window.observations)
        markers: List[OperationalMarker] = []

        for event_id in sorted(window.disturbance_event_ids):
            event = by_event.get(event_id)
            if event is None:
                continue
            marker = OperationalMarker(
                state=spec.state,
                fact=spec.fact,
                correlation_id=event.correlation_id,
                start_at=float(event.timestamp - 0.10),
                expires_at=float(event.timestamp + spec.grace_s),
            )
            markers.append(marker)

            affected = [
                o
                for o in observations
                if o.event_id == event_id
                and o.fact == spec.fact
                and o.state == EvidenceState.INCONSISTENT
            ]
            if not affected:
                continue
            last_inconsistent = max(o.observed_at for o in affected)
            recovery_base = self._recovery_time(
                event.timestamp,
                last_inconsistent,
                spec,
            )
            for idx, obs in enumerate(affected):
                # Small deterministic ordering jitter avoids tied timestamps.
                recovered_at = min(
                    marker.expires_at - 1e-4,
                    recovery_base + 0.001 * idx,
                )
                observations.append(
                    replace(
                        obs,
                        state=EvidenceState.CONSISTENT,
                        observed_at=float(recovered_at),
                        confidence=max(float(obs.confidence), 0.90),
                    )
                )

        return observations, tuple(markers)

    def generate_window(
        self,
        window_id: int,
        *,
        context: str = "normal_indirect",
        attack_family: AttackFamily | str = AttackFamily.NONE,
        hidden_calls: int = 0,
        sophistication: Sophistication | str = Sophistication.ADAPTIVE,
        compromised_origins: Iterable[EvidenceOrigin | str] = (),
        compromise_mode: CompromiseMode | str = CompromiseMode.PASSIVE,
        adaptive_budget: Optional[float] = None,
        disturbance: LegitimateDisturbance | str = LegitimateDisturbance.NONE,
        disturbance_events: int = 1,
    ) -> V7Window:
        base = super().generate_window(
            window_id,
            context=context,
            attack_family=attack_family,
            hidden_calls=hidden_calls,
            sophistication=sophistication,
            compromised_origins=compromised_origins,
            compromise_mode=compromise_mode,
            adaptive_budget=adaptive_budget,
            disturbance=disturbance,
            disturbance_events=disturbance_events,
        )
        observations, markers = self._operational_context(base)
        data = dict(base.__dict__)
        data["observations"] = observations
        return V7Window(
            **data,
            operational_markers=markers,
        )

    def cover_attack_with_operational_marker(
        self,
        window: V7Window,
        *,
        attack_event_id: Optional[str] = None,
    ) -> V7Window:
        """Return the same attack world with a matching grace marker, no recovery.

        This is an evaluator stress case: an attacker acts during a legitimate
        operational state.  Because no consistent recovery evidence is added,
        the v7 detector must defer rather than permanently suppress the attack.
        """
        family = AttackFamily(window.attack_family)
        if family not in ATTACK_OPERATIONAL_STATE or not window.attack_event_ids:
            return window
        event_id = attack_event_id or sorted(window.attack_event_ids)[0]
        by_event = {e.event_id: e for e in window.events}
        event = by_event.get(event_id)
        if event is None:
            return window
        state, fact, grace_s = ATTACK_OPERATIONAL_STATE[family]
        marker = OperationalMarker(
            state=state,
            fact=fact,
            correlation_id=event.correlation_id,
            start_at=float(event.timestamp - 0.10),
            expires_at=float(event.timestamp + grace_s),
            source="stress-operational-context",
        )
        return replace(
            window,
            operational_markers=tuple(window.operational_markers) + (marker,),
        )


class SemanticQuorumDetector(V6SemanticQuorumDetector):
    """Stateful q3of5 semantics with narrowly scoped operational grace."""

    def __init__(
        self,
        target_fpr: float = 0.01,
        *,
        enable_operational_grace: bool = True,
    ):
        super().__init__(
            target_fpr,
            transport_mode=TransportMode.BYZ_3OF4,
        )
        self.enable_operational_grace = bool(enable_operational_grace)

    @staticmethod
    def _event_by_id(window: V7Window):
        return {e.event_id: e for e in window.events}

    def _marker_for(
        self,
        window: V7Window,
        event_id: str,
        fact: SemanticFact,
    ) -> Optional[OperationalMarker]:
        if not self.enable_operational_grace:
            return None
        event = self._event_by_id(window).get(event_id)
        if event is None:
            return None
        matches = [
            marker
            for marker in window.operational_markers
            if marker.applies(event.correlation_id, fact, event.timestamp)
        ]
        if not matches:
            return None
        # A narrower/earlier marker is preferred if several overlap.
        return min(matches, key=lambda m: (m.expires_at, m.start_at, m.state.value))

    def _semantic_score_stateful(
        self,
        window: V7Window,
        *,
        until: Optional[float] = None,
    ):
        selected = {
            fact: self.selected_origins(fact)
            for fact in (
                SemanticFact.TOKEN,
                SemanticFact.NOTIFICATION,
                SemanticFact.SLICE,
                SemanticFact.DISCOVERY,
                SemanticFact.PROCEDURE,
                SemanticFact.ROUTE,
            )
        }
        observations = [
            o
            for o in window.observations
            if o.origin in selected[o.fact]
            and (until is None or o.observed_at <= until)
        ]

        # Timeline combines observation updates and grace-expiry boundaries.
        timeline: List[Tuple[float, int, str, object]] = []
        for obs in observations:
            timeline.append((float(obs.observed_at), 0, "obs", obs))
        if self.enable_operational_grace:
            for marker in window.operational_markers:
                if until is None or marker.expires_at <= until:
                    timeline.append((float(marker.expires_at), 1, "expiry", marker))
        timeline.sort(key=lambda x: (x[0], x[1], x[2]))

        latest: Dict[Tuple[str, SemanticFact], Dict[str, EvidenceState]] = {}
        fact_alert_times: Dict[Tuple[str, SemanticFact], float] = {}
        event_alert_times: Dict[str, float] = {}
        fired_facts: set[SemanticFact] = set()
        deferred: set[Tuple[str, SemanticFact]] = set()
        resolved: set[Tuple[str, SemanticFact]] = set()
        expired_alerts: set[Tuple[str, SemanticFact]] = set()

        by_event = self._event_by_id(window)

        def inconsistent_roots(key: Tuple[str, SemanticFact]) -> int:
            return sum(
                state == EvidenceState.INCONSISTENT
                for state in latest.get(key, {}).values()
            )

        def fire(key: Tuple[str, SemanticFact], at: float, *, expired: bool = False):
            if key in fact_alert_times:
                return
            event_id, fact = key
            fact_alert_times[key] = float(at)
            old = event_alert_times.get(event_id)
            if old is None or at < old:
                event_alert_times[event_id] = float(at)
            fired_facts.add(fact)
            if expired:
                expired_alerts.add(key)

        for at, _priority, kind, payload in timeline:
            if kind == "obs":
                obs = payload
                key = (obs.event_id, obs.fact)
                roots = latest.setdefault(key, {})
                # trust_root is the independence unit, not the raw number of
                # reports. A later report replaces the earlier state.
                roots[obs.trust_root] = obs.state
                if inconsistent_roots(key) < self.profile.q:
                    if key in deferred:
                        resolved.add(key)
                    continue

                marker = self._marker_for(window, obs.event_id, obs.fact)
                if marker is not None and at <= marker.expires_at:
                    deferred.add(key)
                    continue
                fire(key, at)

            else:  # grace expiry
                marker = payload
                for event_id, event in by_event.items():
                    key = (event_id, marker.fact)
                    if key in fact_alert_times:
                        continue
                    if not marker.applies(
                        event.correlation_id,
                        marker.fact,
                        event.timestamp,
                    ):
                        continue
                    if inconsistent_roots(key) >= self.profile.q:
                        fire(key, at, expired=True)
                    elif key in deferred:
                        resolved.add(key)

        semantic_alert = bool(event_alert_times)
        return (
            semantic_alert,
            fired_facts,
            event_alert_times,
            fact_alert_times,
            deferred,
            resolved,
            expired_alerts,
            observations,
        )

    def score(
        self,
        window: V7Window,
        *,
        until: Optional[float] = None,
    ) -> V7DetectionResult:
        # Reuse v6 only for the frozen byz_3of4 transport statistic/calibration.
        base = super().score(window, until=until)
        (
            semantic_alert,
            semantic_facts,
            event_alert_times,
            fact_alert_times,
            deferred,
            resolved,
            expired_alerts,
            observations,
        ) = self._semantic_score_stateful(window, until=until)

        fired_facts = set(semantic_facts)
        if base.transport_alert:
            fired_facts.add(SemanticFact.TRANSPORT)

        # V6 already computed the robust threshold/statistic. Reconstruct only
        # transport alert time because base.first_alert_time may include the v6
        # semantic decision that we intentionally replaced.
        transport_time = float("nan")
        if base.transport_alert:
            transport_time = self._robust_alert_time(
                window,
                float(base.transport_threshold),
            )
        times = list(event_alert_times.values())
        if base.transport_alert and np.isfinite(transport_time):
            if until is None or transport_time <= until:
                times.append(float(transport_time))
        first_alert_time = min(times) if times else float("nan")

        ratio = (
            base.transport_stat / max(abs(base.transport_threshold), 1e-9)
            if np.isfinite(base.transport_stat)
            and np.isfinite(base.transport_threshold)
            else 0.0
        )
        score = max(float(len(fired_facts)), float(ratio))
        unknown_count = sum(o.state == EvidenceState.UNKNOWN for o in observations)
        unknown_fraction = (
            unknown_count / len(observations)
            if observations
            else float("nan")
        )

        return V7DetectionResult(
            alert=semantic_alert or base.transport_alert,
            semantic_alert=semantic_alert,
            transport_alert=bool(base.transport_alert),
            fired_facts=frozenset(fired_facts),
            fired_event_ids=tuple(sorted(event_alert_times)),
            transport_stat=float(base.transport_stat),
            transport_threshold=float(base.transport_threshold),
            score=float(score),
            first_alert_time=float(first_alert_time),
            event_alert_times=event_alert_times,
            fact_alert_times=fact_alert_times,
            unknown_fraction=float(unknown_fraction),
            evidence_count=len(observations),
            grace_deferred_keys=len(deferred),
            grace_resolved_keys=len(resolved),
            grace_expired_alerts=len(expired_alerts),
        )


__all__ = [
    "ATTACK_OPERATIONAL_STATE",
    "AttackFamily",
    "CompromiseMode",
    "EvidenceOrigin",
    "EvidenceState",
    "LegitimateDisturbance",
    "OPERATIONAL_GRACE",
    "OperationalGraceSpec",
    "OperationalMarker",
    "OperationalState",
    "SemanticFact",
    "SemanticQuorumDetector",
    "SemanticSbaSimulator",
    "Sophistication",
    "TransportMode",
    "V7DetectionResult",
    "V7Window",
    "empirical_auc",
]
