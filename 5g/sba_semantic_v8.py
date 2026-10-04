"""5G SBA semantic consistency v8: noisy operational context without oracle leakage.

V8 freezes the selected detection core from v7/v6:

* semantic quorum q3of5;
* global persistence OFF;
* transport byz_3of4.

The change is exclusively in the operational-context model. V7 demonstrated
that narrowly scoped grace can suppress legitimate transients, but its marker
and recovery evidence were generated directly from disturbance ground truth.
V8 replaces that oracle-like coupling with a latent operational-state process.

Generative contract
-------------------
1. An operational state is selected independently of attack labels.
2. The state may manifest a transient semantic inconsistency.
3. A marker is observed through a separate noisy channel with recall < 1,
   latency, optional wrong correlation and false-positive markers.
4. Recovery evidence is another separate channel; it can be missing or arrive
   after grace expiry.
5. The detector sees only observations and marker messages. It never reads the
   latent state, disturbance identity or attack labels.

All probabilities and timings below are synthetic proof-of-concept parameters;
they are deliberately exposed for sensitivity analysis and must not be
interpreted as measured production-5GC rates.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from enum import Enum
from typing import Dict, FrozenSet, Iterable, List, Mapping, Optional, Tuple

import numpy as np

from sba_semantic_v6 import (
    AttackFamily,
    CompromiseMode,
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
class OperationalStateSpec:
    state: OperationalState
    disturbance: LegitimateDisturbance
    fact: SemanticFact
    grace_s: float
    recovery_median_s: float


STATE_SPECS: Mapping[OperationalState, OperationalStateSpec] = {
    OperationalState.TOKEN_REFRESH: OperationalStateSpec(
        OperationalState.TOKEN_REFRESH,
        LegitimateDisturbance.TOKEN_REFRESH_RACE,
        SemanticFact.TOKEN,
        2.0,
        0.55,
    ),
    OperationalState.POLICY_SYNC: OperationalStateSpec(
        OperationalState.POLICY_SYNC,
        LegitimateDisturbance.POLICY_PROPAGATION_DELAY,
        SemanticFact.SLICE,
        8.0,
        2.5,
    ),
    OperationalState.SUBSCRIPTION_SYNC: OperationalStateSpec(
        OperationalState.SUBSCRIPTION_SYNC,
        LegitimateDisturbance.SUBSCRIPTION_DESYNC,
        SemanticFact.NOTIFICATION,
        4.0,
        1.2,
    ),
    OperationalState.NF_RECOVERY: OperationalStateSpec(
        OperationalState.NF_RECOVERY,
        LegitimateDisturbance.NF_RESTART_RECOVERY,
        SemanticFact.PROCEDURE,
        15.0,
        4.5,
    ),
    OperationalState.ROUTE_TRANSITION: OperationalStateSpec(
        OperationalState.ROUTE_TRANSITION,
        LegitimateDisturbance.SCP_BENIGN_REROUTE,
        SemanticFact.ROUTE,
        3.0,
        0.9,
    ),
}

DISTURBANCE_TO_STATE: Mapping[LegitimateDisturbance, OperationalState] = {
    spec.disturbance: state for state, spec in STATE_SPECS.items()
}

ATTACK_STATE: Mapping[AttackFamily, OperationalState] = {
    AttackFamily.SCP_BYPASS: OperationalState.ROUTE_TRANSITION,
    AttackFamily.NO_TOKEN: OperationalState.TOKEN_REFRESH,
    AttackFamily.NOTIFY_ABUSE: OperationalState.SUBSCRIPTION_SYNC,
    AttackFamily.COARSE_SCOPE: OperationalState.POLICY_SYNC,
    AttackFamily.PROCEDURE_SKIP: OperationalState.NF_RECOVERY,
    AttackFamily.CROSS_SLICE_CHAIN: OperationalState.POLICY_SYNC,
}


@dataclass(frozen=True)
class OperationalContextConfig:
    marker_recall: float = 0.85
    false_marker_probability: float = 0.02
    correlation_error_probability: float = 0.03
    marker_latency_median_s: float = 0.12
    marker_latency_sigma: float = 0.45
    recovery_observation_probability: float = 0.90
    recovery_latency_sigma: float = 0.45
    recovery_latency_multiplier: float = 1.0
    disturbance_manifest_probability: float = 0.95

    def validate(self) -> None:
        for name in (
            "marker_recall",
            "false_marker_probability",
            "correlation_error_probability",
            "recovery_observation_probability",
            "disturbance_manifest_probability",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0,1], got {value}")
        for name in (
            "marker_latency_median_s",
            "marker_latency_sigma",
            "recovery_latency_sigma",
            "recovery_latency_multiplier",
        ):
            if float(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class OperationalMarker:
    state: OperationalState
    fact: SemanticFact
    correlation_id: str
    valid_from: float
    expires_at: float
    observed_at: float
    source: str
    is_false_marker: bool = False
    correlation_correct: bool = True

    def covers_event(self, correlation_id: str, fact: SemanticFact, event_time: float) -> bool:
        return (
            self.fact == fact
            and self.correlation_id == correlation_id
            and self.valid_from <= event_time <= self.expires_at
        )


@dataclass
class V8Window(V6Window):
    latent_operational_state: OperationalState = OperationalState.NORMAL
    operational_markers: Tuple[OperationalMarker, ...] = ()
    operational_event_ids: FrozenSet[str] = frozenset()
    recovery_event_ids: FrozenSet[str] = frozenset()
    marker_expected: bool = False
    marker_emitted: bool = False
    marker_correct: bool = False
    recovery_emitted: bool = False
    transient_manifested: bool = False


@dataclass
class V8DetectionResult:
    # DetectionResult-compatible fields are repeated explicitly so v8 stays
    # decoupled from dataclass inheritance details in prior iterations.
    alert: bool
    semantic_alert: bool
    transport_alert: bool
    fired_facts: FrozenSet[SemanticFact]
    fired_event_ids: Tuple[str, ...]
    transport_stat: float
    transport_threshold: float
    score: float
    first_alert_time: float
    event_alert_times: Dict[str, float]
    fact_alert_times: Dict[Tuple[str, SemanticFact], float]
    unknown_fraction: float
    evidence_count: int
    grace_deferred_keys: int = 0
    grace_resolved_keys: int = 0
    grace_expired_alerts: int = 0
    markers_seen: int = 0


class OperationalContextProcess:
    """Independent marker/recovery observation channel for a latent state."""

    def __init__(self, seed: int, cfg: Optional[OperationalContextConfig] = None):
        self.cfg = cfg or OperationalContextConfig()
        self.cfg.validate()
        self.rng = np.random.default_rng(int(seed) + 8_000_021)

    def _lognormal(self, median: float, sigma: float) -> float:
        return float(median * math.exp(self.rng.normal(0.0, sigma)))

    def choose_event(self, window: V6Window):
        candidates = [e for e in window.events if e.event_id not in window.attack_event_ids]
        if not candidates:
            return None
        return candidates[int(self.rng.integers(0, len(candidates)))]

    def emit_true_marker(self, *, event, spec: OperationalStateSpec) -> Optional[OperationalMarker]:
        if self.rng.random() >= self.cfg.marker_recall:
            return None
        correct = self.rng.random() >= self.cfg.correlation_error_probability
        correlation_id = (
            event.correlation_id
            if correct
            else f"wrong-{self.rng.integers(1, 10**9)}"
        )
        delay = self._lognormal(
            self.cfg.marker_latency_median_s,
            self.cfg.marker_latency_sigma,
        )
        return OperationalMarker(
            state=spec.state,
            fact=spec.fact,
            correlation_id=str(correlation_id),
            valid_from=float(event.timestamp - 0.10),
            expires_at=float(event.timestamp + spec.grace_s),
            observed_at=float(event.timestamp + delay),
            source="operational-context-channel",
            is_false_marker=False,
            correlation_correct=bool(correct),
        )

    def emit_false_marker(self, window: V6Window) -> Optional[OperationalMarker]:
        if self.rng.random() >= self.cfg.false_marker_probability or not window.events:
            return None
        states = list(STATE_SPECS)
        state = states[int(self.rng.integers(0, len(states)))]
        spec = STATE_SPECS[state]
        event = window.events[int(self.rng.integers(0, len(window.events)))]
        delay = self._lognormal(
            self.cfg.marker_latency_median_s,
            self.cfg.marker_latency_sigma,
        )
        return OperationalMarker(
            state=state,
            fact=spec.fact,
            correlation_id=str(event.correlation_id),
            valid_from=float(event.timestamp - 0.10),
            expires_at=float(event.timestamp + spec.grace_s),
            observed_at=float(event.timestamp + delay),
            source="false-operational-context",
            is_false_marker=True,
            correlation_correct=True,
        )

    def recovery_delay(self, spec: OperationalStateSpec) -> float:
        return self._lognormal(
            spec.recovery_median_s * self.cfg.recovery_latency_multiplier,
            self.cfg.recovery_latency_sigma,
        )


class SemanticSbaSimulator(V6SemanticSbaSimulator):
    """V6 world plus independently observed operational context."""

    def __init__(
        self,
        *args,
        operational_cfg: Optional[OperationalContextConfig] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.operational_cfg = operational_cfg or OperationalContextConfig()
        self.operational = OperationalContextProcess(self.seed, self.operational_cfg)

    def _manifest_transient(
        self,
        base: V6Window,
        event,
        spec: OperationalStateSpec,
    ) -> Tuple[List[OriginObservation], FrozenSet[str]]:
        if self.operational.rng.random() >= self.operational_cfg.disturbance_manifest_probability:
            return list(base.observations), frozenset()
        disturbance_spec = DISTURBANCE_SPECS[spec.disturbance]
        primary = set(disturbance_spec.primary_origins)
        escalated = self.operational.rng.random() < disturbance_spec.escalation_probability
        out: List[OriginObservation] = []
        for obs in base.observations:
            if obs.event_id != event.event_id or obs.fact != spec.fact:
                out.append(obs)
                continue
            force = obs.origin in primary or (
                escalated and obs.origin == disturbance_spec.escalation_origin
            )
            out.append(
                replace(
                    obs,
                    state=EvidenceState.INCONSISTENT,
                    confidence=max(float(obs.confidence), 0.85),
                )
                if force
                else obs
            )
        return out, frozenset([event.event_id])

    def _add_recovery(
        self,
        observations: List[OriginObservation],
        event,
        spec: OperationalStateSpec,
    ) -> Tuple[List[OriginObservation], bool]:
        affected = [
            o for o in observations
            if o.event_id == event.event_id
            and o.fact == spec.fact
            and o.state == EvidenceState.INCONSISTENT
        ]
        if (
            not affected
            or self.operational.rng.random()
            >= self.operational_cfg.recovery_observation_probability
        ):
            return observations, False
        recovery_at = float(event.timestamp + self.operational.recovery_delay(spec))
        out = list(observations)
        for i, obs in enumerate(affected):
            out.append(
                replace(
                    obs,
                    state=EvidenceState.CONSISTENT,
                    observed_at=float(
                        max(recovery_at + i * 0.001, obs.observed_at + 0.001)
                    ),
                    confidence=max(float(obs.confidence), 0.90),
                )
            )
        return out, True

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
        operational_state: OperationalState | str = OperationalState.NORMAL,
        false_marker: bool = True,
    ) -> V8Window:
        state = OperationalState(operational_state)
        # V6 disturbance generation is deliberately disabled: state, marker,
        # transient evidence and recovery are separate stochastic channels.
        base = super().generate_window(
            window_id,
            context=context,
            attack_family=attack_family,
            hidden_calls=hidden_calls,
            sophistication=sophistication,
            compromised_origins=compromised_origins,
            compromise_mode=compromise_mode,
            adaptive_budget=adaptive_budget,
            disturbance=LegitimateDisturbance.NONE,
            disturbance_events=0,
        )

        observations = list(base.observations)
        markers: List[OperationalMarker] = []
        operational_ids: FrozenSet[str] = frozenset()
        recovery_ids: FrozenSet[str] = frozenset()
        marker_expected = state != OperationalState.NORMAL
        marker_emitted = False
        marker_correct = False
        recovery_emitted = False

        if state != OperationalState.NORMAL:
            spec = STATE_SPECS[state]
            event = self.operational.choose_event(base)
            if event is not None:
                observations, operational_ids = self._manifest_transient(base, event, spec)
                observations, recovery_emitted = self._add_recovery(
                    observations, event, spec
                )
                if recovery_emitted:
                    recovery_ids = frozenset([event.event_id])
                marker = self.operational.emit_true_marker(event=event, spec=spec)
                if marker is not None:
                    markers.append(marker)
                    marker_emitted = True
                    marker_correct = marker.correlation_correct

        if state == OperationalState.NORMAL and false_marker:
            marker = self.operational.emit_false_marker(base)
            if marker is not None:
                markers.append(marker)
                marker_emitted = True

        data = dict(base.__dict__)
        data["observations"] = observations
        return V8Window(
            **data,
            latent_operational_state=state,
            operational_markers=tuple(markers),
            operational_event_ids=operational_ids,
            recovery_event_ids=recovery_ids,
            marker_expected=marker_expected,
            marker_emitted=marker_emitted,
            marker_correct=marker_correct,
            recovery_emitted=recovery_emitted,
            transient_manifested=bool(operational_ids),
        )

    def generate_attack_during_state(
        self,
        window_id: int,
        *,
        attack_family: AttackFamily | str,
        hidden_calls: int = 1,
        sophistication: Sophistication | str = Sophistication.ADAPTIVE,
    ) -> V8Window:
        """Condition an attack to occur while a genuine state is active.

        The marker still traverses the same noisy operational channel. No
        synthetic recovery is created for the malicious inconsistency.
        """
        family = AttackFamily(attack_family)
        state = ATTACK_STATE.get(family, OperationalState.NORMAL)
        base = super().generate_window(
            window_id,
            attack_family=family,
            hidden_calls=hidden_calls,
            sophistication=sophistication,
            disturbance=LegitimateDisturbance.NONE,
            disturbance_events=0,
        )
        observations = list(base.observations)
        markers: List[OperationalMarker] = []
        marker_emitted = False
        marker_correct = False
        operational_ids: FrozenSet[str] = frozenset()

        if state != OperationalState.NORMAL and base.attack_event_ids:
            spec = STATE_SPECS[state]
            event_id = sorted(base.attack_event_ids)[0]
            by_id = {e.event_id: e for e in base.events}
            event = by_id.get(event_id)
            if event is not None:
                operational_ids = frozenset([event.event_id])
                marker = self.operational.emit_true_marker(event=event, spec=spec)
                if marker is not None:
                    markers.append(marker)
                    marker_emitted = True
                    marker_correct = marker.correlation_correct

        data = dict(base.__dict__)
        data["observations"] = observations
        return V8Window(
            **data,
            latent_operational_state=state,
            operational_markers=tuple(markers),
            operational_event_ids=operational_ids,
            recovery_event_ids=frozenset(),
            marker_expected=state != OperationalState.NORMAL,
            marker_emitted=marker_emitted,
            marker_correct=marker_correct,
            recovery_emitted=False,
            transient_manifested=False,
        )


class SemanticQuorumDetector(V6SemanticQuorumDetector):
    """Frozen q3of5/byz_3of4 core plus noisy marker-aware scoped grace."""

    def __init__(
        self,
        target_fpr: float = 0.01,
        *,
        enable_operational_grace: bool = True,
    ):
        super().__init__(target_fpr, transport_mode=TransportMode.BYZ_3OF4)
        self.enable_operational_grace = bool(enable_operational_grace)

    @staticmethod
    def _event_by_id(window: V8Window):
        return {e.event_id: e for e in window.events}

    def _semantic_stateful(self, window: V8Window, *, until: Optional[float] = None):
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
            o for o in window.observations
            if o.origin in selected[o.fact]
            and (until is None or o.observed_at <= until)
        ]
        markers = (
            [m for m in window.operational_markers if until is None or m.observed_at <= until]
            if self.enable_operational_grace
            else []
        )

        # A marker must arrive before quorum confirmation to defer it. Late
        # markers cannot retroactively erase a previously emitted alarm.
        timeline: List[Tuple[float, int, str, object]] = []
        for marker in markers:
            timeline.append((float(marker.observed_at), 0, "marker", marker))
            timeline.append((float(marker.expires_at), 2, "expiry", marker))
        for obs in observations:
            timeline.append((float(obs.observed_at), 1, "obs", obs))
        timeline.sort(key=lambda x: (x[0], x[1], x[2]))

        latest: Dict[Tuple[str, SemanticFact], Dict[str, EvidenceState]] = {}
        active_markers: List[OperationalMarker] = []
        fact_alert_times: Dict[Tuple[str, SemanticFact], float] = {}
        event_alert_times: Dict[str, float] = {}
        fired_facts: set[SemanticFact] = set()
        deferred: set[Tuple[str, SemanticFact]] = set()
        resolved: set[Tuple[str, SemanticFact]] = set()
        expired_alerts: set[Tuple[str, SemanticFact]] = set()
        by_event = self._event_by_id(window)

        def inconsistent_roots(key):
            return sum(
                state == EvidenceState.INCONSISTENT
                for state in latest.get(key, {}).values()
            )

        def marker_for(key, at):
            event_id, fact = key
            event = by_event.get(event_id)
            if event is None:
                return None
            matches = [
                m for m in active_markers
                if m.observed_at <= at <= m.expires_at
                and m.covers_event(event.correlation_id, fact, event.timestamp)
            ]
            return min(matches, key=lambda m: (m.expires_at, m.observed_at)) if matches else None

        def fire(key, at, expired=False):
            if key in fact_alert_times:
                return
            event_id, fact = key
            fact_alert_times[key] = float(at)
            event_alert_times[event_id] = min(
                float(at), event_alert_times.get(event_id, float("inf"))
            )
            fired_facts.add(fact)
            if expired:
                expired_alerts.add(key)

        for at, _priority, kind, payload in timeline:
            if kind == "marker":
                # A marker arriving after its own expiry is stale and ignored.
                if at <= payload.expires_at:
                    active_markers.append(payload)
                continue

            if kind == "obs":
                obs = payload
                key = (obs.event_id, obs.fact)
                latest.setdefault(key, {})[obs.trust_root] = obs.state
                if inconsistent_roots(key) < self.profile.q:
                    if key in deferred:
                        resolved.add(key)
                    continue
                marker = marker_for(key, at)
                if marker is not None:
                    deferred.add(key)
                    continue
                fire(key, at)
                continue

            marker = payload  # expiry
            if marker in active_markers:
                active_markers.remove(marker)
            for event_id, event in by_event.items():
                key = (event_id, marker.fact)
                if key in fact_alert_times:
                    continue
                if not marker.covers_event(
                    event.correlation_id, marker.fact, event.timestamp
                ):
                    continue
                if inconsistent_roots(key) >= self.profile.q:
                    fire(key, at, expired=True)
                elif key in deferred:
                    resolved.add(key)

        return (
            bool(event_alert_times),
            fired_facts,
            event_alert_times,
            fact_alert_times,
            deferred,
            resolved,
            expired_alerts,
            len(markers),
        )

    def score(self, window: V8Window, *, until: Optional[float] = None) -> V8DetectionResult:
        base = super().score(window, until=until)
        (
            semantic_alert,
            semantic_facts,
            event_alert_times,
            fact_alert_times,
            deferred,
            resolved,
            expired_alerts,
            markers_seen,
        ) = self._semantic_stateful(window, until=until)

        fired_facts = set(semantic_facts)
        if base.transport_alert:
            fired_facts.add(SemanticFact.TRANSPORT)

        times = list(event_alert_times.values())
        if base.transport_alert:
            transport_time = self._robust_alert_time(
                window, float(base.transport_threshold)
            )
            if np.isfinite(transport_time):
                times.append(float(transport_time))
        first_alert_time = min(times) if times else float("nan")
        ratio = (
            float(base.transport_stat) / max(abs(float(base.transport_threshold)), 1e-9)
            if np.isfinite(base.transport_stat) and np.isfinite(base.transport_threshold)
            else 0.0
        )
        score = max(float(len(fired_facts)), ratio)

        return V8DetectionResult(
            alert=bool(semantic_alert or base.transport_alert),
            semantic_alert=bool(semantic_alert),
            transport_alert=bool(base.transport_alert),
            fired_facts=frozenset(fired_facts),
            fired_event_ids=tuple(sorted(event_alert_times)),
            transport_stat=float(base.transport_stat),
            transport_threshold=float(base.transport_threshold),
            score=float(score),
            first_alert_time=float(first_alert_time),
            event_alert_times=event_alert_times,
            fact_alert_times=fact_alert_times,
            unknown_fraction=float(base.unknown_fraction),
            evidence_count=int(base.evidence_count),
            grace_deferred_keys=len(deferred),
            grace_resolved_keys=len(resolved),
            grace_expired_alerts=len(expired_alerts),
            markers_seen=int(markers_seen),
        )


__all__ = [
    "ATTACK_STATE",
    "AttackFamily",
    "CompromiseMode",
    "DISTURBANCE_TO_STATE",
    "EvidenceOrigin",
    "EvidenceState",
    "LegitimateDisturbance",
    "OperationalContextConfig",
    "OperationalContextProcess",
    "OperationalMarker",
    "OperationalState",
    "OperationalStateSpec",
    "STATE_SPECS",
    "SemanticFact",
    "SemanticQuorumDetector",
    "SemanticSbaSimulator",
    "Sophistication",
    "TransportMode",
    "V8DetectionResult",
    "V8Window",
    "empirical_auc",
]
