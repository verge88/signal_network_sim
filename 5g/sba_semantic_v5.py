"""5G SBA semantic-consistency model v5: Byzantine hardening.

V5 keeps the v4 event generator and transport channel, but replaces the
semantic voting layer with an explicit trust-origin model suitable for
Byzantine ablations.

Scientific contract
-------------------
* one trust origin can contribute at most one vote to an event/fact quorum;
* repeated reports from the same administrative origin do not add evidence;
* HARD facts (token/slice/discovery) use quorum without temporal persistence;
* SOFT facts (notification/procedure/route) require quorum plus persistence
  across distinct events, unless persistence is explicitly disabled for an
  ablation;
* quorum profiles 2-of-3, 3-of-5 and 4-of-7 use increasingly broad independent
  evidence origins rather than replicated copies of one weak sensor;
* compromised origins may SUPPRESS or INJECT their own reports, but detector
  code never reads compromise ground truth or attack labels;
* POLICY and AUDIT are synthetic independent control/audit evidence origins
  used only for the quorum-size ablation. Their numerical visibility and
  latency parameters are proof-of-concept assumptions and must be calibrated
  before absolute real-network claims are made.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, FrozenSet, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from sba_semantic_v4 import (
    AttackFamily,
    CommunicationMode,
    CompromiseMode,
    CONTEXT_FALSE_EVIDENCE,
    CONTEXT_UNKNOWN,
    CONTEXT_VISIBILITY,
    DetectionResult,
    EvidenceState,
    SemanticFact,
    SemanticSbaSimulator as V4Simulator,
    SemanticWindow as V4Window,
    Sophistication,
    SOPHISTICATION_FACTOR,
    TrustDomain,
    TRANSPORT_OBSERVERS,
    empirical_auc,
)


class EvidenceOrigin(str, Enum):
    CONSUMER = "consumer"
    SCP = "scp"
    NRF = "nrf"
    PRODUCER = "producer"
    NWDAF = "nwdaf"
    POLICY = "policy"
    AUDIT = "audit"


class EvidenceClass(str, Enum):
    HARD = "hard"
    SOFT = "soft"


FACT_CLASS: Mapping[SemanticFact, EvidenceClass] = {
    SemanticFact.TOKEN: EvidenceClass.HARD,
    SemanticFact.SLICE: EvidenceClass.HARD,
    SemanticFact.DISCOVERY: EvidenceClass.HARD,
    SemanticFact.NOTIFICATION: EvidenceClass.SOFT,
    SemanticFact.PROCEDURE: EvidenceClass.SOFT,
    SemanticFact.ROUTE: EvidenceClass.SOFT,
}


@dataclass(frozen=True)
class QuorumProfile:
    name: str
    q: int
    n: int


QUORUM_PROFILES: Mapping[str, QuorumProfile] = {
    "q2of3": QuorumProfile("q2of3", 2, 3),
    "q3of5": QuorumProfile("q3of5", 3, 5),
    "q4of7": QuorumProfile("q4of7", 4, 7),
}


# Origins are ordered by semantic relevance for each fact. A profile takes the
# first n origins, so increasing n means adding genuinely different semantic
# evidence rather than cloning one sensor.
ORIGIN_ORDER: Mapping[SemanticFact, Tuple[EvidenceOrigin, ...]] = {
    SemanticFact.TOKEN: (
        EvidenceOrigin.NRF,
        EvidenceOrigin.POLICY,
        EvidenceOrigin.CONSUMER,
        EvidenceOrigin.PRODUCER,
        EvidenceOrigin.AUDIT,
        EvidenceOrigin.NWDAF,
        EvidenceOrigin.SCP,
    ),
    SemanticFact.NOTIFICATION: (
        EvidenceOrigin.PRODUCER,
        EvidenceOrigin.AUDIT,
        EvidenceOrigin.CONSUMER,
        EvidenceOrigin.POLICY,
        EvidenceOrigin.NWDAF,
        EvidenceOrigin.SCP,
        EvidenceOrigin.NRF,
    ),
    SemanticFact.SLICE: (
        EvidenceOrigin.POLICY,
        EvidenceOrigin.NRF,
        EvidenceOrigin.SCP,
        EvidenceOrigin.CONSUMER,
        EvidenceOrigin.PRODUCER,
        EvidenceOrigin.AUDIT,
        EvidenceOrigin.NWDAF,
    ),
    SemanticFact.DISCOVERY: (
        EvidenceOrigin.NRF,
        EvidenceOrigin.POLICY,
        EvidenceOrigin.SCP,
        EvidenceOrigin.CONSUMER,
        EvidenceOrigin.PRODUCER,
        EvidenceOrigin.AUDIT,
        EvidenceOrigin.NWDAF,
    ),
    SemanticFact.PROCEDURE: (
        EvidenceOrigin.AUDIT,
        EvidenceOrigin.PRODUCER,
        EvidenceOrigin.POLICY,
        EvidenceOrigin.CONSUMER,
        EvidenceOrigin.NWDAF,
        EvidenceOrigin.SCP,
        EvidenceOrigin.NRF,
    ),
    SemanticFact.ROUTE: (
        EvidenceOrigin.SCP,
        EvidenceOrigin.AUDIT,
        EvidenceOrigin.PRODUCER,
        EvidenceOrigin.CONSUMER,
        EvidenceOrigin.POLICY,
        EvidenceOrigin.NRF,
        EvidenceOrigin.NWDAF,
    ),
}


# Base probability that an origin can produce usable evidence for a fact.
# These values are synthetic and intentionally heterogeneous.
ORIGIN_VISIBILITY: Mapping[SemanticFact, Mapping[EvidenceOrigin, float]] = {
    SemanticFact.TOKEN: {
        EvidenceOrigin.CONSUMER: 0.72,
        EvidenceOrigin.SCP: 0.42,
        EvidenceOrigin.NRF: 0.88,
        EvidenceOrigin.PRODUCER: 0.68,
        EvidenceOrigin.NWDAF: 0.42,
        EvidenceOrigin.POLICY: 0.76,
        EvidenceOrigin.AUDIT: 0.50,
    },
    SemanticFact.NOTIFICATION: {
        EvidenceOrigin.CONSUMER: 0.60,
        EvidenceOrigin.SCP: 0.35,
        EvidenceOrigin.NRF: 0.30,
        EvidenceOrigin.PRODUCER: 0.76,
        EvidenceOrigin.NWDAF: 0.44,
        EvidenceOrigin.POLICY: 0.55,
        EvidenceOrigin.AUDIT: 0.64,
    },
    SemanticFact.SLICE: {
        EvidenceOrigin.CONSUMER: 0.60,
        EvidenceOrigin.SCP: 0.68,
        EvidenceOrigin.NRF: 0.82,
        EvidenceOrigin.PRODUCER: 0.58,
        EvidenceOrigin.NWDAF: 0.38,
        EvidenceOrigin.POLICY: 0.84,
        EvidenceOrigin.AUDIT: 0.46,
    },
    SemanticFact.DISCOVERY: {
        EvidenceOrigin.CONSUMER: 0.52,
        EvidenceOrigin.SCP: 0.64,
        EvidenceOrigin.NRF: 0.86,
        EvidenceOrigin.PRODUCER: 0.50,
        EvidenceOrigin.NWDAF: 0.36,
        EvidenceOrigin.POLICY: 0.72,
        EvidenceOrigin.AUDIT: 0.48,
    },
    SemanticFact.PROCEDURE: {
        EvidenceOrigin.CONSUMER: 0.58,
        EvidenceOrigin.SCP: 0.42,
        EvidenceOrigin.NRF: 0.36,
        EvidenceOrigin.PRODUCER: 0.70,
        EvidenceOrigin.NWDAF: 0.52,
        EvidenceOrigin.POLICY: 0.68,
        EvidenceOrigin.AUDIT: 0.72,
    },
    SemanticFact.ROUTE: {
        EvidenceOrigin.CONSUMER: 0.58,
        EvidenceOrigin.SCP: 0.84,
        EvidenceOrigin.NRF: 0.40,
        EvidenceOrigin.PRODUCER: 0.62,
        EvidenceOrigin.NWDAF: 0.34,
        EvidenceOrigin.POLICY: 0.46,
        EvidenceOrigin.AUDIT: 0.74,
    },
}


ORIGIN_LATENCY_MEDIAN_S: Mapping[EvidenceOrigin, float] = {
    EvidenceOrigin.CONSUMER: 0.045,
    EvidenceOrigin.SCP: 0.080,
    EvidenceOrigin.NRF: 0.160,
    EvidenceOrigin.PRODUCER: 0.110,
    EvidenceOrigin.NWDAF: 0.900,
    EvidenceOrigin.POLICY: 0.260,
    EvidenceOrigin.AUDIT: 0.520,
}

FACT_LATENCY_FACTOR: Mapping[SemanticFact, float] = {
    SemanticFact.TOKEN: 1.00,
    SemanticFact.NOTIFICATION: 1.20,
    SemanticFact.SLICE: 1.25,
    SemanticFact.DISCOVERY: 1.40,
    SemanticFact.PROCEDURE: 1.60,
    SemanticFact.ROUTE: 1.10,
}


ORIGIN_TO_DOMAIN: Mapping[EvidenceOrigin, Optional[TrustDomain]] = {
    EvidenceOrigin.CONSUMER: TrustDomain.CONSUMER,
    EvidenceOrigin.SCP: TrustDomain.SCP,
    EvidenceOrigin.NRF: TrustDomain.NRF,
    EvidenceOrigin.PRODUCER: TrustDomain.PRODUCER,
    EvidenceOrigin.NWDAF: TrustDomain.NWDAF,
    EvidenceOrigin.POLICY: None,
    EvidenceOrigin.AUDIT: None,
}


@dataclass(frozen=True)
class OriginObservation:
    event_id: str
    fact: SemanticFact
    origin: EvidenceOrigin
    trust_root: str
    state: EvidenceState
    observed_at: float
    confidence: float = 1.0

    @property
    def inconsistent(self) -> bool:
        return self.state == EvidenceState.INCONSISTENT


@dataclass
class V5Window:
    seed: int
    window_id: int
    context: str
    communication_mode: CommunicationMode
    window_start: float
    window_end: float
    events: list
    observations: List[OriginObservation]
    transport_residuals: Dict[TrustDomain, float]
    transport_observed_at: Dict[TrustDomain, float]
    label: int
    attack_family: AttackFamily
    hidden_calls: int
    sophistication: Sophistication
    compromised_origins: FrozenSet[EvidenceOrigin]
    attack_event_ids: FrozenSet[str]


@dataclass
class V5Config:
    shared_visibility_sigma: float = 0.35
    common_unknown_prob: float = 0.025
    false_evidence_prob: float = 0.0006
    latency_jitter_sigma: float = 0.40
    soft_persistence_events: int = 2
    soft_persistence_window_s: float = 90.0


class SemanticSbaSimulator(V4Simulator):
    """V4 event/transport simulator with seven-origin semantic evidence."""

    def __init__(self, *args, v5_cfg: Optional[V5Config] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.v5_cfg = v5_cfg or V5Config()
        self.v5_rng = np.random.default_rng(self.seed + 5_000_003)

    def _origin_latency(
        self,
        origin: EvidenceOrigin,
        fact: SemanticFact,
    ) -> float:
        median = ORIGIN_LATENCY_MEDIAN_S[origin] * FACT_LATENCY_FACTOR[fact]
        sigma = self.v5_cfg.latency_jitter_sigma
        return float(
            median
            * math.exp(self.v5_rng.normal(-0.5 * sigma * sigma, sigma))
        )

    def _origin_visibility(
        self,
        fact: SemanticFact,
        origin: EvidenceOrigin,
        context: str,
        sophistication: Sophistication,
        truth_inconsistent: bool,
        shared_multiplier: float,
    ) -> float:
        p = ORIGIN_VISIBILITY[fact][origin]
        p *= CONTEXT_VISIBILITY.get(context, 1.0)
        p *= shared_multiplier
        if truth_inconsistent:
            p *= SOPHISTICATION_FACTOR[sophistication][fact]
        return float(np.clip(p, 0.02, 0.98))

    def _origin_observations(
        self,
        events,
        *,
        context: str,
        sophistication: Sophistication,
        compromised: FrozenSet[EvidenceOrigin],
        compromise_mode: CompromiseMode,
    ) -> List[OriginObservation]:
        out: List[OriginObservation] = []
        sigma = self.v5_cfg.shared_visibility_sigma

        for event in events:
            for fact in FACT_CLASS:
                truth = self._truth(event, fact)
                shared_multiplier = math.exp(
                    self.v5_rng.normal(-0.5 * sigma * sigma, sigma)
                )
                common_unknown_p = (
                    self.v5_cfg.common_unknown_prob
                    * CONTEXT_UNKNOWN.get(context, 1.0)
                )
                if truth and sophistication == Sophistication.ADAPTIVE:
                    common_unknown_p *= 1.8
                common_unknown_p = min(common_unknown_p, 0.25)
                common_unknown_draw = self.v5_rng.random()

                for origin in EvidenceOrigin:
                    # Draw all stochastic quantities before applying compromise
                    # state, so PASSIVE compromise cannot alter the RNG stream.
                    latency = self._origin_latency(origin, fact)
                    visibility_draw = self.v5_rng.random()
                    false_draw = self.v5_rng.random()
                    observed_at = event.timestamp + latency
                    p_visible = self._origin_visibility(
                        fact,
                        origin,
                        context,
                        sophistication,
                        truth,
                        shared_multiplier,
                    )

                    if (
                        origin in compromised
                        and compromise_mode == CompromiseMode.SUPPRESS
                    ):
                        state = EvidenceState.CONSISTENT
                        confidence = 0.95
                    elif (
                        origin in compromised
                        and compromise_mode == CompromiseMode.INJECT
                    ):
                        state = EvidenceState.INCONSISTENT
                        confidence = 0.95
                    elif common_unknown_draw < common_unknown_p:
                        state = EvidenceState.UNKNOWN
                        confidence = 0.0
                    elif visibility_draw > p_visible:
                        state = EvidenceState.UNKNOWN
                        confidence = 0.0
                    elif truth:
                        state = EvidenceState.INCONSISTENT
                        confidence = p_visible
                    else:
                        false_p = (
                            self.v5_cfg.false_evidence_prob
                            * CONTEXT_FALSE_EVIDENCE.get(context, 1.0)
                        )
                        state = (
                            EvidenceState.INCONSISTENT
                            if false_draw < false_p
                            else EvidenceState.CONSISTENT
                        )
                        confidence = p_visible

                    out.append(
                        OriginObservation(
                            event_id=event.event_id,
                            fact=fact,
                            origin=origin,
                            trust_root=origin.value,
                            state=state,
                            observed_at=float(observed_at),
                            confidence=float(confidence),
                        )
                    )
        return out

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
    ) -> V5Window:
        sophistication = Sophistication(sophistication)
        compromise_mode = CompromiseMode(compromise_mode)
        compromised = frozenset(EvidenceOrigin(x) for x in compromised_origins)
        base_domains = [
            ORIGIN_TO_DOMAIN[o]
            for o in compromised
            if ORIGIN_TO_DOMAIN[o] is not None
        ]

        base: V4Window = super().generate_window(
            window_id,
            context=context,
            attack_family=attack_family,
            hidden_calls=hidden_calls,
            sophistication=sophistication,
            compromised_domains=base_domains,
            compromise_mode=compromise_mode,
            adaptive_budget=adaptive_budget,
        )
        observations = self._origin_observations(
            base.events,
            context=context,
            sophistication=sophistication,
            compromised=compromised,
            compromise_mode=compromise_mode,
        )
        return V5Window(
            seed=base.seed,
            window_id=base.window_id,
            context=base.context,
            communication_mode=base.communication_mode,
            window_start=base.window_start,
            window_end=base.window_end,
            events=base.events,
            observations=observations,
            transport_residuals=base.transport_residuals,
            transport_observed_at=base.transport_observed_at,
            label=base.label,
            attack_family=base.attack_family,
            hidden_calls=base.hidden_calls,
            sophistication=base.sophistication,
            compromised_origins=compromised,
            attack_event_ids=base.attack_event_ids,
        )


class SemanticQuorumDetector:
    """Trust-origin-aware sequential quorum with optional soft persistence."""

    def __init__(
        self,
        profile: str | QuorumProfile = "q3of5",
        target_fpr: float = 0.01,
        *,
        soft_persistence_events: int = 2,
        soft_persistence_window_s: float = 90.0,
        enable_soft_persistence: bool = True,
    ):
        self.profile = (
            QUORUM_PROFILES[profile]
            if isinstance(profile, str)
            else profile
        )
        self.target_fpr = float(target_fpr)
        self.soft_persistence_events = max(1, int(soft_persistence_events))
        self.soft_persistence_window_s = float(soft_persistence_window_s)
        self.enable_soft_persistence = bool(enable_soft_persistence)
        self.transport_thresholds: Dict[str, float] = {}
        self.global_transport_threshold = math.inf

    def selected_origins(self, fact: SemanticFact) -> FrozenSet[EvidenceOrigin]:
        return frozenset(ORIGIN_ORDER[fact][: self.profile.n])

    @staticmethod
    def transport_statistic(window: V5Window) -> float:
        values = [
            window.transport_residuals[d]
            for d in TRANSPORT_OBSERVERS
            if d in window.transport_residuals
            and np.isfinite(window.transport_residuals[d])
        ]
        return float(np.median(values)) if len(values) >= 2 else float("nan")

    def fit(self, windows: Sequence[V5Window]) -> "SemanticQuorumDetector":
        if not windows or any(w.label for w in windows):
            raise ValueError("benign calibration windows required")
        pairs = [(w, self.transport_statistic(w)) for w in windows]
        pairs = [(w, v) for w, v in pairs if np.isfinite(v)]
        if not pairs:
            raise ValueError("no finite transport observations in calibration")
        q = min(max(1.0 - self.target_fpr, 0.5), 0.999999)
        values = np.asarray([v for _, v in pairs], dtype=float)
        self.global_transport_threshold = float(np.quantile(values, q))
        by_context: Dict[str, List[float]] = {}
        for window, value in pairs:
            by_context.setdefault(window.context, []).append(float(value))
        self.transport_thresholds = {
            context: float(np.quantile(vals, q))
            for context, vals in by_context.items()
        }
        return self

    def score(
        self,
        window: V5Window,
        *,
        until: Optional[float] = None,
    ) -> DetectionResult:
        threshold = float(
            self.transport_thresholds.get(
                window.context, self.global_transport_threshold
            )
        )
        observations = sorted(
            [
                o
                for o in window.observations
                if o.origin in self.selected_origins(o.fact)
                and (until is None or o.observed_at <= until)
            ],
            key=lambda o: (o.observed_at, o.event_id, o.fact.value, o.origin.value),
        )

        roots_by_key: Dict[Tuple[str, SemanticFact], set[str]] = {}
        quorum_reached: set[Tuple[str, SemanticFact]] = set()
        soft_candidates: Dict[SemanticFact, List[Tuple[str, float]]] = {}
        fact_alert_times: Dict[Tuple[str, SemanticFact], float] = {}
        event_alert_times: Dict[str, float] = {}
        fired_facts: set[SemanticFact] = set()

        def fire(event_id: str, fact: SemanticFact, at: float) -> None:
            key = (event_id, fact)
            old = fact_alert_times.get(key)
            if old is None or at < old:
                fact_alert_times[key] = float(at)
            old_event = event_alert_times.get(event_id)
            if old_event is None or at < old_event:
                event_alert_times[event_id] = float(at)
            fired_facts.add(fact)

        for obs in observations:
            if obs.state != EvidenceState.INCONSISTENT:
                continue
            key = (obs.event_id, obs.fact)
            roots = roots_by_key.setdefault(key, set())
            roots.add(obs.trust_root)
            if key in quorum_reached or len(roots) < self.profile.q:
                continue
            quorum_reached.add(key)

            fact_class = FACT_CLASS[obs.fact]
            needs_persistence = (
                fact_class == EvidenceClass.SOFT
                and self.enable_soft_persistence
                and self.soft_persistence_events > 1
            )
            if not needs_persistence:
                fire(obs.event_id, obs.fact, float(obs.observed_at))
                continue

            candidates = soft_candidates.setdefault(obs.fact, [])
            cutoff = float(obs.observed_at) - self.soft_persistence_window_s
            candidates[:] = [(eid, t) for eid, t in candidates if t >= cutoff]
            if all(eid != obs.event_id for eid, _ in candidates):
                candidates.append((obs.event_id, float(obs.observed_at)))
            if len(candidates) >= self.soft_persistence_events:
                confirmation_time = float(obs.observed_at)
                for event_id, _candidate_time in candidates:
                    fire(event_id, obs.fact, confirmation_time)

        semantic_alert = bool(event_alert_times)
        transport_stat = self.transport_statistic(window)
        finite_domains = [
            d
            for d in TRANSPORT_OBSERVERS
            if d in window.transport_residuals
            and np.isfinite(window.transport_residuals[d])
            and d in window.transport_observed_at
        ]
        transport_time = (
            max(window.transport_observed_at[d] for d in finite_domains)
            if finite_domains
            else float("nan")
        )
        transport_ready = (
            np.isfinite(transport_stat)
            and np.isfinite(threshold)
            and (
                until is None
                or (np.isfinite(transport_time) and transport_time <= until)
            )
        )
        transport_alert = bool(transport_ready and transport_stat > threshold)
        if transport_alert:
            fired_facts.add(SemanticFact.TRANSPORT)

        times = list(event_alert_times.values())
        if transport_alert and np.isfinite(transport_time):
            times.append(float(transport_time))
        first_alert_time = min(times) if times else float("nan")

        ratio = (
            transport_stat / max(abs(threshold), 1e-9)
            if np.isfinite(transport_stat) and np.isfinite(threshold)
            else 0.0
        )
        score = max(float(len(fired_facts)), float(ratio))
        unknown_count = sum(o.state == EvidenceState.UNKNOWN for o in observations)
        unknown_fraction = unknown_count / len(observations) if observations else float("nan")

        return DetectionResult(
            alert=semantic_alert or transport_alert,
            semantic_alert=semantic_alert,
            transport_alert=transport_alert,
            fired_facts=frozenset(fired_facts),
            fired_event_ids=tuple(sorted(event_alert_times)),
            transport_stat=float(transport_stat),
            transport_threshold=threshold,
            score=score,
            first_alert_time=float(first_alert_time),
            event_alert_times=event_alert_times,
            fact_alert_times=fact_alert_times,
            unknown_fraction=float(unknown_fraction),
            evidence_count=len(observations),
        )


__all__ = [
    "AttackFamily",
    "CompromiseMode",
    "DetectionResult",
    "EvidenceClass",
    "EvidenceOrigin",
    "EvidenceState",
    "FACT_CLASS",
    "OriginObservation",
    "QUORUM_PROFILES",
    "QuorumProfile",
    "SemanticFact",
    "SemanticQuorumDetector",
    "SemanticSbaSimulator",
    "Sophistication",
    "V5Config",
    "V5Window",
    "empirical_auc",
]
