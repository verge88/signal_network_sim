"""5G SBA semantic-consistency model v4.

Iteration v4 addresses the saturation observed in v3:

* evidence is fact x trust-domain x context dependent;
* evidence can be CONSISTENT, INCONSISTENT or UNKNOWN;
* observations have delivery latency and correlated availability;
* NAIVE / STATISTICAL / ADAPTIVE attackers reduce observability without
  directly writing a class label into detector features;
* trust-domain compromise affects every fact observed by that domain;
* quorum decisions are processed in observation-time order;
* transport is a delayed, context-calibrated statistical channel;
* the detector never reads attack family, hidden-call count, sophistication,
  compromised-domain ground truth, or attack-event IDs.

The numerical visibility/latency values are synthetic proof-of-concept
parameters. They must be calibrated against free5GC/Open5GS/operator traces
before absolute Recall/TTD values are interpreted as real-network estimates.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, FrozenSet, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    from sba_sim_v1 import PRODUCER_TYPE, SERVICES, SbaConfig, Simulator, rng_uuid4
except ImportError:
    from .sba_sim_v1 import PRODUCER_TYPE, SERVICES, SbaConfig, Simulator, rng_uuid4


class TrustDomain(str, Enum):
    CONSUMER = "consumer"
    SCP = "scp"
    NRF = "nrf"
    PRODUCER = "producer"
    NWDAF = "nwdaf"


class CommunicationMode(str, Enum):
    INDIRECT_REQUIRED = "indirect_required"
    DIRECT_ALLOWED = "direct_allowed"


class CompromiseMode(str, Enum):
    PASSIVE = "passive"
    SUPPRESS = "suppress"
    INJECT = "inject"


class Sophistication(str, Enum):
    NAIVE = "naive"
    STATISTICAL = "statistical"
    ADAPTIVE = "adaptive"


class SemanticFact(str, Enum):
    TOKEN = "token"
    NOTIFICATION = "notification"
    SLICE = "slice"
    DISCOVERY = "discovery"
    PROCEDURE = "procedure"
    ROUTE = "route"
    TRANSPORT = "transport"


class EventKind(str, Enum):
    SERVICE_CALL = "service_call"
    NOTIFICATION = "notification"


class AttackFamily(str, Enum):
    NONE = "none"
    SCP_BYPASS = "scp_bypass"
    NO_TOKEN = "no_token"
    NOTIFY_ABUSE = "notify_abuse"
    COARSE_SCOPE = "coarse_scope"
    DISCOVERY_POISON = "discovery_poison"
    PROCEDURE_SKIP = "procedure_skip"
    CROSS_SLICE_CHAIN = "cross_slice_chain"


class EvidenceState(str, Enum):
    CONSISTENT = "consistent"
    INCONSISTENT = "inconsistent"
    UNKNOWN = "unknown"


@dataclass(frozen=True, order=True)
class Snssai:
    sst: int
    sd: str = ""


DEFAULT_SLICES = (
    Snssai(1, "010203"),
    Snssai(1, "112233"),
    Snssai(2, "445566"),
)


@dataclass(frozen=True)
class NFProfile:
    nf_instance_id: str
    nf_type: str
    services: FrozenSet[str]
    snssais: FrozenSet[Snssai]

    def serves(self, service: str, snssai: Snssai) -> bool:
        return service in self.services and snssai in self.snssais


@dataclass(frozen=True)
class SemanticAccessToken:
    token_id: str
    consumer_id: str
    producer_id: str
    producer_type: str
    service: str
    snssais: FrozenSet[Snssai]
    nbf: float
    exp: float

    def authorizes(self, event: "SbaSemanticEvent") -> bool:
        return (
            event.kind == EventKind.SERVICE_CALL
            and self.consumer_id == event.consumer_id
            and self.producer_id == event.producer_id
            and self.producer_type == event.producer_type
            and self.service == event.service
            and self.nbf <= event.timestamp <= self.exp
            and event.snssai in self.snssais
        )


@dataclass
class SbaSemanticEvent:
    event_id: str
    correlation_id: str
    timestamp: float
    kind: EventKind
    consumer_id: str
    producer_id: str
    producer_type: str
    service: str
    snssai: Snssai
    communication_mode: CommunicationMode
    discovered_producer_id: str
    intended_route: Tuple[str, ...]
    actual_route: Tuple[str, ...]
    token: Optional[SemanticAccessToken]
    subscription_active: bool = True
    procedure_predecessor_ok: bool = True
    result: str = "200"
    attack_variant: str = ""


@dataclass(frozen=True)
class FactObservation:
    event_id: str
    fact: SemanticFact
    domain: TrustDomain
    state: EvidenceState
    observed_at: float
    confidence: float = 1.0

    @property
    def available(self) -> bool:
        return self.state != EvidenceState.UNKNOWN

    @property
    def inconsistent(self) -> bool:
        return self.state == EvidenceState.INCONSISTENT


@dataclass
class SemanticWindow:
    seed: int
    window_id: int
    context: str
    communication_mode: CommunicationMode
    window_start: float
    window_end: float
    events: List[SbaSemanticEvent]
    observations: List[FactObservation]
    transport_residuals: Dict[TrustDomain, float]
    transport_observed_at: Dict[TrustDomain, float]
    label: int = 0
    attack_family: AttackFamily = AttackFamily.NONE
    hidden_calls: int = 0
    sophistication: Sophistication = Sophistication.ADAPTIVE
    compromised_domains: FrozenSet[TrustDomain] = frozenset()
    attack_event_ids: FrozenSet[str] = frozenset()

    def observations_for(
        self, fact: SemanticFact, event_id: str
    ) -> List[FactObservation]:
        return [
            o
            for o in self.observations
            if o.fact == fact and o.event_id == event_id
        ]


@dataclass
class SemanticConfig:
    events_per_window_mean: float = 80.0
    transport_observer_noise: float = 0.45
    transport_missing_prob: float = 0.025
    token_lifetime_s: float = 1800.0
    route_scp_name: str = "scp"
    shared_visibility_sigma: float = 0.35
    common_unknown_prob: float = 0.025
    latency_jitter_sigma: float = 0.40
    false_evidence_prob: float = 0.0006


ALL_DOMAINS = tuple(TrustDomain)
FACT_OBSERVERS: Mapping[SemanticFact, Tuple[TrustDomain, ...]] = {
    SemanticFact.TOKEN: (
        TrustDomain.CONSUMER,
        TrustDomain.NRF,
        TrustDomain.PRODUCER,
        TrustDomain.NWDAF,
    ),
    SemanticFact.NOTIFICATION: (
        TrustDomain.CONSUMER,
        TrustDomain.PRODUCER,
        TrustDomain.NWDAF,
    ),
    SemanticFact.SLICE: ALL_DOMAINS,
    SemanticFact.DISCOVERY: ALL_DOMAINS,
    SemanticFact.PROCEDURE: (
        TrustDomain.CONSUMER,
        TrustDomain.PRODUCER,
        TrustDomain.NWDAF,
    ),
    SemanticFact.ROUTE: ALL_DOMAINS,
}
FACT_QUORUM = {
    SemanticFact.TOKEN: 3,
    SemanticFact.NOTIFICATION: 2,
    SemanticFact.SLICE: 3,
    SemanticFact.DISCOVERY: 3,
    SemanticFact.PROCEDURE: 2,
    SemanticFact.ROUTE: 3,
}
TRANSPORT_OBSERVERS = (
    TrustDomain.CONSUMER,
    TrustDomain.SCP,
    TrustDomain.PRODUCER,
    TrustDomain.NWDAF,
)

# Probability that a domain can produce usable evidence for a semantic fact
# before context/sophistication/correlation modifiers are applied.
VISIBILITY: Mapping[SemanticFact, Mapping[TrustDomain, float]] = {
    SemanticFact.TOKEN: {
        TrustDomain.CONSUMER: 0.72,
        TrustDomain.NRF: 0.88,
        TrustDomain.PRODUCER: 0.68,
        TrustDomain.NWDAF: 0.42,
    },
    SemanticFact.NOTIFICATION: {
        TrustDomain.CONSUMER: 0.60,
        TrustDomain.PRODUCER: 0.76,
        TrustDomain.NWDAF: 0.44,
    },
    SemanticFact.SLICE: {
        TrustDomain.CONSUMER: 0.60,
        TrustDomain.SCP: 0.68,
        TrustDomain.NRF: 0.82,
        TrustDomain.PRODUCER: 0.58,
        TrustDomain.NWDAF: 0.38,
    },
    SemanticFact.DISCOVERY: {
        TrustDomain.CONSUMER: 0.52,
        TrustDomain.SCP: 0.64,
        TrustDomain.NRF: 0.86,
        TrustDomain.PRODUCER: 0.50,
        TrustDomain.NWDAF: 0.36,
    },
    SemanticFact.PROCEDURE: {
        TrustDomain.CONSUMER: 0.58,
        TrustDomain.PRODUCER: 0.70,
        TrustDomain.NWDAF: 0.52,
    },
    SemanticFact.ROUTE: {
        TrustDomain.CONSUMER: 0.58,
        TrustDomain.SCP: 0.84,
        TrustDomain.NRF: 0.40,
        TrustDomain.PRODUCER: 0.62,
        TrustDomain.NWDAF: 0.34,
    },
}

CONTEXT_VISIBILITY = {
    "normal_indirect": 1.00,
    "diurnal_peak": 0.93,
    "scale_in": 0.88,
    "api_change": 0.80,
    "code_upgrade": 0.74,
    "direct_allowed": 0.95,
}
CONTEXT_FALSE_EVIDENCE = {
    "normal_indirect": 1.00,
    "diurnal_peak": 1.20,
    "scale_in": 1.50,
    "api_change": 2.50,
    "code_upgrade": 3.00,
    "direct_allowed": 1.10,
}
CONTEXT_UNKNOWN = {
    "normal_indirect": 1.00,
    "diurnal_peak": 1.15,
    "scale_in": 1.35,
    "api_change": 1.45,
    "code_upgrade": 1.65,
    "direct_allowed": 1.05,
}

SOPHISTICATION_FACTOR: Mapping[
    Sophistication, Mapping[SemanticFact, float]
] = {
    Sophistication.NAIVE: {f: 1.00 for f in FACT_OBSERVERS},
    Sophistication.STATISTICAL: {
        SemanticFact.TOKEN: 0.84,
        SemanticFact.NOTIFICATION: 0.82,
        SemanticFact.SLICE: 0.80,
        SemanticFact.DISCOVERY: 0.78,
        SemanticFact.PROCEDURE: 0.80,
        SemanticFact.ROUTE: 0.74,
    },
    Sophistication.ADAPTIVE: {
        SemanticFact.TOKEN: 0.66,
        SemanticFact.NOTIFICATION: 0.64,
        SemanticFact.SLICE: 0.58,
        SemanticFact.DISCOVERY: 0.56,
        SemanticFact.PROCEDURE: 0.60,
        SemanticFact.ROUTE: 0.50,
    },
}
SOPHISTICATION_BUDGET = {
    Sophistication.NAIVE: 0.0,
    Sophistication.STATISTICAL: 0.5,
    Sophistication.ADAPTIVE: 1.0,
}

DOMAIN_LATENCY_MEDIAN_S = {
    TrustDomain.CONSUMER: 0.045,
    TrustDomain.SCP: 0.080,
    TrustDomain.NRF: 0.160,
    TrustDomain.PRODUCER: 0.110,
    TrustDomain.NWDAF: 0.900,
}
FACT_LATENCY_FACTOR = {
    SemanticFact.TOKEN: 1.00,
    SemanticFact.NOTIFICATION: 1.20,
    SemanticFact.SLICE: 1.25,
    SemanticFact.DISCOVERY: 1.40,
    SemanticFact.PROCEDURE: 1.60,
    SemanticFact.ROUTE: 1.10,
}


@dataclass
class DetectionResult:
    alert: bool
    semantic_alert: bool
    transport_alert: bool
    fired_facts: FrozenSet[SemanticFact]
    fired_event_ids: Tuple[str, ...]
    transport_stat: float
    transport_threshold: float
    score: float
    first_alert_time: float = float("nan")
    event_alert_times: Dict[str, float] = field(default_factory=dict)
    fact_alert_times: Dict[Tuple[str, SemanticFact], float] = field(
        default_factory=dict
    )
    unknown_fraction: float = float("nan")
    evidence_count: int = 0


class SemanticQuorumDetector:
    """Sequential semantic quorum + delayed statistical transport channel."""

    def __init__(self, target_fpr: float = 0.01):
        self.target_fpr = float(target_fpr)
        self.transport_thresholds: Dict[str, float] = {}
        self.global_transport_threshold = math.inf

    @staticmethod
    def transport_statistic(window: SemanticWindow) -> float:
        values = [
            window.transport_residuals[d]
            for d in TRANSPORT_OBSERVERS
            if d in window.transport_residuals
            and np.isfinite(window.transport_residuals[d])
        ]
        if len(values) < 2:
            return float("nan")
        return float(np.median(values))

    def fit(
        self, windows: Sequence[SemanticWindow]
    ) -> "SemanticQuorumDetector":
        if not windows or any(w.label for w in windows):
            raise ValueError("benign calibration windows required")

        pairs = [
            (w, self.transport_statistic(w))
            for w in windows
        ]
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
            context: float(np.quantile(values, q))
            for context, values in by_context.items()
        }
        return self

    def score(
        self,
        window: SemanticWindow,
        *,
        until: Optional[float] = None,
    ) -> DetectionResult:
        threshold = float(
            self.transport_thresholds.get(
                window.context, self.global_transport_threshold
            )
        )

        observations = sorted(
            (
                o
                for o in window.observations
                if until is None or o.observed_at <= until
            ),
            key=lambda o: (o.observed_at, o.event_id, o.fact.value),
        )

        inconsistent_counts: Dict[Tuple[str, SemanticFact], int] = {}
        fact_alert_times: Dict[Tuple[str, SemanticFact], float] = {}
        event_alert_times: Dict[str, float] = {}
        fired_facts: set[SemanticFact] = set()

        for obs in observations:
            if obs.state != EvidenceState.INCONSISTENT:
                continue
            key = (obs.event_id, obs.fact)
            inconsistent_counts[key] = inconsistent_counts.get(key, 0) + 1
            if (
                key not in fact_alert_times
                and inconsistent_counts[key] >= FACT_QUORUM[obs.fact]
            ):
                fact_alert_times[key] = float(obs.observed_at)
                fired_facts.add(obs.fact)
                previous = event_alert_times.get(obs.event_id)
                if previous is None or obs.observed_at < previous:
                    event_alert_times[obs.event_id] = float(obs.observed_at)

        semantic_alert = bool(event_alert_times)

        transport_stat = self.transport_statistic(window)
        transport_time = float("nan")
        transport_alert = False
        finite_transport_domains = [
            d
            for d in TRANSPORT_OBSERVERS
            if d in window.transport_residuals
            and np.isfinite(window.transport_residuals[d])
            and d in window.transport_observed_at
        ]
        if finite_transport_domains:
            transport_time = max(
                window.transport_observed_at[d]
                for d in finite_transport_domains
            )

        transport_ready = (
            np.isfinite(transport_stat)
            and np.isfinite(threshold)
            and (
                until is None
                or (
                    np.isfinite(transport_time)
                    and transport_time <= until
                )
            )
        )
        if transport_ready and transport_stat > threshold:
            transport_alert = True
            fired_facts.add(SemanticFact.TRANSPORT)

        alert_times = list(event_alert_times.values())
        if transport_alert and np.isfinite(transport_time):
            alert_times.append(float(transport_time))
        first_alert_time = (
            min(alert_times) if alert_times else float("nan")
        )

        ratio = (
            transport_stat / max(abs(threshold), 1e-9)
            if np.isfinite(transport_stat) and np.isfinite(threshold)
            else 0.0
        )
        score = max(float(len(fired_facts)), float(ratio))

        unknown_count = sum(
            o.state == EvidenceState.UNKNOWN for o in observations
        )
        unknown_fraction = (
            unknown_count / len(observations)
            if observations
            else float("nan")
        )

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


class SemanticSbaSimulator:
    CONTROL_CONTEXTS = (
        "normal_indirect",
        "diurnal_peak",
        "scale_in",
        "api_change",
        "code_upgrade",
        "direct_allowed",
    )

    def __init__(
        self,
        cfg: Optional[SbaConfig] = None,
        seed: int = 0,
        semantic_cfg: Optional[SemanticConfig] = None,
    ):
        self.cfg = cfg or SbaConfig()
        self.seed = int(seed)
        self.semantic_cfg = semantic_cfg or SemanticConfig()
        self.rng = np.random.default_rng(seed)
        self.base = Simulator(self.cfg, seed=seed + 104729)
        self.t = 0.0
        self.consumer_ids = [
            f"nf-consumer-{i:02d}" for i in range(self.cfg.n_consumers)
        ]
        self.consumer_profiles, self.producer_profiles = self._profiles()

    def _profiles(self):
        consumers: Dict[str, NFProfile] = {}
        for i, consumer_id in enumerate(self.consumer_ids):
            slices = {DEFAULT_SLICES[i % 2]}
            if i % 4 == 0:
                slices.add(DEFAULT_SLICES[2])
            consumers[consumer_id] = NFProfile(
                consumer_id,
                "CONSUMER",
                frozenset(SERVICES),
                frozenset(slices),
            )

        producers: Dict[str, NFProfile] = {}
        for service in SERVICES:
            producer_id = f"{PRODUCER_TYPE[service].lower()}-01"
            old = producers.get(producer_id)
            services = set(old.services) if old else set()
            services.add(service)
            producers[producer_id] = NFProfile(
                producer_id,
                PRODUCER_TYPE[service],
                frozenset(services),
                frozenset(DEFAULT_SLICES),
            )
        return consumers, producers

    @staticmethod
    def communication_mode_for_context(
        context: str,
    ) -> CommunicationMode:
        return (
            CommunicationMode.DIRECT_ALLOWED
            if context == "direct_allowed"
            else CommunicationMode.INDIRECT_REQUIRED
        )

    def _producer(self, service: str) -> NFProfile:
        return next(
            p for p in self.producer_profiles.values()
            if service in p.services
        )

    def _token(
        self,
        consumer: NFProfile,
        producer: NFProfile,
        service: str,
        slices: FrozenSet[Snssai],
        timestamp: float,
        *,
        consumer_id: Optional[str] = None,
        exp: Optional[float] = None,
    ) -> SemanticAccessToken:
        return SemanticAccessToken(
            str(rng_uuid4(self.rng)),
            consumer_id or consumer.nf_instance_id,
            producer.nf_instance_id,
            producer.nf_type,
            service,
            slices,
            timestamp - 60.0,
            (
                timestamp + self.semantic_cfg.token_lifetime_s
                if exp is None
                else float(exp)
            ),
        )

    def _event(
        self,
        context: str,
        kind: EventKind = EventKind.SERVICE_CALL,
    ) -> SbaSemanticEvent:
        mode = self.communication_mode_for_context(context)
        consumer = self.consumer_profiles[
            str(self.rng.choice(self.consumer_ids))
        ]
        service = str(self.rng.choice(SERVICES))
        producer = self._producer(service)
        valid = sorted(consumer.snssais & producer.snssais)
        snssai = valid[int(self.rng.integers(0, len(valid)))]
        timestamp = self.t + float(self.rng.random() * self.cfg.window_s)
        token = self._token(
            consumer,
            producer,
            service,
            frozenset(consumer.snssais & producer.snssais),
            timestamp,
        )

        if (
            mode == CommunicationMode.DIRECT_ALLOWED
            and self.rng.random() < 0.5
        ):
            route = (consumer.nf_instance_id, producer.nf_instance_id)
        else:
            route = (
                consumer.nf_instance_id,
                self.semantic_cfg.route_scp_name,
                producer.nf_instance_id,
            )

        return SbaSemanticEvent(
            event_id=str(rng_uuid4(self.rng)),
            correlation_id=str(rng_uuid4(self.rng)),
            timestamp=timestamp,
            kind=kind,
            consumer_id=consumer.nf_instance_id,
            producer_id=producer.nf_instance_id,
            producer_type=producer.nf_type,
            service=service,
            snssai=snssai,
            communication_mode=mode,
            discovered_producer_id=producer.nf_instance_id,
            intended_route=route,
            actual_route=route,
            token=token,
        )

    def _attack_event(
        self,
        family: AttackFamily,
        context: str,
        sophistication: Sophistication,
    ) -> SbaSemanticEvent:
        kind = (
            EventKind.NOTIFICATION
            if family == AttackFamily.NOTIFY_ABUSE
            else EventKind.SERVICE_CALL
        )
        event = self._event(context, kind)
        consumer = self.consumer_profiles[event.consumer_id]
        producer = self.producer_profiles[event.producer_id]

        if family == AttackFamily.SCP_BYPASS:
            event.communication_mode = CommunicationMode.INDIRECT_REQUIRED
            event.intended_route = (
                event.consumer_id,
                self.semantic_cfg.route_scp_name,
                event.producer_id,
            )
            event.actual_route = (event.consumer_id, event.producer_id)
            event.attack_variant = (
                "minimal_direct_bypass"
                if sophistication == Sophistication.ADAPTIVE
                else "direct_bypass"
            )

        elif family == AttackFamily.NO_TOKEN:
            if sophistication == Sophistication.NAIVE:
                event.token = None
                event.attack_variant = "missing_token"
            elif sophistication == Sophistication.STATISTICAL:
                event.token = self._token(
                    consumer,
                    producer,
                    event.service,
                    frozenset(consumer.snssais & producer.snssais),
                    event.timestamp,
                    exp=event.timestamp - 1.0,
                )
                event.attack_variant = "expired_token"
            else:
                event.token = self._token(
                    consumer,
                    producer,
                    event.service,
                    frozenset(consumer.snssais & producer.snssais),
                    event.timestamp,
                    consumer_id="shadow-consumer",
                )
                event.attack_variant = "plausible_wrong_subject"

        elif family == AttackFamily.NOTIFY_ABUSE:
            event.subscription_active = False
            event.attack_variant = (
                "stale_subscription"
                if sophistication == Sophistication.ADAPTIVE
                else "orphan_notification"
            )

        elif family in (
            AttackFamily.COARSE_SCOPE,
            AttackFamily.CROSS_SLICE_CHAIN,
        ):
            forbidden = [
                s for s in producer.snssais if s not in consumer.snssais
            ]
            if forbidden:
                event.snssai = forbidden[0]
            event.token = self._token(
                consumer,
                producer,
                event.service,
                producer.snssais,
                event.timestamp,
            )
            event.attack_variant = "overbroad_scope"
            if family == AttackFamily.CROSS_SLICE_CHAIN:
                event.discovered_producer_id = "stale-producer-01"
                event.procedure_predecessor_ok = False
                event.attack_variant = "cross_slice_chain"

        elif family == AttackFamily.DISCOVERY_POISON:
            event.discovered_producer_id = (
                "stale-producer-01"
                if sophistication == Sophistication.ADAPTIVE
                else "rogue-producer-01"
            )
            event.attack_variant = "stale_discovery"

        elif family == AttackFamily.PROCEDURE_SKIP:
            event.procedure_predecessor_ok = False
            event.attack_variant = (
                "minimal_predecessor_skip"
                if sophistication == Sophistication.ADAPTIVE
                else "procedure_skip"
            )

        return event

    def _truth(
        self,
        event: SbaSemanticEvent,
        fact: SemanticFact,
    ) -> bool:
        consumer = self.consumer_profiles[event.consumer_id]
        producer = self.producer_profiles.get(event.producer_id)
        if producer is None:
            return True

        if fact == SemanticFact.TOKEN:
            return (
                event.kind == EventKind.SERVICE_CALL
                and (
                    event.token is None
                    or not event.token.authorizes(event)
                )
            )
        if fact == SemanticFact.NOTIFICATION:
            return (
                event.kind == EventKind.NOTIFICATION
                and not event.subscription_active
            )
        if fact == SemanticFact.SLICE:
            return (
                event.snssai not in consumer.snssais
                or event.snssai not in producer.snssais
                or (
                    event.token is not None
                    and event.snssai not in event.token.snssais
                )
            )
        if fact == SemanticFact.DISCOVERY:
            return (
                event.discovered_producer_id != event.producer_id
                or not producer.serves(event.service, event.snssai)
            )
        if fact == SemanticFact.PROCEDURE:
            return not event.procedure_predecessor_ok
        if fact == SemanticFact.ROUTE:
            endpoints_ok = (
                len(event.actual_route) >= 2
                and event.actual_route[0] == event.consumer_id
                and event.actual_route[-1] == event.producer_id
            )
            return (
                not endpoints_ok
                or (
                    event.communication_mode
                    == CommunicationMode.INDIRECT_REQUIRED
                    and self.semantic_cfg.route_scp_name
                    not in event.actual_route[1:-1]
                )
            )
        raise KeyError(fact)

    def _latency(
        self,
        domain: TrustDomain,
        fact: SemanticFact,
    ) -> float:
        median = (
            DOMAIN_LATENCY_MEDIAN_S[domain]
            * FACT_LATENCY_FACTOR[fact]
        )
        sigma = self.semantic_cfg.latency_jitter_sigma
        return float(
            median
            * math.exp(
                self.rng.normal(-0.5 * sigma * sigma, sigma)
            )
        )

    def _visibility_probability(
        self,
        fact: SemanticFact,
        domain: TrustDomain,
        context: str,
        sophistication: Sophistication,
        truth_inconsistent: bool,
        shared_multiplier: float,
    ) -> float:
        p = VISIBILITY[fact][domain]
        p *= CONTEXT_VISIBILITY.get(context, 1.0)
        p *= shared_multiplier
        if truth_inconsistent:
            p *= SOPHISTICATION_FACTOR[sophistication][fact]
        return float(np.clip(p, 0.02, 0.98))

    def _observations(
        self,
        events: Sequence[SbaSemanticEvent],
        *,
        context: str,
        sophistication: Sophistication,
        compromised: FrozenSet[TrustDomain],
        compromise_mode: CompromiseMode,
    ) -> List[FactObservation]:
        observations: List[FactObservation] = []
        sigma = self.semantic_cfg.shared_visibility_sigma

        for event in events:
            for fact, domains in FACT_OBSERVERS.items():
                truth = self._truth(event, fact)
                shared_multiplier = math.exp(
                    self.rng.normal(-0.5 * sigma * sigma, sigma)
                )
                common_unknown_p = (
                    self.semantic_cfg.common_unknown_prob
                    * CONTEXT_UNKNOWN.get(context, 1.0)
                )
                if truth and sophistication == Sophistication.ADAPTIVE:
                    common_unknown_p *= 1.8
                common_unknown = self.rng.random() < min(
                    common_unknown_p, 0.25
                )

                for domain in domains:
                    observed_at = event.timestamp + self._latency(
                        domain, fact
                    )

                    if (
                        domain in compromised
                        and compromise_mode == CompromiseMode.SUPPRESS
                    ):
                        state = EvidenceState.CONSISTENT
                        confidence = 0.95
                    elif (
                        domain in compromised
                        and compromise_mode == CompromiseMode.INJECT
                    ):
                        state = EvidenceState.INCONSISTENT
                        confidence = 0.95
                    elif common_unknown:
                        state = EvidenceState.UNKNOWN
                        confidence = 0.0
                    else:
                        p_visible = self._visibility_probability(
                            fact,
                            domain,
                            context,
                            sophistication,
                            truth,
                            shared_multiplier,
                        )
                        if self.rng.random() > p_visible:
                            state = EvidenceState.UNKNOWN
                            confidence = 0.0
                        elif truth:
                            state = EvidenceState.INCONSISTENT
                            confidence = p_visible
                        else:
                            false_p = (
                                self.semantic_cfg.false_evidence_prob
                                * CONTEXT_FALSE_EVIDENCE.get(
                                    context, 1.0
                                )
                            )
                            state = (
                                EvidenceState.INCONSISTENT
                                if self.rng.random() < false_p
                                else EvidenceState.CONSISTENT
                            )
                            confidence = p_visible

                    observations.append(
                        FactObservation(
                            event_id=event.event_id,
                            fact=fact,
                            domain=domain,
                            state=state,
                            observed_at=float(observed_at),
                            confidence=float(confidence),
                        )
                    )

        return observations

    def _transport(
        self,
        base_window,
        *,
        context: str,
        window_end: float,
        compromised: FrozenSet[TrustDomain],
        compromise_mode: CompromiseMode,
    ) -> Tuple[
        Dict[TrustDomain, float],
        Dict[TrustDomain, float],
    ]:
        delta = float(base_window.served_total - base_window.streams_total)
        scale = max(
            math.sqrt(max(base_window.lam_hat * base_window.et_hat, 1.0)),
            1.0,
        )
        center = delta / scale

        shared_noise = float(
            self.rng.normal(0.0, self.semantic_cfg.transport_observer_noise * 0.55)
        )
        residuals: Dict[TrustDomain, float] = {}
        observed_at: Dict[TrustDomain, float] = {}

        context_missing = (
            self.semantic_cfg.transport_missing_prob
            * CONTEXT_UNKNOWN.get(context, 1.0)
        )

        for domain in TRANSPORT_OBSERVERS:
            value = (
                center
                + shared_noise
                + float(
                    self.rng.normal(
                        0.0,
                        self.semantic_cfg.transport_observer_noise * 0.75,
                    )
                )
            )

            if (
                domain in compromised
                and compromise_mode == CompromiseMode.SUPPRESS
            ):
                value = float(
                    self.rng.normal(
                        0.0, self.semantic_cfg.transport_observer_noise
                    )
                )
            elif (
                domain in compromised
                and compromise_mode == CompromiseMode.INJECT
            ):
                value += 20.0
            elif self.rng.random() < min(context_missing, 0.20):
                value = float("nan")

            residuals[domain] = float(value)
            median = DOMAIN_LATENCY_MEDIAN_S[domain]
            observed_at[domain] = float(
                window_end
                + median
                * math.exp(
                    self.rng.normal(
                        -0.5
                        * self.semantic_cfg.latency_jitter_sigma**2,
                        self.semantic_cfg.latency_jitter_sigma,
                    )
                )
            )

        return residuals, observed_at

    def generate_window(
        self,
        window_id: int,
        *,
        context: str = "normal_indirect",
        attack_family: AttackFamily | str = AttackFamily.NONE,
        hidden_calls: int = 0,
        sophistication: Sophistication | str = Sophistication.ADAPTIVE,
        compromised_domains: Iterable[TrustDomain | str] = (),
        compromise_mode: CompromiseMode | str = CompromiseMode.PASSIVE,
        adaptive_budget: Optional[float] = None,
    ) -> SemanticWindow:
        family = AttackFamily(attack_family)
        sophistication = Sophistication(sophistication)
        compromise_mode = CompromiseMode(compromise_mode)
        compromised = frozenset(
            TrustDomain(d) for d in compromised_domains
        )

        window_start = float(self.t)
        window_end = float(self.t + self.cfg.window_s)
        control = (
            ""
            if context in ("normal_indirect", "direct_allowed")
            else context
        )

        budget = (
            SOPHISTICATION_BUDGET[sophistication]
            if adaptive_budget is None
            else float(np.clip(adaptive_budget, 0.0, 1.0))
        )

        if (
            family
            in (
                AttackFamily.SCP_BYPASS,
                AttackFamily.NO_TOKEN,
                AttackFamily.NOTIFY_ABUSE,
            )
            and hidden_calls > 0
        ):
            base_window = self.base.attack_window(
                window_id,
                hidden=int(hidden_calls),
                via_scope=family.value,
                family="M1_additive",
                budget=budget,
                params=np.asarray([0.5, 0.5]),
                mode="eval",
            )
        else:
            base_window = self.base.benign_window(
                window_id, mode="eval", control=control
            )

        benign_count = int(
            max(0, self.rng.poisson(self.semantic_cfg.events_per_window_mean))
        )
        events = [self._event(context) for _ in range(benign_count)]

        attack_events: List[SbaSemanticEvent] = []
        if family != AttackFamily.NONE and hidden_calls > 0:
            attack_events = [
                self._attack_event(family, context, sophistication)
                for _ in range(int(hidden_calls))
            ]
            events.extend(attack_events)

        observations = self._observations(
            events,
            context=context,
            sophistication=sophistication,
            compromised=compromised,
            compromise_mode=compromise_mode,
        )
        transport, transport_times = self._transport(
            base_window,
            context=context,
            window_end=window_end,
            compromised=compromised,
            compromise_mode=compromise_mode,
        )

        self.t += self.cfg.window_s
        return SemanticWindow(
            seed=self.seed,
            window_id=window_id,
            context=context,
            communication_mode=self.communication_mode_for_context(context),
            window_start=window_start,
            window_end=window_end,
            events=events,
            observations=observations,
            transport_residuals=transport,
            transport_observed_at=transport_times,
            label=int(family != AttackFamily.NONE and hidden_calls > 0),
            attack_family=family,
            hidden_calls=int(hidden_calls),
            sophistication=sophistication,
            compromised_domains=compromised,
            attack_event_ids=frozenset(e.event_id for e in attack_events),
        )


def empirical_auc(
    labels: Sequence[int], scores: Sequence[float]
) -> float:
    y = np.asarray(labels, dtype=int)
    s = np.asarray(scores, dtype=float)
    pos = s[y == 1]
    neg = s[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    wins = 0.0
    for value in pos:
        wins += float(np.sum(value > neg))
        wins += 0.5 * float(np.sum(value == neg))
    return wins / float(len(pos) * len(neg))
