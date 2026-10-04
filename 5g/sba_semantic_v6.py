"""5G SBA semantic consistency v6: Byzantine-resistant transport.

Iteration v5 showed that q3of5 semantic voting removed observed semantic false
alarms from a single INJECT origin, while residual Byzantine false alarms came
from the transport channel. V6 therefore freezes the semantic candidate at
q3of5 without global persistence and changes the transport decision.

The v6 transport score is the *second largest context/domain-normalized
residual* among the available physical transport observers. A single arbitrary
high report can occupy only the largest position; it cannot raise the second
largest score by itself. The final threshold is calibrated on benign windows,
so the score is still interpreted relative to the requested FPR.

V6 also adds synthetic legitimate semantic disturbances. They do not change
attack labels or attack-event ground truth. Instead they model short-lived
cross-NF disagreement (token refresh races, delayed policy propagation,
subscription desynchronization, NF restart/recovery and benign SCP rerouting)
by perturbing a small set of semantic observations. These are proof-of-concept
stress scenarios, not calibrated operator-network rates.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from enum import Enum
from typing import Dict, FrozenSet, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from sba_semantic_v5 import (
    AttackFamily,
    CompromiseMode,
    DetectionResult,
    EvidenceOrigin,
    EvidenceState,
    OriginObservation,
    SemanticFact,
    SemanticQuorumDetector as V5SemanticQuorumDetector,
    SemanticSbaSimulator as V5SemanticSbaSimulator,
    Sophistication,
    TrustDomain,
    TRANSPORT_OBSERVERS,
    V5Window,
    empirical_auc,
)


class TransportMode(str, Enum):
    LEGACY_MEDIAN = "legacy_median"
    BYZ_2OF4 = "byz_2of4"


class LegitimateDisturbance(str, Enum):
    NONE = "none"
    TOKEN_REFRESH_RACE = "token_refresh_race"
    POLICY_PROPAGATION_DELAY = "policy_propagation_delay"
    SUBSCRIPTION_DESYNC = "subscription_desync"
    NF_RESTART_RECOVERY = "nf_restart_recovery"
    SCP_BENIGN_REROUTE = "scp_benign_reroute"


@dataclass(frozen=True)
class DisturbanceSpec:
    fact: SemanticFact
    primary_origins: Tuple[EvidenceOrigin, EvidenceOrigin]
    escalation_origin: EvidenceOrigin
    escalation_probability: float


# A legitimate disturbance normally creates disagreement in two origins. With
# a small explicit probability a third origin sees the same transient state,
# making these scenarios capable of challenging q3of5 instead of being
# structurally impossible to alert on.
DISTURBANCE_SPECS: Mapping[LegitimateDisturbance, DisturbanceSpec] = {
    LegitimateDisturbance.TOKEN_REFRESH_RACE: DisturbanceSpec(
        SemanticFact.TOKEN,
        (EvidenceOrigin.CONSUMER, EvidenceOrigin.PRODUCER),
        EvidenceOrigin.POLICY,
        0.05,
    ),
    LegitimateDisturbance.POLICY_PROPAGATION_DELAY: DisturbanceSpec(
        SemanticFact.SLICE,
        (EvidenceOrigin.POLICY, EvidenceOrigin.NRF),
        EvidenceOrigin.SCP,
        0.05,
    ),
    LegitimateDisturbance.SUBSCRIPTION_DESYNC: DisturbanceSpec(
        SemanticFact.NOTIFICATION,
        (EvidenceOrigin.CONSUMER, EvidenceOrigin.PRODUCER),
        EvidenceOrigin.AUDIT,
        0.05,
    ),
    LegitimateDisturbance.NF_RESTART_RECOVERY: DisturbanceSpec(
        SemanticFact.PROCEDURE,
        (EvidenceOrigin.PRODUCER, EvidenceOrigin.NWDAF),
        EvidenceOrigin.AUDIT,
        0.05,
    ),
    LegitimateDisturbance.SCP_BENIGN_REROUTE: DisturbanceSpec(
        SemanticFact.ROUTE,
        (EvidenceOrigin.SCP, EvidenceOrigin.CONSUMER),
        EvidenceOrigin.AUDIT,
        0.05,
    ),
}


@dataclass
class V6Window(V5Window):
    disturbance: LegitimateDisturbance = LegitimateDisturbance.NONE
    disturbance_event_ids: FrozenSet[str] = frozenset()


@dataclass(frozen=True)
class RobustDomainModel:
    center: float
    scale: float


class SemanticSbaSimulator(V5SemanticSbaSimulator):
    """V5 simulator plus label-free legitimate semantic disturbances."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.disturbance_rng = np.random.default_rng(self.seed + 6_000_007)

    def _apply_disturbance(
        self,
        window: V5Window,
        disturbance: LegitimateDisturbance,
        disturbance_events: int,
    ) -> Tuple[List[OriginObservation], FrozenSet[str]]:
        if disturbance == LegitimateDisturbance.NONE or disturbance_events <= 0:
            return list(window.observations), frozenset()

        spec = DISTURBANCE_SPECS[disturbance]
        candidates = [
            e.event_id
            for e in window.events
            if e.event_id not in window.attack_event_ids
        ]
        if not candidates:
            return list(window.observations), frozenset()

        k = min(int(disturbance_events), len(candidates))
        chosen = frozenset(
            str(x)
            for x in self.disturbance_rng.choice(
                np.asarray(candidates, dtype=object),
                size=k,
                replace=False,
            )
        )

        escalated: set[str] = set()
        for event_id in chosen:
            if self.disturbance_rng.random() < spec.escalation_probability:
                escalated.add(event_id)

        out: List[OriginObservation] = []
        primary = set(spec.primary_origins)
        for obs in window.observations:
            if obs.event_id not in chosen or obs.fact != spec.fact:
                out.append(obs)
                continue

            force_inconsistent = obs.origin in primary or (
                obs.event_id in escalated
                and obs.origin == spec.escalation_origin
            )
            if force_inconsistent:
                out.append(
                    replace(
                        obs,
                        state=EvidenceState.INCONSISTENT,
                        confidence=max(float(obs.confidence), 0.85),
                    )
                )
            else:
                out.append(obs)
        return out, chosen

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
    ) -> V6Window:
        disturbance = LegitimateDisturbance(disturbance)
        base = super().generate_window(
            window_id,
            context=context,
            attack_family=attack_family,
            hidden_calls=hidden_calls,
            sophistication=sophistication,
            compromised_origins=compromised_origins,
            compromise_mode=compromise_mode,
            adaptive_budget=adaptive_budget,
        )
        observations, disturbed_ids = self._apply_disturbance(
            base,
            disturbance,
            disturbance_events,
        )
        data = dict(base.__dict__)
        data["observations"] = observations
        return V6Window(
            **data,
            disturbance=disturbance,
            disturbance_event_ids=disturbed_ids,
        )


class SemanticQuorumDetector(V5SemanticQuorumDetector):
    """q3of5 semantic candidate + optional Byzantine-resistant transport."""

    def __init__(
        self,
        target_fpr: float = 0.01,
        *,
        transport_mode: TransportMode | str = TransportMode.BYZ_2OF4,
    ):
        super().__init__(
            "q3of5",
            target_fpr,
            soft_persistence_events=2,
            soft_persistence_window_s=90.0,
            enable_soft_persistence=False,
        )
        self.transport_mode = TransportMode(transport_mode)
        self.robust_models: Dict[
            str, Dict[TrustDomain, RobustDomainModel]
        ] = {}
        self.global_robust_models: Dict[TrustDomain, RobustDomainModel] = {}
        self.robust_thresholds: Dict[str, float] = {}
        self.global_robust_threshold = math.inf

    @staticmethod
    def _location_scale(values: Sequence[float]) -> RobustDomainModel:
        x = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
        if len(x) == 0:
            return RobustDomainModel(0.0, 1.0)
        center = float(np.median(x))
        mad = float(np.median(np.abs(x - center)))
        scale = 1.4826 * mad
        if not np.isfinite(scale) or scale < 1e-6:
            std = float(np.std(x)) if len(x) > 1 else 0.0
            scale = std if np.isfinite(std) and std >= 1e-6 else 1.0
        return RobustDomainModel(center, scale)

    def _fit_robust_models(self, windows: Sequence[V6Window]) -> None:
        global_values: Dict[TrustDomain, List[float]] = {
            d: [] for d in TRANSPORT_OBSERVERS
        }
        context_values: Dict[str, Dict[TrustDomain, List[float]]] = {}
        for window in windows:
            per_context = context_values.setdefault(
                window.context,
                {d: [] for d in TRANSPORT_OBSERVERS},
            )
            for domain in TRANSPORT_OBSERVERS:
                value = window.transport_residuals.get(domain, float("nan"))
                if np.isfinite(value):
                    global_values[domain].append(float(value))
                    per_context[domain].append(float(value))

        self.global_robust_models = {
            d: self._location_scale(values)
            for d, values in global_values.items()
        }
        self.robust_models = {
            context: {
                d: (
                    self._location_scale(values[d])
                    if len(values[d]) >= 10
                    else self.global_robust_models[d]
                )
                for d in TRANSPORT_OBSERVERS
            }
            for context, values in context_values.items()
        }

    def _domain_z(
        self,
        window: V6Window,
        domain: TrustDomain,
    ) -> float:
        value = window.transport_residuals.get(domain, float("nan"))
        if not np.isfinite(value):
            return float("nan")
        model = self.robust_models.get(window.context, {}).get(
            domain,
            self.global_robust_models.get(domain, RobustDomainModel(0.0, 1.0)),
        )
        return float((value - model.center) / max(model.scale, 1e-9))

    def robust_transport_score(
        self,
        window: V6Window,
        *,
        until: Optional[float] = None,
    ) -> float:
        values: List[float] = []
        for domain in TRANSPORT_OBSERVERS:
            observed_at = window.transport_observed_at.get(domain, float("nan"))
            if until is not None and (
                not np.isfinite(observed_at) or observed_at > until
            ):
                continue
            z = self._domain_z(window, domain)
            if np.isfinite(z):
                values.append(float(z))
        # At least three reports are required before the 2-of-4 statistic is
        # actionable. This prevents a pair of early reports from masquerading
        # as a robust quorum when one may be Byzantine.
        if len(values) < 3:
            return float("nan")
        values.sort(reverse=True)
        return float(values[1])

    def _robust_alert_time(self, window: V6Window, threshold: float) -> float:
        arrivals = sorted(
            [
                float(t)
                for d, t in window.transport_observed_at.items()
                if d in TRANSPORT_OBSERVERS and np.isfinite(t)
            ]
        )
        for at in arrivals:
            score = self.robust_transport_score(window, until=at)
            if np.isfinite(score) and score > threshold:
                return float(at)
        return float("nan")

    def fit(self, windows: Sequence[V6Window]) -> "SemanticQuorumDetector":
        # V5 fit is retained for the legacy-median ablation and semantic model.
        super().fit(windows)
        self._fit_robust_models(windows)

        global_scores: List[float] = []
        by_context: Dict[str, List[float]] = {}
        for window in windows:
            score = self.robust_transport_score(window)
            if np.isfinite(score):
                global_scores.append(float(score))
                by_context.setdefault(window.context, []).append(float(score))
        if not global_scores:
            raise ValueError("no finite robust transport scores in calibration")

        q = min(max(1.0 - self.target_fpr, 0.5), 0.999999)
        self.global_robust_threshold = float(np.quantile(global_scores, q))
        self.robust_thresholds = {
            context: (
                float(np.quantile(scores, q))
                if len(scores) >= 10
                else self.global_robust_threshold
            )
            for context, scores in by_context.items()
        }
        return self

    def score(
        self,
        window: V6Window,
        *,
        until: Optional[float] = None,
    ) -> DetectionResult:
        legacy = super().score(window, until=until)
        if self.transport_mode == TransportMode.LEGACY_MEDIAN:
            return legacy

        # Keep v5 semantic q3of5 decision but remove its legacy transport vote.
        fired_facts = set(legacy.fired_facts)
        fired_facts.discard(SemanticFact.TRANSPORT)
        event_alert_times = dict(legacy.event_alert_times)
        fact_alert_times = dict(legacy.fact_alert_times)
        semantic_alert = bool(event_alert_times)

        threshold = float(
            self.robust_thresholds.get(
                window.context,
                self.global_robust_threshold,
            )
        )
        transport_score = self.robust_transport_score(window, until=until)
        full_transport_time = self._robust_alert_time(window, threshold)
        transport_alert = bool(
            np.isfinite(transport_score)
            and np.isfinite(threshold)
            and transport_score > threshold
            and (
                until is None
                or (
                    np.isfinite(full_transport_time)
                    and full_transport_time <= until
                )
            )
        )
        if transport_alert:
            fired_facts.add(SemanticFact.TRANSPORT)

        times = list(event_alert_times.values())
        if transport_alert and np.isfinite(full_transport_time):
            times.append(float(full_transport_time))
        first_alert_time = min(times) if times else float("nan")

        ratio = (
            transport_score / max(abs(threshold), 1e-9)
            if np.isfinite(transport_score) and np.isfinite(threshold)
            else 0.0
        )
        score = max(float(len(fired_facts)), float(ratio))

        return DetectionResult(
            alert=semantic_alert or transport_alert,
            semantic_alert=semantic_alert,
            transport_alert=transport_alert,
            fired_facts=frozenset(fired_facts),
            fired_event_ids=tuple(sorted(event_alert_times)),
            transport_stat=float(transport_score),
            transport_threshold=threshold,
            score=score,
            first_alert_time=float(first_alert_time),
            event_alert_times=event_alert_times,
            fact_alert_times=fact_alert_times,
            unknown_fraction=float(legacy.unknown_fraction),
            evidence_count=int(legacy.evidence_count),
        )


__all__ = [
    "AttackFamily",
    "CompromiseMode",
    "DetectionResult",
    "DISTURBANCE_SPECS",
    "EvidenceOrigin",
    "EvidenceState",
    "LegitimateDisturbance",
    "RobustDomainModel",
    "SemanticFact",
    "SemanticQuorumDetector",
    "SemanticSbaSimulator",
    "Sophistication",
    "TransportMode",
    "TrustDomain",
    "V6Window",
    "empirical_auc",
]
