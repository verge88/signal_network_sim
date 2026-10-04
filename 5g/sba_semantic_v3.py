"""Event-level trust-domain semantics for the existing 5G SBA simulator.

Scientific contract:
* compromise is a capability, not an observable class signature;
* direct SBI is benign when policy allows it;
* token/notification/slice/discovery/procedure/route are semantic gates;
* transport remains a context-calibrated statistical invariant;
* detector code never reads attack labels or compromised-domain ground truth.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Dict, FrozenSet, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    from sba_sim_v1 import PRODUCER_TYPE, SERVICES, SbaConfig, Simulator, rng_uuid4
except ImportError:
    from .sba_sim_v1 import PRODUCER_TYPE, SERVICES, SbaConfig, Simulator, rng_uuid4


class TrustDomain(str, Enum):
    CONSUMER="consumer"; SCP="scp"; NRF="nrf"; PRODUCER="producer"; NWDAF="nwdaf"
class CommunicationMode(str, Enum):
    INDIRECT_REQUIRED="indirect_required"; DIRECT_ALLOWED="direct_allowed"
class CompromiseMode(str, Enum):
    PASSIVE="passive"; SUPPRESS="suppress"; INJECT="inject"
class SemanticFact(str, Enum):
    TOKEN="token"; NOTIFICATION="notification"; SLICE="slice"; DISCOVERY="discovery"; PROCEDURE="procedure"; ROUTE="route"; TRANSPORT="transport"
class EventKind(str, Enum):
    SERVICE_CALL="service_call"; NOTIFICATION="notification"
class AttackFamily(str, Enum):
    NONE="none"; SCP_BYPASS="scp_bypass"; NO_TOKEN="no_token"; NOTIFY_ABUSE="notify_abuse"; COARSE_SCOPE="coarse_scope"; DISCOVERY_POISON="discovery_poison"; PROCEDURE_SKIP="procedure_skip"; CROSS_SLICE_CHAIN="cross_slice_chain"


@dataclass(frozen=True, order=True)
class Snssai:
    sst:int; sd:str=""

DEFAULT_SLICES=(Snssai(1,"010203"), Snssai(1,"112233"), Snssai(2,"445566"))

@dataclass(frozen=True)
class NFProfile:
    nf_instance_id:str; nf_type:str; services:FrozenSet[str]; snssais:FrozenSet[Snssai]
    def serves(self, service:str, snssai:Snssai)->bool:
        return service in self.services and snssai in self.snssais

@dataclass(frozen=True)
class SemanticAccessToken:
    token_id:str; consumer_id:str; producer_id:str; producer_type:str; service:str
    snssais:FrozenSet[Snssai]; nbf:float; exp:float
    def authorizes(self,e:"SbaSemanticEvent")->bool:
        return e.kind==EventKind.SERVICE_CALL and self.consumer_id==e.consumer_id and self.producer_id==e.producer_id and self.producer_type==e.producer_type and self.service==e.service and self.nbf<=e.timestamp<=self.exp and e.snssai in self.snssais

@dataclass
class SbaSemanticEvent:
    event_id:str; correlation_id:str; timestamp:float; kind:EventKind
    consumer_id:str; producer_id:str; producer_type:str; service:str; snssai:Snssai
    communication_mode:CommunicationMode; discovered_producer_id:str
    intended_route:Tuple[str,...]; actual_route:Tuple[str,...]
    token:Optional[SemanticAccessToken]; subscription_active:bool=True
    procedure_predecessor_ok:bool=True; result:str="200"

@dataclass(frozen=True)
class FactObservation:
    event_id:str; fact:SemanticFact; domain:TrustDomain; inconsistent:bool; available:bool=True

@dataclass
class SemanticWindow:
    seed:int; window_id:int; context:str; communication_mode:CommunicationMode
    events:List[SbaSemanticEvent]; observations:List[FactObservation]
    transport_residuals:Dict[TrustDomain,float]
    label:int=0; attack_family:AttackFamily=AttackFamily.NONE; hidden_calls:int=0
    compromised_domains:FrozenSet[TrustDomain]=frozenset()
    def observations_for(self,fact:SemanticFact,event_id:str)->List[FactObservation]:
        return [o for o in self.observations if o.fact==fact and o.event_id==event_id and o.available]

@dataclass
class SemanticConfig:
    events_per_window_mean:float=80.0
    evidence_miss_prob:float=0.03
    false_evidence_prob:float=0.0005
    transport_observer_noise:float=0.45
    token_lifetime_s:float=1800.0
    route_scp_name:str="scp"


ALL_DOMAINS=tuple(TrustDomain)
FACT_OBSERVERS:Mapping[SemanticFact,Tuple[TrustDomain,...]]={
    SemanticFact.TOKEN:(TrustDomain.CONSUMER,TrustDomain.NRF,TrustDomain.PRODUCER,TrustDomain.NWDAF),
    SemanticFact.NOTIFICATION:(TrustDomain.CONSUMER,TrustDomain.PRODUCER,TrustDomain.NWDAF),
    SemanticFact.SLICE:ALL_DOMAINS,
    SemanticFact.DISCOVERY:ALL_DOMAINS,
    SemanticFact.PROCEDURE:(TrustDomain.CONSUMER,TrustDomain.PRODUCER,TrustDomain.NWDAF),
    SemanticFact.ROUTE:ALL_DOMAINS,
}
FACT_QUORUM={SemanticFact.TOKEN:3,SemanticFact.NOTIFICATION:2,SemanticFact.SLICE:3,SemanticFact.DISCOVERY:3,SemanticFact.PROCEDURE:2,SemanticFact.ROUTE:3}
TRANSPORT_OBSERVERS=(TrustDomain.CONSUMER,TrustDomain.SCP,TrustDomain.PRODUCER,TrustDomain.NWDAF)

@dataclass
class DetectionResult:
    alert:bool; fired_facts:FrozenSet[SemanticFact]; fired_event_ids:Tuple[str,...]
    transport_stat:float; transport_threshold:float; score:float


class SemanticQuorumDetector:
    def __init__(self,target_fpr:float=0.01):
        self.target_fpr=float(target_fpr); self.transport_thresholds:Dict[str,float]={}; self.global_transport_threshold=math.inf
    @staticmethod
    def transport_statistic(w:SemanticWindow)->float:
        x=[w.transport_residuals[d] for d in TRANSPORT_OBSERVERS if d in w.transport_residuals and np.isfinite(w.transport_residuals[d])]
        return float(np.median(x)) if x else 0.0
    def fit(self,windows:Sequence[SemanticWindow])->"SemanticQuorumDetector":
        if not windows or any(w.label for w in windows): raise ValueError("benign calibration windows required")
        vals=np.asarray([self.transport_statistic(w) for w in windows],float); q=min(max(1-self.target_fpr,.5),.999999)
        self.global_transport_threshold=float(np.quantile(vals,q)); by:Dict[str,List[float]]={}
        for w,v in zip(windows,vals): by.setdefault(w.context,[]).append(float(v))
        self.transport_thresholds={c:float(np.quantile(v,q)) for c,v in by.items()}; return self
    def score(self,w:SemanticWindow)->DetectionResult:
        facts:set[SemanticFact]=set(); events:set[str]=set()
        for eid in {e.event_id for e in w.events}:
            for fact in FACT_OBSERVERS:
                if sum(o.inconsistent for o in w.observations_for(fact,eid))>=FACT_QUORUM[fact]:
                    facts.add(fact); events.add(eid)
        stat=self.transport_statistic(w); thr=float(self.transport_thresholds.get(w.context,self.global_transport_threshold))
        if stat>thr: facts.add(SemanticFact.TRANSPORT)
        ratio=stat/max(abs(thr),1e-9) if np.isfinite(thr) else 0.0
        return DetectionResult(bool(facts),frozenset(facts),tuple(sorted(events)),stat,thr,max(float(len(facts)),float(ratio)))


class SemanticSbaSimulator:
    CONTROL_CONTEXTS=("normal_indirect","diurnal_peak","scale_in","api_change","code_upgrade","direct_allowed")
    def __init__(self,cfg:Optional[SbaConfig]=None,seed:int=0,semantic_cfg:Optional[SemanticConfig]=None):
        self.cfg=cfg or SbaConfig(); self.seed=int(seed); self.semantic_cfg=semantic_cfg or SemanticConfig(); self.rng=np.random.default_rng(seed)
        self.base=Simulator(self.cfg,seed=seed+104729); self.t=0.0
        self.consumer_ids=[f"nf-consumer-{i:02d}" for i in range(self.cfg.n_consumers)]
        self.consumer_profiles,self.producer_profiles=self._profiles()
    def _profiles(self):
        cs={}
        for i,cid in enumerate(self.consumer_ids):
            s={DEFAULT_SLICES[i%2]}
            if i%4==0:s.add(DEFAULT_SLICES[2])
            cs[cid]=NFProfile(cid,"CONSUMER",frozenset(SERVICES),frozenset(s))
        ps={}
        for svc in SERVICES:
            pid=f"{PRODUCER_TYPE[svc].lower()}-01"; old=ps.get(pid); ss=set(old.services) if old else set(); ss.add(svc)
            ps[pid]=NFProfile(pid,PRODUCER_TYPE[svc],frozenset(ss),frozenset(DEFAULT_SLICES))
        return cs,ps
    @staticmethod
    def communication_mode_for_context(context:str)->CommunicationMode:
        return CommunicationMode.DIRECT_ALLOWED if context=="direct_allowed" else CommunicationMode.INDIRECT_REQUIRED
    def _producer(self,service:str)->NFProfile:
        return next(p for p in self.producer_profiles.values() if service in p.services)
    def _token(self,c:NFProfile,p:NFProfile,svc:str,slices:FrozenSet[Snssai],t:float)->SemanticAccessToken:
        return SemanticAccessToken(str(rng_uuid4(self.rng)),c.nf_instance_id,p.nf_instance_id,p.nf_type,svc,slices,t-60,t+self.semantic_cfg.token_lifetime_s)
    def _event(self,context:str,kind:EventKind=EventKind.SERVICE_CALL)->SbaSemanticEvent:
        mode=self.communication_mode_for_context(context); c=self.consumer_profiles[str(self.rng.choice(self.consumer_ids))]; svc=str(self.rng.choice(SERVICES)); p=self._producer(svc)
        valid=sorted(c.snssais&p.snssais); s=valid[int(self.rng.integers(0,len(valid)))]; t=self.t+float(self.rng.random()*self.cfg.window_s)
        tok=self._token(c,p,svc,frozenset(c.snssais&p.snssais),t)
        route=(c.nf_instance_id,p.nf_instance_id) if mode==CommunicationMode.DIRECT_ALLOWED and self.rng.random()<.5 else (c.nf_instance_id,self.semantic_cfg.route_scp_name,p.nf_instance_id)
        return SbaSemanticEvent(str(rng_uuid4(self.rng)),str(rng_uuid4(self.rng)),t,kind,c.nf_instance_id,p.nf_instance_id,p.nf_type,svc,s,mode,p.nf_instance_id,route,route,tok)
    def _attack_event(self,f:AttackFamily,context:str)->SbaSemanticEvent:
        e=self._event(context,EventKind.NOTIFICATION if f==AttackFamily.NOTIFY_ABUSE else EventKind.SERVICE_CALL); c=self.consumer_profiles[e.consumer_id]; p=self.producer_profiles[e.producer_id]
        if f==AttackFamily.SCP_BYPASS:
            e.communication_mode=CommunicationMode.INDIRECT_REQUIRED; e.intended_route=(e.consumer_id,self.semantic_cfg.route_scp_name,e.producer_id); e.actual_route=(e.consumer_id,e.producer_id)
        elif f==AttackFamily.NO_TOKEN:e.token=None
        elif f==AttackFamily.NOTIFY_ABUSE:e.subscription_active=False
        elif f in (AttackFamily.COARSE_SCOPE,AttackFamily.CROSS_SLICE_CHAIN):
            bad=[s for s in p.snssais if s not in c.snssais]; e.snssai=bad[0] if bad else e.snssai; e.token=self._token(c,p,e.service,p.snssais,e.timestamp)
            if f==AttackFamily.CROSS_SLICE_CHAIN:e.discovered_producer_id="rogue-producer-01"; e.procedure_predecessor_ok=False
        elif f==AttackFamily.DISCOVERY_POISON:e.discovered_producer_id="rogue-producer-01"
        elif f==AttackFamily.PROCEDURE_SKIP:e.procedure_predecessor_ok=False
        return e
    def _truth(self,e:SbaSemanticEvent,f:SemanticFact)->bool:
        c=self.consumer_profiles[e.consumer_id]; p=self.producer_profiles.get(e.producer_id)
        if p is None:return True
        if f==SemanticFact.TOKEN:return e.kind==EventKind.SERVICE_CALL and (e.token is None or not e.token.authorizes(e))
        if f==SemanticFact.NOTIFICATION:return e.kind==EventKind.NOTIFICATION and not e.subscription_active
        if f==SemanticFact.SLICE:return e.snssai not in c.snssais or e.snssai not in p.snssais or (e.token is not None and e.snssai not in e.token.snssais)
        if f==SemanticFact.DISCOVERY:return e.discovered_producer_id!=e.producer_id or not p.serves(e.service,e.snssai)
        if f==SemanticFact.PROCEDURE:return not e.procedure_predecessor_ok
        if f==SemanticFact.ROUTE:
            endpoints=len(e.actual_route)>=2 and e.actual_route[0]==e.consumer_id and e.actual_route[-1]==e.producer_id
            return (not endpoints) or (e.communication_mode==CommunicationMode.INDIRECT_REQUIRED and self.semantic_cfg.route_scp_name not in e.actual_route[1:-1])
        raise KeyError(f)
    def _observations(self,events,compromised:FrozenSet[TrustDomain],mode:CompromiseMode):
        out=[]
        for e in events:
            for f,domains in FACT_OBSERVERS.items():
                truth=self._truth(e,f)
                for d in domains:
                    x=truth
                    if d in compromised and mode==CompromiseMode.SUPPRESS:x=False
                    elif d in compromised and mode==CompromiseMode.INJECT:x=True
                    elif truth and self.rng.random()<self.semantic_cfg.evidence_miss_prob:x=False
                    elif not truth and self.rng.random()<self.semantic_cfg.false_evidence_prob:x=True
                    out.append(FactObservation(e.event_id,f,d,bool(x)))
        return out
    def _transport(self,w,compromised,mode):
        delta=float(w.served_total-w.streams_total); scale=max(math.sqrt(max(w.lam_hat*w.et_hat,1.0)),1.0); out={}
        for d in TRANSPORT_OBSERVERS:
            x=delta/scale+float(self.rng.normal(0,self.semantic_cfg.transport_observer_noise))
            if d in compromised and mode==CompromiseMode.SUPPRESS:x=float(self.rng.normal(0,self.semantic_cfg.transport_observer_noise))
            elif d in compromised and mode==CompromiseMode.INJECT:x+=20
            out[d]=x
        return out
    def generate_window(self,window_id:int,*,context:str="normal_indirect",attack_family:AttackFamily|str=AttackFamily.NONE,hidden_calls:int=0,compromised_domains:Iterable[TrustDomain|str]=(),compromise_mode:CompromiseMode|str=CompromiseMode.PASSIVE,adaptive_budget:float=1.0)->SemanticWindow:
        f=AttackFamily(attack_family); mode=CompromiseMode(compromise_mode); comp=frozenset(TrustDomain(d) for d in compromised_domains); control="" if context in ("normal_indirect","direct_allowed") else context
        if f in (AttackFamily.SCP_BYPASS,AttackFamily.NO_TOKEN,AttackFamily.NOTIFY_ABUSE) and hidden_calls>0:
            bw=self.base.attack_window(window_id,hidden=int(hidden_calls),via_scope=f.value,family="M1_additive",budget=float(np.clip(adaptive_budget,0,1)),params=np.asarray([.5,.5]),mode="eval")
        else:bw=self.base.benign_window(window_id,mode="eval",control=control)
        events=[self._event(context) for _ in range(int(max(0,self.rng.poisson(self.semantic_cfg.events_per_window_mean))))]
        if f!=AttackFamily.NONE and hidden_calls>0:events.extend(self._attack_event(f,context) for _ in range(int(hidden_calls)))
        obs=self._observations(events,comp,mode); tr=self._transport(bw,comp,mode); self.t+=self.cfg.window_s
        return SemanticWindow(self.seed,window_id,context,self.communication_mode_for_context(context),events,obs,tr,int(f!=AttackFamily.NONE and hidden_calls>0),f,int(hidden_calls),comp)


def empirical_auc(labels:Sequence[int],scores:Sequence[float])->float:
    y=np.asarray(labels,int); s=np.asarray(scores,float); pos=s[y==1]; neg=s[y==0]
    if not len(pos) or not len(neg):return float("nan")
    return sum(float(np.sum(p>neg))+.5*float(np.sum(p==neg)) for p in pos)/float(len(pos)*len(neg))
