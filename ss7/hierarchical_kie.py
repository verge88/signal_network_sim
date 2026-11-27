"""Isolated aggregate SS7 experiment; no live signaling or protocol exploitation."""
import argparse
import hashlib
import hmac
import json
from dataclasses import replace
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest, HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from ss7.fix import (SimConfig, Ss7Sim, RngBook, Episode, Flow, Service,
                     Sophistication, MASK_FAMILIES, KeyCustody, NodeState,
                     MSG_TYPES, OWN, INTERVALS_PER_DAY, derived, make_episodes)

CLASSES = ("normal","sms_intercept","location_track","signaling_dos","irsf","slave_compromise")
# The four external scenarios represent aggregate traffic effects, not attack intent.
FAMILIES = {"sms_intercept": (Service.SMS_MT,), "location_track": (Service.LOCATION,),
            "signaling_dos": (Service.SMS_MT, Service.ROAM_SRI, Service.LOCATION),
            "irsf": (Service.VOICE_INT,)}
OWN_COLS = list(OWN) + ["mix_"+m for m in MSG_TYPES]
INV = ["link_mean","link_absmax","link_min","od_mean","od_absmax","transit_balance","peer_coverage"]
BASE = OWN_COLS + INV + ["missing","invalid"]
LOCAL = BASE + [c+"_history" for c in BASE] + [c+"_change" for c in BASE]
K, BUDGET = 3, 20.0

def encode(p):
    return json.dumps(p,sort_keys=True,separators=(",",":"),allow_nan=False).encode()

def key_for(seed,nid):
    # Reproducible simulation keys, not production key management.
    return hashlib.sha256(f"simulation:{seed}:{nid}".encode()).digest()

def pack(report,key):
    s=report.state
    p={"schema":1,"nid":int(s.nid),"t":int(s.t),"mtype":[int(v) for v in s.mtype],
       "dest":[[int(k),int(v)] for k,v in sorted(s.dest.items())],
       "out":[[int(k[0]),int(k[1]),int(v)] for k,v in sorted(s.link_out.items())],
       "in":[[int(k[0]),int(k[1]),int(v)] for k,v in sorted(s.link_in.items())],
       "counts":[int(v) for v in (s.dlg_begin,s.dlg_end,s.originated,s.terminated,s.transited)]}
    mac=hmac.new(key,encode(p),hashlib.sha256).hexdigest() if report.integrity_ok else "0"*64
    return encode({"payload":p,"mac":mac})

def unpack(wire,key,nid,t):
    e=json.loads(wire); p=e["payload"]
    if not hmac.compare_digest(e["mac"],hmac.new(key,encode(p),hashlib.sha256).hexdigest()):
        raise ValueError("invalid MAC")
    if (p["schema"],p["nid"],p["t"])!=(1,nid,t):
        raise ValueError("stale or foreign report")
    s=NodeState(nid,t,np.asarray(p["mtype"],dtype=np.int64),dict(p["dest"]),
                {(a,b):v for a,b,v in p["out"]},{(a,b):v for a,b,v in p["in"]},*p["counts"])
    s.check_identities()
    return s

def deliver(reports,seed,t,rng,loss=.01,deadline=.8):
    """Simulated separate KIE poll/response; lost and late reports are unavailable."""
    accepted,status={},{}
    stats=dict(polls=0,reports=0,accepted=0,missing=0,invalid=0,offered_bytes=0,accepted_bytes=0)
    for nid,report in sorted(reports.items()):
        key=key_for(seed,nid); wire=pack(report,key)
        stats["polls"]+=1
        stats["offered_bytes"]+=len(encode({"schema":1,"nid":int(nid),"t":t}))
        poll_lost,response_lost=rng.random(2)<loss
        delay=rng.lognormal(np.log(.18),.65)
        if not poll_lost:
            stats["reports"]+=1; stats["offered_bytes"]+=len(wire)
        if poll_lost or response_lost or delay>deadline:
            status[nid]="missing"; stats["missing"]+=1; continue
        try:
            accepted[nid]=unpack(wire,key,nid,t)
        except (ValueError,AssertionError,KeyError,TypeError):
            status[nid]="invalid"; stats["invalid"]+=1; continue
        status[nid]="accepted"; stats["accepted"]+=1; stats["accepted_bytes"]+=len(wire)
    return accepted,status,stats

def residual(a,b):
    # Canonical link counters are offered counts, not measured delivered packets.
    return (a-b)/np.sqrt(max(a,b,1))

def observe(nid,received,status,topo):
    row={c:np.nan for c in BASE}
    row.update(missing=float(status[nid]=="missing"),invalid=float(status[nid]=="invalid"))
    if nid not in received:
        return row
    s=received[nid]; row.update(derived(s))
    row.update({"mix_"+m:float(v)/max(s.total(),1) for m,v in zip(MSG_TYPES,s.mtype)})
    peers=[p for p in topo.graph.neighbors(nid) if not topo.nodes[p].external]
    res=[]; od=[]
    for p in peers:
        if p not in received: continue
        o=received[p]; k=tuple(sorted((nid,p)))
        res.extend([residual(s.link_out.get(k,0),o.link_in.get(k,0)),
                    residual(s.link_in.get(k,0),o.link_out.get(k,0))])
    for p,o in received.items():
        if p!=nid and (p in s.dest or nid in o.dest):
            od.append(residual(s.dest.get(p,0),o.dest.get(nid,0)))
    if res:
        row.update(link_mean=np.mean(res),link_min=np.min(res),link_absmax=np.max(np.abs(res)))
    if od:
        row.update(od_mean=np.mean(od),od_absmax=np.max(np.abs(od)))
    row["peer_coverage"]=sum(p in received for p in peers)/max(len(peers),1)
    if topo.nodes[nid].ntype.value in ("STP","IGW"):
        row["transit_balance"]=residual(sum(s.link_in.values()),sum(s.link_out.values()))
    return row

class AggregateWorld(Ss7Sim):
    def _build_flows(self,rng):
        flows=super()._build_flows(rng)
        # Background legitimate activity in all service families.
        for ext in self.topo.external:
            for dst in self.topo.hlr:
                flows.append(Flow(ext,dst,Service.LOCATION,float(rng.gamma(1.5,.12))))
            for sp in self.topo.zone_sp.values():
                for dst in sp:
                    flows.append(Flow(ext,dst,Service.VOICE_INT,float(rng.gamma(1.5,.12))))
        return flows

    def aggregate_step(self,t,episodes,effects):
        active=[e for e in effects if e["start"]<=t<=e["end"] and e["class"] in FAMILIES]
        original=self.flows
        if active:
            self.flows=[replace(f,rate=f.rate*float(np.prod([
                e["factor"] for e in active if f.dst==e["nid"] and f.src in self.topo.external
                and f.service in FAMILIES[e["class"]]]))) for f in original]
        try:
            return self.step_true(t,episodes)
        finally:
            self.flows=original

def schedule(sim,cfg,rng,repetitions,zero=False):
    kinds=list(CLASSES[1:])*repetitions; rng.shuffle(kinds)
    edges=np.linspace(cfg.feat_window+10,cfg.n_intervals-2,len(kinds)+1,dtype=int)
    if min(np.diff(edges))<12: raise ValueError("too few windows")
    effects,comp=[],[]
    for i,kind in enumerate(kinds):
        a,b=map(int,edges[i:i+2])
        start=a+int(rng.integers(1,max(2,(b-a)//4)))
        end=min(b-2,start+max(5,int((b-a)*rng.uniform(.30,.60))))
        nodes=(sim.topo.slaves if kind=="slave_compromise" else
               [n for s in sim.topo.zone_sp.values() for n in s] if kind=="irsf" else sim.topo.hlr)
        nid=int(rng.choice(nodes)); factor=float(rng.uniform(1.2,4.0))
        e=dict(id=i,nid=nid,start=start,end=end,factor=1.0 if zero else factor)
        e["class"]=kind
        if kind=="slave_compromise":
            rate=float(rng.choice([.05,.15,.4,1.,2.5])); family=str(rng.choice(MASK_FAMILIES))
            params=rng.random(2); budget=float(rng.uniform(.7,1.0))
            comp.append(Episode("compromise",nid,start,end,hidden_rate=0. if zero else rate,
                       family=family,budget=budget,params=params,sophistication=Sophistication.ADAPTIVE))
            e.update(hidden_rate=0. if zero else rate,family=family)
        effects.append(e)
    return effects,comp

def add_history(df):
    d=df.sort_values(["nid","t"]).reset_index(drop=True)
    past=d.groupby("nid",sort=False)[BASE].transform(lambda s:s.shift(1).rolling(12,min_periods=3).median())
    delta=d[BASE].to_numpy()-past.to_numpy()
    past.columns=[c+"_history" for c in BASE]
    return pd.concat([d,past,pd.DataFrame(delta,columns=[c+"_change" for c in BASE])],axis=1)

def generate(seed,days,repetitions=2,loss=.01,zero=False,clean=False,transport_overhead=0.):
    cfg=SimConfig(seed=seed,days=days,key_custody=KeyCustody.NODE_SOFTWARE)
    book=RngBook(seed); sim=AggregateWorld(cfg,book)
    if transport_overhead:
        sim.transport_overhead_bytes=transport_overhead
        sim.mean_msg_bits=sim._mean_wire_message_bits()
        sim.topo.size_capacities(sim._nominal_offered(),book["topology"],cfg.priors)
    effects,comp=schedule(sim,cfg,np.random.default_rng(seed+19203),repetitions,zero)
    normal=make_episodes(sim,cfg,book["episode"],n_compromise=0,n_ext=0,n_normal=6*repetitions)
    if clean: effects,comp=[],[]
    obs,truth=[],[]
    totals=dict(polls=0,reports=0,accepted=0,missing=0,invalid=0,offered_bytes=0,
                accepted_bytes=0,signaling_link_bytes=0.)
    for t in range(cfg.n_intervals):
        states,hidden,meta=sim.aggregate_step(t,normal+comp,effects)
        reports=sim.step_reports(t,states,hidden,meta,normal+comp)
        rec,status,stats=deliver(reports,seed,t,np.random.default_rng([seed,t,981]),loss)
        for k,v in stats.items(): totals[k]+=v
        totals["signaling_link_bytes"]+=sum(meta["offered"].values())*cfg.interval_s/8
        active={e["nid"]:e for e in effects if e["start"]<=t<=e["end"]}
        for nid in sim.topo.slaves:
            obs.append(dict(nid=nid,t=t,zone=sim.topo.nodes[nid].zone,**observe(nid,rec,status,sim.topo)))
            e=active.get(nid)
            truth.append(dict(nid=nid,t=t,attack=e["class"] if e else "normal",episode=e["id"] if e else -1))
    frame=add_history(pd.DataFrame(obs)).merge(pd.DataFrame(truth),on=["nid","t"],validate="one_to_one")
    totals["days"]=cfg.n_intervals/INTERVALS_PER_DAY
    totals["bytes_per_second"]=totals["offered_bytes"]/(cfg.n_intervals*cfg.interval_s)
    totals["application_to_signaling_link_bytes"]=totals["offered_bytes"]/max(totals["signaling_link_bytes"],1)
    return frame,effects,totals

def matrix(d,cols):
    return d[cols].replace([np.inf,-np.inf],np.nan).fillna(0.).to_numpy(float)

def fit_local(d,seed,cols):
    x=matrix(d,cols)
    rf=RandomForestClassifier(n_estimators=160,min_samples_leaf=8,class_weight="balanced_subsample",
                              n_jobs=2,random_state=seed).fit(x,d.attack)
    iso=IsolationForest(n_estimators=100,random_state=seed,n_jobs=2).fit(x[d.attack.eq("normal")])
    return rf,iso

def local_scores(d,models,cols):
    rf,iso=models; x=matrix(d,cols)
    p=pd.DataFrame(rf.predict_proba(x),columns=rf.classes_,index=d.index).reindex(columns=CLASSES,fill_value=0.)
    p["anomaly"]=-iso.score_samples(x)
    return p

def peer_mean(d,p,keys):
    g=p.groupby([d[k] for k in keys],sort=False)
    return ((g.transform("sum")-p)/(g.transform("count")-1).replace(0,np.nan)).fillna(0.)

def hierarchy(d,models,cols):
    p=local_scores(d,models,cols)
    z=peer_mean(d,p,["t","zone"])
    z.columns=["zone_peer_"+c for c in p]
    zm=pd.concat([d[["t","zone"]],p],axis=1).groupby(["t","zone"]).mean()
    net=zm.groupby(level="t").mean().reindex(d.t).set_axis(d.index)
    net.columns=["network_"+c for c in p]
    return pd.concat([p,z,net],axis=1).to_numpy(float)

def fit_hierarchy(a,b,seed,cols):
    local=fit_local(a,seed,cols); counts=b.attack.value_counts()
    weight=b.attack.map(lambda k:len(b)/(len(counts)*counts[k])).to_numpy()
    fusion=HistGradientBoostingClassifier(max_iter=150,max_leaf_nodes=15,l2_regularization=2.,
                                           random_state=seed,early_stopping=False)
    fusion.fit(hierarchy(b,local,cols),b.attack,sample_weight=weight)
    return local,fusion

def predict(d,models,cols):
    local,fusion=models
    return 1-fusion.predict_proba(hierarchy(d,local,cols))[:,list(fusion.classes_).index("normal")]

def cascade(train,v,t,seed,cols):
    x=matrix(train,cols); benign=train.attack.eq("normal").to_numpy()
    iso=IsolationForest(n_estimators=100,n_jobs=2,random_state=seed).fit(x[benign])
    g=-iso.score_samples(x); cutoff=np.quantile(g[benign],.8); sel=g>cutoff
    rf=RandomForestClassifier(n_estimators=160,min_samples_leaf=8,n_jobs=2,
                              class_weight="balanced_subsample",random_state=seed).fit(x[sel],(~benign)[sel].astype(int))
    if len(rf.classes_)!=2: raise ValueError("cascade training lacks both classes")
    def score(d):
        xx=matrix(d,cols); gg=-iso.score_samples(xx)
        s=.5*np.clip((gg-g.min())/max(cutoff-g.min(),1e-9),0,1); keep=gg>cutoff
        if keep.any(): s[keep]=.5+.5*rf.predict_proba(xx[keep])[:,1]
        return s
    return score(v),score(t)

def threshold(d,score,budget=BUDGET):
    neg=np.sort(score[d.attack.eq("normal")])
    allowed=int(np.floor(budget*d.t.nunique()/INTERVALS_PER_DAY))
    if not len(neg): raise ValueError("no validation negatives")
    return float(neg[max(0,len(neg)-allowed-1)])

def metrics(d,score,cutoff,name,split):
    q=d[["nid","t","attack","episode"]].assign(fire=score>cutoff)
    pos=q.attack.ne("normal"); days=q.t.nunique()/INTERVALS_PER_DAY; rows=[]
    for kind in ("all",)+CLASSES[1:]:
        sub=q[pos] if kind=="all" else q[q.attack.eq(kind)]
        hits,delays=[],[]
        for _,g in sub.groupby(["episode","nid"]):
            g=g.sort_values("t"); run=0; last=None; hit=None
            for t,fire in zip(g.t,g.fire):
                run=(run+1 if last is not None and t==last+1 else 1) if fire else 0
                last=t
                if run>=K: hit=int(t)-int(g.t.iloc[0]); break
            hits.append(hit is not None)
            if hit is not None: delays.append(hit*30)
        rows.append(dict(model=name,split=split,attack=kind,episodes=len(hits),detected=sum(hits),
            episode_recall=np.mean(hits) if hits else np.nan,window_recall=sub.fire.mean(),
            median_delay_s=np.median(delays) if delays else np.nan,
            p90_delay_s=np.percentile(delays,90) if delays else np.nan,
            window_fpr=q.loc[~pos,"fire"].mean(),false_alerts_per_day=q.loc[~pos,"fire"].sum()/days,
            auc=roc_auc_score(pos,score) if kind=="all" else np.nan,
            ap=average_precision_score(pos,score) if kind=="all" else np.nan,threshold=cutoff))
    return rows

def run(args):
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    seeds=[args.seed+i*1009 for i in range(4)]
    kw=dict(days=args.days,repetitions=args.repetitions,loss=args.loss,transport_overhead=args.transport_overhead)
    nullkw=dict(kw,days=min(args.days,.12),repetitions=1)
    na,_,_=generate(args.seed+9091,zero=True,**nullkw)
    nb,_,_=generate(args.seed+9091,clean=True,**nullkw)
    np.testing.assert_allclose(na[LOCAL],nb[LOCAL],rtol=0,atol=1e-10,equal_nan=True)
    (out/"null_gate.json").write_text(json.dumps(dict(passed=True,rows=len(na),features=LOCAL),indent=2))
    frames=[]; overhead=[]
    for role,seed in zip(["train_local","train_fusion","validation","test"],seeds):
        df,ep,stats=generate(seed,**kw)
        if set(df.attack)!=set(CLASSES): raise AssertionError(f"{role}: incomplete class coverage")
        frames.append(df); overhead.append(dict(role=role,seed=seed,**stats))
        pd.DataFrame(ep).to_csv(out/f"episodes_{role}.csv",index=False)
    pd.DataFrame(overhead).to_csv(out/"kie_overhead.csv",index=False)
    a,b,v,t=frames; train=pd.concat([a,b],ignore_index=True); results=[]
    local=fit_local(train,args.seed,LOCAL)
    flat=lambda d:1-local_scores(d,local,LOCAL)["normal"].to_numpy()
    pairs={"flat_rf":(flat(v),flat(t)),"hard_cascade":cascade(train,v,t,args.seed,LOCAL)}
    for name,cols in [("hierarchy",LOCAL),("hierarchy_without_cross_node_invariants",
                   [c for c in LOCAL if not any(c==i or c.startswith(i+"_") for i in INV)])]:
        models=fit_hierarchy(a,b,args.seed,cols)
        pairs[name]=(predict(v,models,cols),predict(t,models,cols))
    for name,(sv,st) in pairs.items():
        cutoff=threshold(v,sv)
        results+=metrics(v,sv,cutoff,name,"validation")+metrics(t,st,cutoff,name,"test")
        t[["nid","t","attack","episode"]].assign(score=st,alarm=st>cutoff).to_csv(out/f"predictions_{name}.csv",index=False)
    result=pd.DataFrame(results); result.to_csv(out/"metrics.csv",index=False)
    (out/"manifest.json").write_text(json.dumps(dict(seeds=seeds,args=vars(args),classes=CLASSES,
        window_s=30,confirmation_windows=K,alert_budget=BUDGET,features=LOCAL,
        scope="aggregate proxies; simulated KIE; no operator validation"),indent=2))
    print(result.loc[result.split.eq("test")].to_string(index=False),flush=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--seed",type=int,default=6101); p.add_argument("--days",type=float,default=.5)
    p.add_argument("--repetitions",type=int,default=2); p.add_argument("--loss",type=float,default=.01)
    p.add_argument("--transport-overhead",type=float,default=0.); p.add_argument("--out",required=True)
    a=p.parse_args()
    if not 0<=a.loss<1 or a.transport_overhead<0 or a.days<=0 or a.repetitions<1: p.error("invalid parameters")
    run(a)

if __name__=="__main__": main()
