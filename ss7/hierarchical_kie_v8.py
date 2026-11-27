"""V8: isolated synthetic transaction-cohort observations, not real protocol traces."""
import argparse, hashlib, hmac, json
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from ss7.fix import Service,TCAP_SERVICES
from ss7.hierarchical_kie import (
    AggregateWorld,schedule,deliver,encode,CLASSES,BUDGET,K,SimConfig,RngBook,
    KeyCustody,make_episodes,INTERVALS_PER_DAY
)
from ss7.hierarchical_kie_v2 import stack,xmat,group_weights,L1
from ss7.hierarchical_kie_v3 import rf_scores,valid,evaluate_budget
from ss7.hierarchical_kie_quick import (
    trim,KINDS,score_experts,metadata,role_reference,cuts_for,margin
)
from ss7.hierarchical_kie_v6 import (
    observe_v6,features_v6,ALL6,L6,G6,train_v6,fit_regularized,graph_only
)
from ss7.hierarchical_kie_v7 import graph_compromise,graph_score,prepare,ALLOCATIONS
COMP="slave_compromise"
FAMILIES=[(s,e) for s in Service for e in (False,True)]
FIDX={k:i for i,k in enumerate(FAMILIES)}
DIM=len(FAMILIES)
OBS_LOCAL=[f"mon_{s.value}_{'ext' if ext else 'int'}_{k}"
           for s,ext in FAMILIES for k in ("begin","end","expire","open")]
OBS_GRAPH=["mon_endpoint_residual","mon_dialogue_residual","mon_transit_residual",
           "mon_peer_residual","mon_peer_coverage","mon_missing","mon_invalid"]
L8=L6+OBS_LOCAL+[c+"_mean4" for c in OBS_LOCAL]+["mon_missing","mon_invalid"]
G8=G6+OBS_GRAPH+[c+"_mean4" for c in OBS_GRAPH[:5]]
ALL8=list(dict.fromkeys(ALL6+L8+G8))
ALLOC8=dict(ALLOCATIONS["local_priority"])

def observer_key(seed,nid):
    return hashlib.sha256(f"independent-observer:{seed}:{nid}".encode()).digest()

def pack_observer(seed,nid,t,a,transit):
    p=dict(schema=2,nid=int(nid),t=int(t),counts=np.asarray(a,dtype=int).tolist(),transit=int(transit))
    return encode(dict(payload=p,mac=hmac.new(observer_key(seed,nid),encode(p),hashlib.sha256).hexdigest()))

def unpack_observer(seed,nid,t,wire):
    e=json.loads(wire); p=e["payload"]
    mac=hmac.new(observer_key(seed,nid),encode(p),hashlib.sha256).hexdigest()
    if not hmac.compare_digest(e["mac"],mac): raise ValueError("observer MAC")
    if (p["schema"],p["nid"],p["t"])!=(2,nid,t): raise ValueError("observer freshness")
    a=np.asarray(p["counts"])
    if a.shape!=(DIM,4) or a.dtype.kind not in "iu" or (a<0).any(): raise ValueError("observer counts")
    tr=p["transit"]
    if not isinstance(tr,int) or tr<0: raise ValueError("observer transit")
    return a,tr

class EventObserver:
    """Independent synthetic cohort tap. Capture/delay/expiry are assumptions.
    Receives no attack labels, hidden state, or attack parameters.
    """
    def __init__(self,topo,seed,capture=.98):
        self.topo=topo; self.seed=seed; self.capture=capture
        self.nodes=[n for n in topo.nodes if not topo.nodes[n].external]
        self.open={n:np.zeros(DIM,dtype=np.int64) for n in self.nodes}
        self.queue=defaultdict(list)
    def begin(self,t):
        self.t=t; self.rng=np.random.default_rng([self.seed,t,8071])
        self.rows={n:np.zeros((DIM,4),dtype=np.int64) for n in self.nodes}
        self.transit=dict.fromkeys(self.nodes,0)
        for n,i,count,kind in self.queue.pop(t,[]):
            self.rows[n][i,kind]+=count; self.open[n][i]-=count
    def on_batch(self,path,service,count,closed):
        if len(path)<2: return
        for n,other in [(path[0],path[-1]),(path[-1],path[0])]:
            if n not in self.rows: continue
            i=FIDX[service,bool(self.topo.nodes[other].external)]
            ok=int(self.rng.binomial(closed,self.capture))
            fail=int(self.rng.binomial(count-closed,self.capture))
            self.rows[n][i,0]+=ok+fail; self.open[n][i]+=ok+fail
            delays=self.rng.multinomial(ok,[.80,.15,.05])
            self.rows[n][i,1]+=int(delays[0]); self.open[n][i]-=int(delays[0])
            for dt,c in enumerate(delays[1:],1):
                if c: self.queue[self.t+dt].append((n,i,int(c),1))
            if fail: self.queue[self.t+3].append((n,i,fail,2))
        for n in path[1:-1]:
            if n in self.transit: self.transit[n]+=int(self.rng.binomial(count,self.capture))
    def deliver(self,t,loss):
        rng=np.random.default_rng([self.seed,t,8072])
        received,status={},{}
        stats=dict(observer_offered_bytes=0,observer_accepted_bytes=0,observer_polls=0,
                   observer_accepted=0,observer_missing=0,observer_invalid=0)
        for n in sorted(self.nodes):
            a=self.rows[n].copy(); a[:,3]=self.open[n]
            if (a<0).any(): raise AssertionError("negative cohort count")
            wire=pack_observer(self.seed,n,t,a,self.transit[n])
            stats["observer_polls"]+=1
            stats["observer_offered_bytes"]+=len(encode(dict(schema=2,nid=int(n),t=t)))
            lost_poll,lost_reply=rng.random(2)<loss
            delay=rng.lognormal(np.log(.18),.65)
            if not lost_poll: stats["observer_offered_bytes"]+=len(wire)
            if lost_poll or lost_reply or delay>.8:
                status[n]="missing"; stats["observer_missing"]+=1; continue
            try: received[n]=unpack_observer(self.seed,n,t,wire)
            except (ValueError,TypeError,KeyError):
                status[n]="invalid"; stats["observer_invalid"]+=1; continue
            status[n]="accepted"; stats["observer_accepted"]+=1
            stats["observer_accepted_bytes"]+=len(wire)
        return received,status,stats

def res(a,b):
    return (a-b)/np.sqrt(a+b+1.)

def observe_v8(nid,received,status,topo,mon,ms):
    d=observe_v6(nid,received,status,topo)
    d.update({c:np.nan for c in OBS_LOCAL+OBS_GRAPH})
    d.update(mon_missing=float(ms[nid]=="missing"),mon_invalid=float(ms[nid]=="invalid"))
    if nid not in mon: return d
    a,transit=mon[nid]
    d.update(dict(zip(OBS_LOCAL,np.log1p(a.ravel()))))
    if nid in received:
        s=received[nid]
        d["mon_endpoint_residual"]=res(a[:,0].sum(),s.originated+s.terminated)
        d["mon_dialogue_residual"]=res(sum(a[i,0] for i,(svc,_) in enumerate(FAMILIES)
                                           if svc in TCAP_SERVICES),s.dlg_begin)
        d["mon_transit_residual"]=res(transit,s.transited)
    peers=[p for p in topo.graph.neighbors(nid) if not topo.nodes[p].external]
    values=[]
    for p in peers:
        if p in received and p in mon:
            o=received[p]; b,tr=mon[p]
            values.append(abs(res(b[:,0].sum()+tr,o.originated+o.terminated+o.transited)))
    d["mon_peer_coverage"]=len(values)/max(len(peers),1)
    if values: d["mon_peer_residual"]=float(np.mean(values))
    return d

def features_v8(d):
    d=features_v6(d)
    cols=OBS_LOCAL+OBS_GRAPH[:5]
    q=d.groupby("nid",sort=False)[cols].transform(lambda s:s.rolling(4,min_periods=2).mean())
    q.columns=[c+"_mean4" for c in cols]
    return pd.concat([d,q],axis=1)

def generate(seed,days,repetitions=2,loss=.01,zero=False,clean=False,transport_overhead=0.):
    cfg=SimConfig(seed=seed,days=days,key_custody=KeyCustody.NODE_SOFTWARE)
    book=RngBook(seed); sim=AggregateWorld(cfg,book)
    observer=EventObserver(sim.topo,seed); sim._kie_observer=observer
    if transport_overhead:
        sim.transport_overhead_bytes=transport_overhead
        sim.mean_msg_bits=sim._mean_wire_message_bits()
        sim.topo.size_capacities(sim._nominal_offered(),book["topology"],cfg.priors)
    effects,comp=schedule(sim,cfg,np.random.default_rng(seed+19203),repetitions,zero)
    normal=make_episodes(sim,cfg,book["episode"],n_compromise=0,n_ext=0,n_normal=6*repetitions)
    if clean: effects,comp=[],[]
    obs,truth=[],[]
    totals=dict(polls=0,reports=0,accepted=0,missing=0,invalid=0,offered_bytes=0,
                accepted_bytes=0,signaling_link_bytes=0.,observer_offered_bytes=0,observer_accepted_bytes=0,
                observer_polls=0,observer_accepted=0,observer_missing=0,observer_invalid=0)
    for t in range(cfg.n_intervals):
        observer.begin(t)
        states,hidden,meta=sim.aggregate_step(t,normal+comp,effects)
        reports=sim.step_reports(t,states,hidden,meta,normal+comp)
        rec,status,stats=deliver(reports,seed,t,np.random.default_rng([seed,t,981]),loss)
        mon,ms,extra=observer.deliver(t,loss)
        for k,v in extra.items(): totals[k]+=v
        for k,v in stats.items(): totals[k]+=v
        totals["signaling_link_bytes"]+=sum(meta["offered"].values())*cfg.interval_s/8
        active={e["nid"]:e for e in effects if e["start"]<=t<=e["end"]}
        for nid in sim.topo.slaves:
            obs.append(dict(nid=nid,t=t,zone=sim.topo.nodes[nid].zone,role=sim.topo.nodes[nid].ntype.value,**observe_v8(nid,rec,status,sim.topo,mon,ms)))
            e=active.get(nid)
            truth.append(dict(nid=nid,t=t,attack=e["class"] if e else "normal",episode=e["id"] if e else -1))
    frame=features_v8(pd.DataFrame(obs)).merge(pd.DataFrame(truth),on=["nid","t"],validate="one_to_one")
    totals["days"]=cfg.n_intervals/INTERVALS_PER_DAY
    totals["bytes_per_second"]=totals["offered_bytes"]/(cfg.n_intervals*cfg.interval_s)
    totals["application_to_signaling_link_bytes"]=totals["offered_bytes"]/max(totals["signaling_link_bytes"],1)
    frame['world']=seed
    return frame,effects,totals






def train_new(d,seed):
    models={}; roles={}
    for i,k in enumerate(KINDS):
        roles[k]=sorted(d.loc[d.attack.eq(k),"role"].unique())
        g=d[d.role.isin(roles[k])].reset_index(drop=True)
        y=g.attack.eq(k).to_numpy()
        if k==COMP:
            cols=list(dict.fromkeys(L1+OBS_LOCAL+OBS_GRAPH[:3]+[c+"_mean4" for c in OBS_GRAPH[:3]]))
            m=HistGradientBoostingClassifier(max_iter=70,max_leaf_nodes=12,
                min_samples_leaf=35,l2_regularization=6.,learning_rate=.07,
                random_state=seed+i,early_stopping=False)
            models[k]=("hgb",m.fit(xmat(g,cols),y,sample_weight=group_weights(g,y)),cols)
        else:
            models[k]=("rf",fit_regularized(g,L8,y,seed+i,70,20),L8)
    return models,roles

def graph_new(d):
    q=graph_only(d)
    return pd.concat([q,d[OBS_GRAPH+[c+"_mean4" for c in OBS_GRAPH[:5]]]],axis=1)

def graph_fit(d,seed):
    q=graph_new(d); cols=list(q.columns)
    return fit_regularized(metadata(q,d),cols,d.attack.eq(COMP).to_numpy(),seed,80,20),cols

def run(args):
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    kw=dict(days=args.days,repetitions=args.repetitions,loss=.01)
    na,_,_=generate(args.seed+9900,days=.10,repetitions=1,zero=True)
    nb,_,_=generate(args.seed+9900,days=.10,repetitions=1,clean=True)
    np.testing.assert_allclose(na[ALL8],nb[ALL8],atol=1e-10,rtol=0,equal_nan=True)
    (out/"null_gate.json").write_text(json.dumps({"passed":True,"rows":len(na)}))
    plan=[("train",0),("train",1),("train",2),("fusion",3),
          ("reference",4),("validation",5),("test",6),("test",7),
          ("reference",8),("validation",9),("validation",10),("validation",11)]
    data={k:[] for k,_ in plan}; overhead=[]
    for role,i in plan:
        seed=args.seed+1009*i
        print("generate",role,seed,flush=True)
        d,ep,s=generate(seed,**kw); d=trim(d)
        if set(d.attack)!=set(CLASSES): raise AssertionError("incomplete classes")
        data[role].append(d)
        overhead.append(dict(role=role,seed=seed,**s))
        pd.DataFrame(ep).to_csv(out/f"episodes_{role}_{seed}.csv",index=False)
    train,fusion,ref,val,test=(stack(data[k]) for k in ["train","fusion","reference","validation","test"])
    old,oroles=train_v6(train,args.seed,70,20)
    og,oc=graph_compromise(fusion,args.seed+19)
    new,nroles=train_new(train,args.seed)
    ng,nc=graph_fit(fusion,args.seed+19)
    def raw(d,enhanced):
        if enhanced:
            s=score_experts(d,new,nroles)
            s["zone"]=rf_scores(graph_new(d),nc,ng)[True].where(valid(d),np.nan)
        else:
            s=score_experts(d,old,oroles)
            s["zone"]=graph_score(d,og,oc)
        return s
    rows=[]
    for enhanced,name in [(False,"v7_fixed"),(True,"v8_hierarchy")]:
        rr,rv,rt=[raw(d,enhanced) for d in [ref,val,test]]
        stats=role_reference(ref,rr)
        v=prepare(val,rv,stats,"median",False)
        t=prepare(test,rt,stats,"median",False)
        variants=[(name,ALLOC8)]
        if enhanced:
            variants += [("v8_local_ablation",{k:.2 for k in KINDS}),
                         ("v8_graph_ablation",{"zone":1.})]
        for budget in (20.,100.,500.):
            for variant,alloc in variants:
                cuts=cuts_for(val,v,budget,alloc)
                vm,tm=margin(v,cuts),margin(t,cuts)
                rows.extend(evaluate_budget(val,vm,0.,variant,"validation",budget))
                rows.extend(evaluate_budget(test,tm,0.,variant,"test",budget))
                if variant==name and budget==BUDGET:
                    test[["world","nid","t","role","attack","episode"]].assign(
                        score=tm,alarm=tm>0).join(t.add_prefix("channel_")).to_csv(out/f"predictions_{name}.csv",index=False)
    resdf=pd.DataFrame(rows); resdf.to_csv(out/"metrics.csv",index=False)
    over=pd.DataFrame(overhead); over.to_csv(out/"kie_overhead.csv",index=False)
    (out/"manifest.json").write_text(json.dumps(dict(args=vars(args),plan=plan,
        allocation=ALLOC8,confirmation_windows=K,primary_budget=BUDGET,
        observer_capture=.98,observer_timeout_windows=3,observer_delay_prob=[.8,.15,.05],
        scope="synthetic event cohorts; independent trusted observer assumed; no real trace or intent validation"),indent=2))
    print("PRIMARY",flush=True)
    print(resdf[(resdf.split=="test")&(resdf.budget==BUDGET)].to_string(index=False),flush=True)
    print("BUDGET SENSITIVITY",flush=True)
    print(resdf[(resdf.split=="test")&(resdf.attack=="all")].to_string(index=False),flush=True)
    print("KIE OVERHEAD",flush=True)
    print(over.to_string(index=False),flush=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--seed",type=int,required=True)
    p.add_argument("--days",type=float,default=.16)
    p.add_argument("--repetitions",type=int,default=2)
    p.add_argument("--out",required=True)
    a=p.parse_args(); run(a)
if __name__=="__main__": main()
