"""V3: independent hierarchical KIE decision channels in an isolated simulator.

External classes remain aggregate effects, not operational protocol attacks.
No live traffic, target identification, or claims of malicious-intent attribution.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, IsolationForest
from ss7.hierarchical_kie import (
    AggregateWorld, schedule, deliver, CLASSES, LOCAL, K, BUDGET, SimConfig,
    RngBook, KeyCustody, make_episodes, INTERVALS_PER_DAY, cascade
)
from ss7.hierarchical_kie_v2 import (
    observe_v2, features, ALL_FEATURES, L1, L2, TRACK_LOCAL, TRACK_GRAPH,
    stack, xmat, group_weights, evaluate, pooled_threshold
)

# All fields derive exclusively from authenticated KIE or static topology.
EXTRA = ["external_od_share","external_od_log","external_od_concentration",
         "external_od_distinct","internal_od_log","od_dialogue_ratio"]
V3_EXTRA = EXTRA + [c+"_slow_change" for c in EXTRA+TRACK_LOCAL] + [
    c+"_fast_change" for c in EXTRA+TRACK_LOCAL] + [
    c+"_role_peer_delta" for c in EXTRA+TRACK_LOCAL]
LOCAL3=list(dict.fromkeys(L1+EXTRA+[c for c in V3_EXTRA if c.endswith('_slow_change') or c.endswith('_fast_change')]+[
    c for c in LOCAL if c not in L2 and not any(c.startswith(g) for g in TRACK_GRAPH)]))
GRAPH3=list(dict.fromkeys(L2+[c+"_role_peer_delta" for c in EXTRA+TRACK_LOCAL]))
ALL3=list(dict.fromkeys(ALL_FEATURES+LOCAL3+GRAPH3))
EXTERNAL=list(CLASSES[1:5])
COMPROMISE="slave_compromise"
BUDGETS=(20.,100.,500.,1000.)
ALLOC={"service":.50,"compromise":.40,"corroborated":.10}

def observe_v3(nid,received,status,topo):
    d=observe_v2(nid,received,status,topo)
    d.update({c:np.nan for c in EXTRA})
    if nid not in received: return d
    s=received[nid]
    ext=[v for p,v in s.dest.items() if p in topo.nodes and topo.nodes[p].external]
    internal=[v for p,v in s.dest.items() if p in topo.nodes and not topo.nodes[p].external]
    a,b=sum(ext),sum(internal)
    d.update(external_od_share=a/max(a+b,1),external_od_log=np.log1p(a),
        external_od_concentration=max(ext,default=0)/max(a,1),
        external_od_distinct=sum(v>0 for v in ext),internal_od_log=np.log1p(b),
        od_dialogue_ratio=(a+b)/max(s.dlg_begin,1))
    return d

def features_v3(d):
    d=features(d)
    cols=EXTRA+TRACK_LOCAL
    g=d.groupby("nid",sort=False)[cols]
    fast=g.transform(lambda s:s.rolling(4,min_periods=2).mean())
    slow=g.transform(lambda s:s.rolling(12,min_periods=5).mean())
    ref=g.transform(lambda s:s.shift(13).rolling(96,min_periods=12).median())
    f=fast-ref; f.columns=[c+"_fast_change" for c in cols]
    s=slow-ref; s.columns=[c+"_slow_change" for c in cols]
    # Compare against same-role nodes in other zones as well as own zone.
    peers=d.groupby(["t","role"],sort=False)[cols]
    avg=(peers.transform("sum")-d[cols])/(peers.transform("count")-1).replace(0,np.nan)
    p=d[cols]-avg; p.columns=[c+"_role_peer_delta" for c in cols]
    return pd.concat([d,f,s,p],axis=1)

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
            obs.append(dict(nid=nid,t=t,zone=sim.topo.nodes[nid].zone,role=sim.topo.nodes[nid].ntype.value,**observe_v3(nid,rec,status,sim.topo)))
            e=active.get(nid)
            truth.append(dict(nid=nid,t=t,attack=e["class"] if e else "normal",episode=e["id"] if e else -1))
    frame=features_v3(pd.DataFrame(obs)).merge(pd.DataFrame(truth),on=["nid","t"],validate="one_to_one")
    totals["days"]=cfg.n_intervals/INTERVALS_PER_DAY
    totals["bytes_per_second"]=totals["offered_bytes"]/(cfg.n_intervals*cfg.interval_s)
    totals["application_to_signaling_link_bytes"]=totals["offered_bytes"]/max(totals["signaling_link_bytes"],1)
    frame['world']=seed
    return frame,effects,totals



def fit_rf(d,cols,seed,binary=None,n=140):
    x=xmat(d,cols); benign=x[d.attack.eq("normal")]
    med=np.zeros(x.shape[1],dtype=float)
    observed=np.isfinite(benign).any(axis=0)
    med[observed]=np.nanmedian(benign[:,observed],axis=0)
    # Missingness indicators preserve observation availability explicitly.
    xx=np.concatenate([np.where(np.isfinite(x),x,med),~np.isfinite(x)],axis=1)
    if binary is None:
        y=d.attack.to_numpy()
        w=np.zeros(len(d))
        for kind in np.unique(y):
            take=y==kind
            keys=["world"] if kind=="normal" else ["world","episode"]
            q=d.loc[take,keys]
            size=q.groupby(keys)[keys[0]].transform("size").to_numpy()
            w[take]=(1/size)/(1/size).sum()*len(d)/len(np.unique(y))
    else:
        y=np.asarray(binary,dtype=bool); w=group_weights(d,y)
    rf=RandomForestClassifier(n_estimators=n,min_samples_leaf=6,max_features="sqrt",
        n_jobs=2,random_state=seed)
    rf.fit(xx,y,sample_weight=w)
    return rf,med

def rf_scores(d,cols,model):
    rf,med=model; x=xmat(d,cols)
    x=np.concatenate([np.where(np.isfinite(x),x,med),~np.isfinite(x)],axis=1)
    p=pd.DataFrame(rf.predict_proba(x),columns=rf.classes_,index=d.index)
    return p

def valid(d):
    return d.missing.eq(0)&d.invalid.eq(0)

def role_scores(d,raw):
    return raw.where(valid(d),np.nan,axis=0)

def fit_compromise(d,seed):
    # Preserve the successful V2 local representation.
    y=d.attack.eq(COMPROMISE).to_numpy()
    m=HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=15,min_samples_leaf=30,
        l2_regularization=4.,learning_rate=.07,early_stopping=False,random_state=seed)
    return m.fit(xmat(d,L1),y,sample_weight=group_weights(d,y))

def local_raw(d,rf,comp):
    p=rf_scores(d,LOCAL3,rf).reindex(columns=CLASSES,fill_value=0.)
    raw=pd.DataFrame({"service":p[EXTERNAL].sum(axis=1),
                      "compromise":comp.decision_function(xmat(d,L1)),
                      "rf_compromise":p[COMPROMISE],
                      "any_rf":1-p["normal"]},index=d.index)
    return raw.where(valid(d),np.nan,axis=0),p.where(valid(d),np.nan,axis=0)

def fusion_frame(d,p,raw):
    # The fusion model is trained on disjoint worlds. Node/time/world IDs are
    # grouping metadata, never input features.
    q=d[GRAPH3].copy()
    for c in p:
        q["local_"+c]=p[c]
        gp=p[c].groupby([d.world,d.t,d.zone],sort=False)
        q["zone_peer_"+c]=(gp.transform("sum")-p[c])/(gp.transform("count")-1).replace(0,np.nan)
    q["local_comp_logit"]=raw["compromise"]
    return q

class Scale:
    """Unbounded robust role-specific calibration (no artificial score cap)."""
    def fit(self,d,raw):
        self.params={}
        for role in ["*"]+sorted(d.role.unique()):
            take=d.attack.eq("normal")
            if role!="*": take&=d.role.eq(role)
            a=raw.loc[take].to_numpy(float)
            q=np.nanquantile(a,[.5,.95,.995],axis=0)
            q=np.where(np.isfinite(q),q,0.)
            scale=np.maximum(q[2]-q[1],np.maximum((q[1]-q[0])*.1,1e-5))
            self.params[role]=(q[1],scale)
        return self
    def transform(self,d,raw):
        a=raw.to_numpy(float); out=np.empty_like(a)
        for role in d.role.unique():
            take=d.role.eq(role).to_numpy()
            center,scale=self.params.get(role,self.params["*"])
            out[take]=(a[take]-center)/scale
        return pd.DataFrame(out,index=d.index,columns=raw.columns)

def temporal(d,z):
    # An immediate evidence channel and a causal, exponentially decayed mean.
    # No upper clipping; missing observations cannot themselves raise alarms.
    a=z.to_numpy(float); out=np.full_like(a,-np.inf)
    times=d.t.to_numpy()
    for _,idx in d.groupby(["world","nid"],sort=False).indices.items():
        idx=np.asarray(idx); idx=idx[np.argsort(times[idx])]
        state=np.zeros(a.shape[1]); last=None
        for j in idx:
            if last is not None and times[j]!=last+1: state[:]=0.
            ok=np.isfinite(a[j])
            state=.85*state+.15*np.where(ok,a[j],0.)
            out[j,ok]=np.maximum(a[j,ok],state[ok])
            last=times[j]
    return pd.DataFrame(out,index=d.index,columns=z.columns)

def channel_thresholds(d,score,budget):
    cuts={c:pooled_threshold(d,score[c].to_numpy(),budget*ALLOC[c]) for c in ALLOC}
    if not all(np.isfinite(v) for v in cuts.values()):
        raise ValueError("insufficient valid reference scores")
    return cuts

def union_margin(score,cuts):
    return np.max(np.column_stack([score[c].to_numpy()-v for c,v in cuts.items()]),axis=1)

def finite_score(a):
    # For rank metrics only: missing scores remain strictly below finite scores.
    a=np.asarray(a,dtype=float); ok=np.isfinite(a)
    return np.where(ok,a,np.min(a[ok])-1 if ok.any() else -1e12)

def evaluate_budget(d,score,cut,name,split,budget):
    # Preserve threshold decisions even when all observations are missing.
    a=np.asarray(score,dtype=float); ok=np.isfinite(a)
    floor=min(float(cut),float(np.min(a[ok])) if ok.any() else float(cut))-1
    a=np.where(ok,a,floor)
    rows=evaluate(d,a,cut,name,split)
    for row in rows: row["budget"]=budget
    return rows

def fit_novelty(d,cols,seed):
    x=xmat(d,cols); benign=d.attack.eq("normal")&valid(d)
    med=np.nanmedian(x[benign],axis=0); med=np.where(np.isfinite(med),med,0.)
    iso=IsolationForest(n_estimators=120,max_samples=2048,random_state=seed,n_jobs=2)
    iso.fit(np.where(np.isfinite(x[benign]),x[benign],med))
    return iso,med

def novelty(d,cols,model):
    iso,med=model; x=xmat(d,cols)
    return pd.Series(-iso.score_samples(np.where(np.isfinite(x),x,med)),index=d.index).where(valid(d),np.nan,axis=0)

def run(args):
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    kw=dict(days=args.days,repetitions=args.repetitions,loss=args.loss,
            transport_overhead=args.transport_overhead)
    nullkw=dict(kw,days=.12,repetitions=1)
    na,_,_=generate(args.seed+19000,zero=True,**nullkw)
    nb,_,_=generate(args.seed+19000,clean=True,**nullkw)
    np.testing.assert_allclose(na[ALL3],nb[ALL3],atol=1e-10,rtol=0,equal_nan=True)
    (out/"null_gate.json").write_text(json.dumps(dict(passed=True,rows=len(na),features=ALL3),indent=2))
    roles=["train"]*5+["fusion"]*3+["reference"]*2+["validation"]*3+["test"]*3
    data={r:[] for r in set(roles)}; stats=[]
    for i,role in enumerate(roles):
        seed=args.seed+i*1009
        print("generate",role,seed,flush=True)
        d,ep,s=generate(seed,**kw)
        if set(d.attack)!=set(CLASSES): raise AssertionError("incomplete class coverage")
        data[role].append(d); stats.append(dict(role=role,seed=seed,**s))
        pd.DataFrame(ep).to_csv(out/f"episodes_{role}_{seed}.csv",index=False)
    overhead=pd.DataFrame(stats); overhead.to_csv(out/"kie_overhead.csv",index=False)
    train,fusion,ref,val,test=(stack(data[k]) for k in ["train","fusion","reference","validation","test"])
    full=stack([train,fusion])
    rf=fit_rf(train,LOCAL3,args.seed)
    comp=fit_compromise(train,args.seed)
    fraw,fp=local_raw(fusion,rf,comp)
    ff=fusion_frame(fusion,fp,fraw)
    fcols=list(ff.columns)
    # Copy only fixed observable features and training metadata.
    ff=ff.assign(world=fusion.world,episode=fusion.episode,attack=fusion.attack)
    graph=fit_rf(ff,fcols,args.seed+19,binary=fusion.attack.ne("normal"))
    def raw(d):
        x,p=local_raw(d,rf,comp)
        g=fusion_frame(d,p,x)
        x["corroborated"]=rf_scores(g,fcols,graph)[True].where(valid(d),np.nan,axis=0)
        return x
    rr,rv,rt=raw(ref),raw(val),raw(test)
    sc=Scale().fit(ref,rr)
    zr,zv,zt=[temporal(d,sc.transform(d,s)) for d,s in [(ref,rr),(val,rv),(test,rt)]]
    # Independently calibrate each branch after temporal processing.
    # Exclude unavailable values from reference quantiles.
    sc2=Scale().fit(ref,zr.replace(-np.inf,np.nan))
    sv=sc2.transform(val,zv.replace(-np.inf,np.nan)).fillna(-np.inf)
    st=sc2.transform(test,zt.replace(-np.inf,np.nan)).fillna(-np.inf)
    # Flat RF, trained with the same total training-world budget as the hierarchy.
    flat=fit_rf(full,LOCAL,args.seed+37)
    fv=1-rf_scores(val,LOCAL,flat)["normal"].to_numpy()
    ft=1-rf_scores(test,LOCAL,flat)["normal"].to_numpy()
    cv,ct=cascade(full,val,test,args.seed,LOCAL)
    iso=fit_novelty(full,LOCAL3,args.seed+41)
    nv=novelty(val,LOCAL3,iso).fillna(-np.inf).to_numpy()
    nt=novelty(test,LOCAL3,iso).fillna(-np.inf).to_numpy()
    pairs={"flat_rf":(fv,ft),"hard_cascade":(cv,ct),
           "service_only":(sv.service.to_numpy(),st.service.to_numpy()),
           "compromise_only":(sv.compromise.to_numpy(),st.compromise.to_numpy()),
           "corroborated_only":(sv.corroborated.to_numpy(),st.corroborated.to_numpy()),
           "novelty_only":(nv,nt),
           "local_rf":(sv.any_rf.to_numpy(),st.any_rf.to_numpy())}
    rows=[]; threshold_records=[]
    for budget in BUDGETS:
        cuts=channel_thresholds(val,sv,budget)
        threshold_records.extend(dict(budget=budget,channel=c,threshold=v,allocation=ALLOC[c])
                                 for c,v in cuts.items())
        vm,tm=union_margin(sv,cuts),union_margin(st,cuts)
        rows+=evaluate_budget(val,vm,0.,"hierarchy_v3","validation",budget)
        rows+=evaluate_budget(test,tm,0.,"hierarchy_v3","test",budget)
        for name,(a,b) in pairs.items():
            cut=pooled_threshold(val,a,budget)
            rows+=evaluate_budget(val,a,cut,name,"validation",budget)
            rows+=evaluate_budget(test,b,cut,name,"test",budget)
        if budget==BUDGET:
            test[["world","nid","t","role","attack","episode"]].assign(
                score=tm,alarm=tm>0).join(st.add_prefix("channel_")).to_csv(out/"predictions_v3.csv",index=False)
    # Leave-one-class-out supervised transfer; the omitted class is never used
    # in classifier fitting, threshold estimation uses normal validation only.
    # The benign-only novelty detector is a separate reference, not a proof of
    # detecting unknown real-world attack mechanisms.
    loo=[]
    for i,kind in enumerate(CLASSES[1:]):
        known=full.loc[full.attack.ne(kind)].reset_index(drop=True)
        mod=fit_rf(known,LOCAL3,args.seed+101+i,binary=known.attack.ne("normal"),n=100)
        v=rf_scores(val,LOCAL3,mod)[True].where(valid(val),-np.inf).to_numpy()
        t=rf_scores(test,LOCAL3,mod)[True].where(valid(test),-np.inf).to_numpy()
        cut=pooled_threshold(val,v,BUDGET)
        loo.extend(row for row in evaluate_budget(test,t,cut,"loo_"+kind,"test",BUDGET)
                   if row["attack"]==kind)
    pd.DataFrame(loo).to_csv(out/"leave_one_attack_out.csv",index=False)
    pd.DataFrame(threshold_records).to_csv(out/"channel_thresholds.csv",index=False)
    result=pd.DataFrame(rows); result.to_csv(out/"metrics.csv",index=False)
    (out/"manifest.json").write_text(json.dumps(dict(args=vars(args),roles=roles,
        seeds=[args.seed+i*1009 for i in range(len(roles))],features=ALL3,
        budgets=BUDGETS,allocation=ALLOC,confirmation_windows=K,
        scope="aggregate traffic effects; simulated KIE; no live validation"),indent=2))
    print("PRIMARY",flush=True)
    print(result[(result.split=="test")&(result.budget==BUDGET)].to_string(index=False),flush=True)
    print("BUDGET SENSITIVITY",flush=True)
    print(result[(result.split=="test")&(result.attack=="all")].to_string(index=False),flush=True)
    print("LEAVE ONE ATTACK OUT",flush=True)
    print(pd.DataFrame(loo).to_string(index=False),flush=True)
    print("KIE OVERHEAD",flush=True)
    print(overhead.to_string(index=False),flush=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--seed",type=int,required=True)
    p.add_argument("--days",type=float,default=.35)
    p.add_argument("--repetitions",type=int,default=3)
    p.add_argument("--loss",type=float,default=.01)
    p.add_argument("--transport-overhead",type=float,default=0.)
    p.add_argument("--out",required=True)
    a=p.parse_args()
    if a.days<.12 or a.repetitions<1 or not 0<=a.loss<1 or a.transport_overhead<0:
        p.error("invalid parameters")
    run(a)

if __name__=="__main__": main()
