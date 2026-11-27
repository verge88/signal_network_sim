"""V2: independently calibrated, temporal, hierarchical KIE evidence fusion.
All inputs are simulated aggregate reports. No live network access.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.metrics import roc_auc_score, average_precision_score
from ss7.hierarchical_kie import (
    AggregateWorld, schedule, deliver, observe, add_history, matrix, fit_hierarchy,
    predict, hierarchy, cascade, fit_local, local_scores, CLASSES, LOCAL, OWN_COLS, INV,
    BASE, K, BUDGET, SimConfig, RngBook, KeyCustody, make_episodes, INTERVALS_PER_DAY,
    MSG_TYPES
)

EXTRA_LOCAL = ["log_count_"+m for m in MSG_TYPES]+["log_total","log_dialogues","dialogue_gap",
    "external_link_share","log_originated","log_terminated","log_transited"]
EXTRA_GRAPH = ["relative_link_mean","relative_link_min","relative_link_rms",
    "relative_od_mean","relative_od_min","relative_od_rms","peer_deficit_fraction",
    "peer_signed_median","peer_count"]
TRACK_LOCAL = EXTRA_LOCAL + OWN_COLS
TRACK_GRAPH = EXTRA_GRAPH + INV
TRACK = TRACK_LOCAL + TRACK_GRAPH

def observe_v2(nid,received,status,topo):
    d=observe(nid,received,status,topo)
    d.update({c:np.nan for c in EXTRA_LOCAL+EXTRA_GRAPH})
    if nid not in received: return d
    s=received[nid]
    d.update({"log_count_"+m:np.log1p(v) for m,v in zip(MSG_TYPES,s.mtype)})
    d.update(log_total=np.log1p(s.total()),log_dialogues=np.log1p(s.dlg_begin),
        dialogue_gap=(s.dlg_begin-s.dlg_end)/max(s.dlg_begin,1),
        log_originated=np.log1p(s.originated),log_terminated=np.log1p(s.terminated),
        log_transited=np.log1p(s.transited))
    ext=0; lr=[]; od=[]; peer_res=[]
    for p in topo.graph.neighbors(nid):
        k=tuple(sorted((nid,p)))
        a,b=s.link_out.get(k,0),s.link_in.get(k,0)
        if topo.nodes[p].external: ext+=a+b
        if p not in received: continue
        o=received[p]; c,e=o.link_in.get(k,0),o.link_out.get(k,0)
        v=[(a-c)/(a+c+30.),(b-e)/(b+e+30.)]
        lr.extend(v); peer_res.append(np.mean(v))
    d["external_link_share"]=ext/max(s.total(),1)
    for p,o in received.items():
        if p!=nid and (p in s.dest or nid in o.dest):
            a,b=s.dest.get(p,0),o.dest.get(nid,0)
            od.append((a-b)/(a+b+10.))
    for label,vals in [("link",lr),("od",od)]:
        if vals:
            d["relative_"+label+"_mean"]=np.mean(vals)
            d["relative_"+label+"_min"]=np.min(vals)
            d["relative_"+label+"_rms"]=np.sqrt(np.mean(np.square(vals)))
    if peer_res:
        d.update(peer_deficit_fraction=np.mean(np.array(peer_res)<-.04),
                 peer_signed_median=np.median(peer_res),peer_count=len(peer_res))
    return d

def features(d):
    # Legacy features preserved for paired comparison.
    d=add_history(d)
    g=d.groupby("nid",sort=False)[TRACK]
    # Past-only long reference; the gap avoids immediate adaptation to onset.
    ref=g.transform(lambda s:s.shift(9).rolling(96,min_periods=12).median())
    short=g.transform(lambda s:s.rolling(4,min_periods=2).mean())
    dev=short-ref
    ref.columns=[c+"_ref" for c in TRACK]
    dev.columns=[c+"_relative_change" for c in TRACK]
    # Zone peers of the same role, excluding the subject.
    z=d.groupby(["t","zone","role"],sort=False)[TRACK_LOCAL]
    peers=(z.transform("sum")-d[TRACK_LOCAL])/(z.transform("count")-1).replace(0,np.nan)
    peers=(d[TRACK_LOCAL]-peers)
    peers.columns=[c+"_zone_delta" for c in TRACK_LOCAL]
    return pd.concat([d,ref,dev,peers],axis=1)

L1=TRACK_LOCAL + ["missing","invalid"] + [c+"_relative_change" for c in TRACK_LOCAL]
L2=TRACK_GRAPH + [c+"_relative_change" for c in TRACK_GRAPH]+[c+"_zone_delta" for c in TRACK_LOCAL]
ALL_FEATURES=list(dict.fromkeys(L1+L2+LOCAL))

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
            obs.append(dict(nid=nid,t=t,zone=sim.topo.nodes[nid].zone,role=sim.topo.nodes[nid].ntype.value,**observe_v2(nid,rec,status,sim.topo)))
            e=active.get(nid)
            truth.append(dict(nid=nid,t=t,attack=e["class"] if e else "normal",episode=e["id"] if e else -1))
    frame=features(pd.DataFrame(obs)).merge(pd.DataFrame(truth),on=["nid","t"],validate="one_to_one")
    totals["days"]=cfg.n_intervals/INTERVALS_PER_DAY
    totals["bytes_per_second"]=totals["offered_bytes"]/(cfg.n_intervals*cfg.interval_s)
    totals["application_to_signaling_link_bytes"]=totals["offered_bytes"]/max(totals["signaling_link_bytes"],1)
    frame['world']=seed
    return frame,effects,totals


def stack(frames):
    return pd.concat(frames,ignore_index=True)

def xmat(d,cols):
    return d[cols].replace([np.inf,-np.inf],np.nan).to_numpy(float)

def group_weights(d,y):
    # Each world contributes equal negative weight; each positive episode
    # contributes equal positive weight. No dependence on validation/test.
    y=np.asarray(y,dtype=bool); w=np.zeros(len(d))
    for mask,keys in [(~y,["world"]),(y,["world","episode"])]:
        p=d.loc[mask,keys]
        sizes=p.groupby(keys)[keys[0]].transform("size").to_numpy()
        if len(p)==0: raise ValueError("empty binary class")
        a=1./sizes; w[mask]=a/a.sum()*len(d)/2
    return w

def fit_heads(d,cols,seed):
    x=xmat(d,cols); heads={}
    for i,kind in enumerate(CLASSES[1:]):
        y=d.attack.eq(kind).to_numpy()
        m=HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=15,min_samples_leaf=30,
              l2_regularization=4.,learning_rate=.07,early_stopping=False,random_state=seed+i)
        heads[kind]=m.fit(x,y,sample_weight=group_weights(d,y))
    # Benign-only novelty channel. No known attack labels enter its fit.
    benign=d.attack.eq("normal") & d.missing.eq(0) & d.invalid.eq(0)
    # Median imputation fit only on benign training observations.
    med=np.nanmedian(x[benign],axis=0); med=np.where(np.isfinite(med),med,0.)
    iso=IsolationForest(n_estimators=120,max_samples=2048,random_state=seed,n_jobs=2)
    iso.fit(np.where(np.isfinite(x[benign]),x[benign],med))
    return heads,iso,med

def raw_scores(d,cols,fit):
    heads,iso,med=fit; x=xmat(d,cols)
    out=pd.DataFrame({k:m.decision_function(x) for k,m in heads.items()},index=d.index)
    out["novelty"]=-iso.score_samples(np.where(np.isfinite(x),x,med))
    # Unavailable reports are not affirmative evidence of malicious traffic.
    out.loc[d.missing.eq(1)|d.invalid.eq(1),:]=np.nan
    return out

class TailScale:
    """Role-specific robust tail scaling; no probability interpretation.
    Piecewise-linear extrapolation avoids saturation of empirical percentiles.
    """
    def fit(self,d,raw):
        self.stats={}
        benign=d.attack.eq("normal")
        for role in ["*"]+sorted(d.role.unique()):
            take=benign if role=="*" else benign & d.role.eq(role)
            v=raw.loc[take].to_numpy(float)
            q=np.nanquantile(v,[.5,.95,.995],axis=0)
            q=np.where(np.isfinite(q),q,0.)
            scale=np.maximum(q[2]-q[1],np.maximum((q[1]-q[0])*.1,1e-4))
            self.stats[role]=(q[1],scale)
        return self
    def transform(self,d,raw):
        x=raw.to_numpy(float); out=np.empty_like(x)
        for role in d.role.unique():
            mask=d.role.eq(role).to_numpy()
            center,scale=self.stats.get(role,self.stats["*"])
            out[mask]=np.clip((x[mask]-center)/scale,-12,30)
        return pd.DataFrame(out,index=d.index,columns=raw.columns)

def accumulate(d,z):
    # Causal leaky CUSUM and immediate score. Missing windows decay the state.
    a=z.to_numpy(float); out=np.zeros_like(a)
    for _,idx in d.groupby(["world","nid"],sort=False).indices.items():
        idx=np.asarray(idx); idx=idx[np.argsort(d.t.to_numpy()[idx])]
        state=np.zeros(a.shape[1]); last=None
        for j in idx:
            if last is not None and d.t.iat[j]!=last+1: state[:]=0
            valid=np.isfinite(a[j]); ev=np.where(valid,np.clip(a[j]-.8,-2,5),0.)
            state=np.maximum(0,.92*state+ev)
            out[j]=np.maximum(np.where(valid,a[j],-12.),state/3.)
            # Never confirm a missing report as a positive window by itself.
            out[j,~valid]=-12.
            last=d.t.iat[j]
    return pd.DataFrame(out,index=d.index,columns=z.columns)

def stage_scores(d,raw1,raw2,sc1,sc2):
    a=accumulate(d,sc1.transform(d,raw1))
    b=accumulate(d,sc2.transform(d,raw2))
    # L2 graph evidence is computed from actual neighbor reports. L3 uses
    # bounded additive corroboration, without rejecting weak local candidates.
    both=np.maximum(a,b)+.25*np.maximum(np.minimum(a,b),0.)
    return a,b,both

def pooled_threshold(d,score,budget=BUDGET):
    neg=np.sort(np.asarray(score)[d.attack.eq("normal")])
    days=d[["world","t"]].drop_duplicates().shape[0]/INTERVALS_PER_DAY
    allowed=int(np.floor(budget*days))
    if not len(neg): raise ValueError("no negatives")
    return float(neg[max(0,len(neg)-allowed-1)])

def evaluate(d,score,cut,name,split):
    q=d[["world","nid","t","attack","episode"]].assign(fire=np.asarray(score)>cut)
    pos=q.attack.ne("normal")
    days=q[["world","t"]].drop_duplicates().shape[0]/INTERVALS_PER_DAY
    false_inc=0
    for _,g in q.groupby(["world","nid"]):
        g=g.sort_values("t"); run=0; last=None
        for t,fire,normal in zip(g.t,g.fire,g.attack.eq("normal")):
            run=(run+1 if last is not None and t==last+1 else 1) if fire and normal else 0
            if run==K: false_inc+=1
            last=t
    rows=[]
    for kind in ("all",)+CLASSES[1:]:
        sub=q[pos] if kind=="all" else q[q.attack.eq(kind)]
        hits=[]; delays=[]
        for _,g in sub.groupby(["world","episode","nid"]):
            g=g.sort_values("t"); run=0; last=None; hit=None
            for t,fire in zip(g.t,g.fire):
                run=(run+1 if last is not None and t==last+1 else 1) if fire else 0
                if run>=K: hit=int(t)-int(g.t.iloc[0]); break
                last=t
            hits.append(hit is not None)
            if hit is not None: delays.append(hit*30)
        rows.append(dict(model=name,split=split,attack=kind,episodes=len(hits),
            detected=sum(hits),episode_recall=np.mean(hits) if hits else np.nan,
            window_recall=sub.fire.mean(),window_fpr=q.loc[~pos,"fire"].mean(),
            false_alerts_per_day=q.loc[~pos,"fire"].sum()/days,false_incidents_per_day=false_inc/days,
            median_delay_s=np.median(delays) if delays else np.nan,
            auc=roc_auc_score(pos,score) if kind=="all" else np.nan,
            ap=average_precision_score(pos,score) if kind=="all" else np.nan,threshold=cut))
    return rows

def run(args):
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    kw=dict(days=args.days,repetitions=args.repetitions,loss=args.loss,
            transport_overhead=args.transport_overhead)
    nullkw=dict(kw,days=.12,repetitions=1)
    na,_,_=generate(args.seed+9000,zero=True,**nullkw)
    nb,_,_=generate(args.seed+9000,clean=True,**nullkw)
    np.testing.assert_allclose(na[ALL_FEATURES],nb[ALL_FEATURES],atol=1e-10,rtol=0,equal_nan=True)
    (out/"null_gate.json").write_text(json.dumps(dict(passed=True,rows=len(na),features=ALL_FEATURES),indent=2))
    roles=["train"]*4+["fusion"]*2+["reference"]*2+["validation"]*2+["test"]*2
    data={r:[] for r in set(roles)}; stats=[]
    for i,role in enumerate(roles):
        seed=args.seed+i*1009
        print("generate",role,seed,flush=True)
        d,ep,s=generate(seed,**kw)
        if set(d.attack)!=set(CLASSES): raise AssertionError("incomplete class coverage")
        data[role].append(d); stats.append(dict(role=role,seed=seed,**s))
        pd.DataFrame(ep).to_csv(out/f"episodes_{role}_{seed}.csv",index=False)
    pd.DataFrame(stats).to_csv(out/"kie_overhead.csv",index=False)
    train,fusion,ref,val,test=(stack(data[k]) for k in ["train","fusion","reference","validation","test"])
    full=stack([train,fusion])
    f1=fit_heads(full,L1,args.seed); f2=fit_heads(full,L2,args.seed+19)
    raw=lambda d:(raw_scores(d,L1,f1),raw_scores(d,L2,f2))
    rr1,rr2=raw(ref)
    sc1=TailScale().fit(ref,rr1); sc2=TailScale().fit(ref,rr2)
    v1,v2,vb=stage_scores(val,*raw(val),sc1,sc2)
    t1,t2,tb=stage_scores(test,*raw(test),sc1,sc2)
    # Normalize accumulated evidence against independent reference data once more.
    r1,r2,rb=stage_scores(ref,rr1,rr2,sc1,sc2)
    def calibrated(rv,v,t,cols):
        sc=TailScale().fit(ref,rv[cols])
        return sc.transform(val,v[cols]).max(axis=1).to_numpy(),sc.transform(test,t[cols]).max(axis=1).to_numpy()
    known=list(CLASSES[1:])
    pairs={"specialized_hierarchy":calibrated(rb,vb,tb,known+["novelty"]),
           "known_only":calibrated(rb,vb,tb,known),
           "local_only":calibrated(r1,v1,t1,known+["novelty"]),
           "graph_only":calibrated(r2,v2,t2,known+["novelty"]),
           "novelty_only":calibrated(rb,vb,tb,["novelty"])}
    # Original models evaluated on each world separately to prevent mixing
    # identical node IDs/timestamps across different topologies.
    legacy_local=fit_local(train,args.seed,LOCAL)
    legacy_fusion=HistGradientBoostingClassifier(max_iter=150,max_leaf_nodes=15,l2_regularization=2.,
        random_state=args.seed,early_stopping=False)
    chunks=[]
    for _,g in fusion.groupby("world",sort=False):
        chunks.append(pd.DataFrame(hierarchy(g.reset_index(drop=True),legacy_local,LOCAL),index=g.index))
    xx=pd.concat(chunks).sort_index().to_numpy()
    counts=fusion.attack.value_counts()
    weight=fusion.attack.map(lambda k:len(fusion)/(len(counts)*counts[k])).to_numpy()
    legacy_fusion.fit(xx,fusion.attack,sample_weight=weight)
    old=(legacy_local,legacy_fusion)
    def old_scores(d):
        parts=[]
        for _,g in d.groupby("world",sort=False):
            parts.append(pd.Series(predict(g.reset_index(drop=True),old,LOCAL),index=g.index))
        return pd.concat(parts).sort_index().to_numpy()
    pairs["legacy_hierarchy"]=old_scores(val),old_scores(test)
    pairs["hard_cascade"]=cascade(full,val,test,args.seed,LOCAL)
    flat=fit_local(full,args.seed,LOCAL)
    pairs["flat_rf"]=(1-local_scores(val,flat,LOCAL)["normal"].to_numpy(),
                       1-local_scores(test,flat,LOCAL)["normal"].to_numpy())
    rows=[]
    for name,(sv,st) in pairs.items():
        cut=pooled_threshold(val,sv)
        rows+=evaluate(val,sv,cut,name,"validation")+evaluate(test,st,cut,name,"test")
        test[["world","nid","t","role","attack","episode"]].assign(score=st,alarm=st>cut).to_csv(
            out/f"predictions_{name}.csv",index=False)
    res=pd.DataFrame(rows); res.to_csv(out/"metrics.csv",index=False)
    # Matched intent control: identical observations cannot distinguish legitimate
    # service effects from malicious intent. No separate empirical FPR claim.
    (out/"identifiability.json").write_text(json.dumps(dict(
        statement="External proxy effects do not identify malicious intent; identical benign effects yield identical scores.",
        scope="logical matched observation control, not operator validation"),indent=2))
    (out/"manifest.json").write_text(json.dumps(dict(args=vars(args),roles=roles,
        seeds=[args.seed+i*1009 for i in range(len(roles))],L1=L1,L2=L2,
        budget=BUDGET,confirmation_windows=K,scope="aggregate effect proxies; simulated KIE"),indent=2))
    print(res[res.split.eq("test")].to_string(index=False),flush=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--seed",type=int,required=True)
    p.add_argument("--days",type=float,default=.4)
    p.add_argument("--repetitions",type=int,default=3)
    p.add_argument("--loss",type=float,default=.01)
    p.add_argument("--transport-overhead",type=float,default=0.)
    p.add_argument("--out",required=True)
    args=p.parse_args()
    if args.days<.12 or args.repetitions<1 or not 0<=args.loss<1 or args.transport_overhead<0:
        p.error("invalid parameters")
    run(args)

if __name__=="__main__": main()
