"""V6 diagnostic: defensive SS7 KIE aggregate observation and graph invariants.

No live network operations. No claim that aggregate effects identify attack intent.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from ss7.hierarchical_kie import (
    AggregateWorld, schedule, deliver, CLASSES, LOCAL, BUDGET, K,
    SimConfig, RngBook, KeyCustody, make_episodes, INTERVALS_PER_DAY, MSG_TYPES
)
from ss7.hierarchical_kie_v2 import L1, stack, xmat, group_weights, pooled_threshold
from ss7.hierarchical_kie_v3 import observe_v3,fit_rf,rf_scores,valid,evaluate_budget
from ss7.hierarchical_kie_quick import (
    features_quick,trim,ALLNEW,LNEW,GNEW,KINDS,ALLOC,fit_experts,score_experts,
    zone_features,metadata,role_reference,normalize,causal_scores,cuts_for,margin
)
COMP="slave_compromise"
# Aggregate consistency indicators, not transaction state or authorization.
PAIR=[("SRI_SM","MT_FSM"),("ATI","PSI"),("SRI","PRN"),("IAM","ANM"),("REL","RLC")]
LP=["op_pair_"+a+"_"+b for a,b in PAIR]+["dialogue_deficit","map_dialogue_ratio",
    "external_od_fraction","external_od_dispersion","external_od_to_internal",
    "origin_termination_balance","map_isup_logratio"]
GP=["edge_signed_residual","edge_abs_residual","edge_rms_residual",
    "edge_signed_deficit","edge_positive_fraction","edge_negative_fraction",
    "edge_coverage","edge_volume_log","od_reciprocity_abs"]
P6=LP+GP
L6=list(dict.fromkeys(LNEW+LP+[c+s for c in LP for s in ["_short","_medium","_long","_drift"]]))
G6=list(dict.fromkeys(GNEW+GP+[c+s for c in GP for s in ["_short","_medium","_long","_drift"]]))
ALL6=list(dict.fromkeys(ALLNEW+L6+G6))
ALLOC6=dict(ALLOC)
def normalized_residual(a,b):
    return (a-b)/np.sqrt(a+b+1.)

def observe_v6(nid,received,status,topo):
    d=observe_v3(nid,received,status,topo)
    d.update({c:np.nan for c in P6})
    if nid not in received: return d
    s=received[nid]; cnt=dict(zip(MSG_TYPES,s.mtype))
    for a,b in PAIR:
        d["op_pair_"+a+"_"+b]=normalized_residual(cnt[a],cnt[b])
    m=sum(cnt[x] for x in MSG_TYPES if x not in ("IAM","ACM","ANM","REL","RLC"))
    v=sum(cnt[x] for x in ("IAM","ACM","ANM","REL","RLC"))
    ext=np.array([v for p,v in s.dest.items() if p in topo.nodes and topo.nodes[p].external],float)
    internal=sum(v for p,v in s.dest.items() if p in topo.nodes and not topo.nodes[p].external)
    e=ext.sum()
    d.update(dialogue_deficit=normalized_residual(s.dlg_begin,s.dlg_end),
        map_dialogue_ratio=np.log1p(m)-np.log1p(s.dlg_begin),
        external_od_fraction=e/max(e+internal,1.),
        external_od_dispersion=float(np.sum((ext/max(e,1.))**2)),
        external_od_to_internal=np.log1p(e)-np.log1p(internal),
        origin_termination_balance=normalized_residual(s.originated,s.terminated),
        map_isup_logratio=np.log1p(m)-np.log1p(v))
    residuals=[]; volumes=[]; peers=[]
    for p in topo.graph.neighbors(nid):
        if topo.nodes[p].external: continue
        peers.append(p)
        if p not in received: continue
        o=received[p]; k=tuple(sorted((nid,p)))
        for a,b in [(s.link_out.get(k,0),o.link_in.get(k,0)),
                    (s.link_in.get(k,0),o.link_out.get(k,0))]:
            residuals.append(normalized_residual(a,b)); volumes.append(a+b)
    d["edge_coverage"]=len(residuals)/max(2*len(peers),1)
    if residuals:
        z=np.asarray(residuals); d.update(edge_signed_residual=z.mean(),
            edge_abs_residual=np.abs(z).max(),edge_rms_residual=np.sqrt(np.mean(z*z)),
            edge_signed_deficit=z.min(),edge_positive_fraction=np.mean(z>1),
            edge_negative_fraction=np.mean(z<-1),edge_volume_log=np.log1p(sum(volumes)))
    od=[]
    for p,o in received.items():
        if p!=nid and (p in s.dest or nid in o.dest):
            od.append(abs(normalized_residual(s.dest.get(p,0),o.dest.get(nid,0))))
    if od: d["od_reciprocity_abs"]=np.mean(od)
    return d

def features_v6(d):
    d=features_quick(d)
    d=d.sort_values(["nid","t"]).reset_index(drop=True)
    g=d.groupby("nid",sort=False)[P6]
    pieces=[d]
    for span,minimum,suffix in [(3,2,"_short"),(8,3,"_medium"),(20,6,"_long")]:
        v=g.transform(lambda s:s.rolling(span,min_periods=minimum).mean())
        v.columns=[c+suffix for c in P6]; pieces.append(v)
    ref=g.transform(lambda s:s.shift(9).rolling(64,min_periods=12).median())
    z=d[P6]-ref; z.columns=[c+"_drift" for c in P6]; pieces.append(z)
    return pd.concat(pieces,axis=1)

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
            obs.append(dict(nid=nid,t=t,zone=sim.topo.nodes[nid].zone,role=sim.topo.nodes[nid].ntype.value,**observe_v6(nid,rec,status,sim.topo)))
            e=active.get(nid)
            truth.append(dict(nid=nid,t=t,attack=e["class"] if e else "normal",episode=e["id"] if e else -1))
    frame=features_v6(pd.DataFrame(obs)).merge(pd.DataFrame(truth),on=["nid","t"],validate="one_to_one")
    totals["days"]=cfg.n_intervals/INTERVALS_PER_DAY
    totals["bytes_per_second"]=totals["offered_bytes"]/(cfg.n_intervals*cfg.interval_s)
    totals["application_to_signaling_link_bytes"]=totals["offered_bytes"]/max(totals["signaling_link_bytes"],1)
    frame['world']=seed
    return frame,effects,totals





def fit_regularized(d,cols,y,seed,trees,leaf):
    x=xmat(d,cols); benign=x[d.attack.eq("normal")]
    med=np.zeros(x.shape[1]); available=np.isfinite(benign).any(axis=0)
    med[available]=np.nanmedian(benign[:,available],axis=0)
    xx=np.concatenate([np.where(np.isfinite(x),x,med),~np.isfinite(x)],axis=1)
    model=RandomForestClassifier(n_estimators=trees,min_samples_leaf=leaf,
        max_depth=12 if leaf>=20 else None,max_features="sqrt",n_jobs=2,random_state=seed)
    return model.fit(xx,y,sample_weight=group_weights(d,y)),med

def train_v6(d,seed,trees=70,leaf=20):
    mods={}; roles={}
    for i,k in enumerate(KINDS):
        roles[k]=sorted(d.loc[d.attack.eq(k),"role"].unique())
        g=d[d.role.isin(roles[k])].reset_index(drop=True)
        y=g.attack.eq(k).to_numpy()
        if k==COMP:
            # Separate local HGB expert on augmented local-only features.
            cols=list(dict.fromkeys(L1+LP+[c+s for c in LP for s in ["_short","_medium","_long","_drift"]]))
            model=HistGradientBoostingClassifier(max_iter=70,max_leaf_nodes=12,
                min_samples_leaf=35,l2_regularization=6.,learning_rate=.07,
                random_state=seed+i,early_stopping=False)
            mods[k]=("hgb",model.fit(xmat(g,cols),y,sample_weight=group_weights(g,y)),cols)
        else:
            mods[k]=("rf",fit_regularized(g,L6,y,seed+i,trees,leaf),L6)
    return mods,roles

def graph_only(d):
    # No local detector scores in this representation.
    q=d[G6].copy()
    for c in GP:
        g=d[c].groupby([d.world,d.t,d.zone],sort=False)
        peer=(g.transform("sum")-d[c])/(g.transform("count")-1).replace(0,np.nan)
        q[c+"_zone_centered"]=d[c]-peer
    return q

def score_graph(d,model,cols):
    return rf_scores(graph_only(d),cols,model)[True].where(valid(d),np.nan)

def build_graph(fusion,seed):
    q=graph_only(fusion); cols=list(q.columns)
    return fit_regularized(metadata(q,fusion),cols,fusion.attack.ne("normal").to_numpy(),seed,80,20),cols

def normalized(d,raw,stats):
    return causal_scores(d,normalize(d,raw,stats))

def evaluate_variant(rows,name,va,te,vs,ts,budget,alloc):
    cuts=cuts_for(va,vs,budget,alloc)
    vm,tm=margin(vs,cuts),margin(ts,cuts)
    rows.extend(evaluate_budget(va,vm,0.,name,"validation",budget))
    rows.extend(evaluate_budget(te,tm,0.,name,"test",budget))
    return cuts,tm

def diagnostic(d,s,role_map,stage,tree_count,leaf):
    rows=[]
    for c in KINDS:
        mask=d.attack.isin(["normal",c])&d.role.isin(role_map[c])
        y=d.loc[mask,"attack"].eq(c)
        a=s.loc[mask,c].to_numpy(float)
        ok=np.isfinite(a); a=np.where(ok,a,np.min(a[ok])-1 if ok.any() else -1e12)
        rows.append(dict(stage=stage,trees=tree_count,min_leaf=leaf,channel=c,
            auc=roc_auc_score(y,a) if y.nunique()==2 else np.nan,
            ap=average_precision_score(y,a) if y.nunique()==2 else np.nan))
    return rows

def run(args):
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    kw=dict(days=args.days,repetitions=args.repetitions,loss=.01)
    na,_,_=generate(args.seed+9900,days=.10,repetitions=1,zero=True)
    nb,_,_=generate(args.seed+9900,days=.10,repetitions=1,clean=True)
    np.testing.assert_allclose(na[ALL6],nb[ALL6],atol=1e-10,rtol=0,equal_nan=True)
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
    # Baseline: V4 with expanded calibration, same seed/representation.
    old,oroles=fit_experts(train,args.seed,LNEW)
    ff=score_experts(fusion,old,oroles)
    fz=zone_features(fusion,ff); fcols=list(fz.columns)
    oz=fit_rf(metadata(fz,fusion),fcols,args.seed+19,binary=fusion.attack.ne("normal"),n=80)
    def oldscore(d):
        s=score_experts(d,old,oroles)
        s["zone"]=rf_scores(zone_features(d,s),fcols,oz)[True].where(valid(d),np.nan)
        return s
    ors,ovs,ots=map(oldscore,[ref,val,test])
    os=role_reference(ref,ors); ov=normalized(val,ovs,os); ot=normalized(test,ots,os)
    # Graph-only channel is retained independently, no veto over local decisions.
    graph,gcols=build_graph(fusion,args.seed+19)
    def score(d,m,roles):
        s=score_experts(d,m,roles)
        s["zone"]=score_graph(d,graph,gcols)
        return s
    # Controlled training diagnostics: fixed grid, no test-set selection.
    # Training AUROC is resubstitution, not an out-of-sample estimate.
    candidates={}; curves=[]; val_results=[]
    for trees,leaf in [(30,6),(70,6),(70,20),(100,35)]:
        m,roles=train_v6(train,args.seed,trees,leaf)
        tr,rr,vr=[score(d,m,roles) for d in [train,ref,val]]
        stats=role_reference(ref,rr)
        tr=normalized(train,tr,stats); vs=normalized(val,vr,stats)
        curves+=diagnostic(train,tr,roles,"train",trees,leaf)
        curves+=diagnostic(val,vs,roles,"validation",trees,leaf)
        cuts=cuts_for(val,vs,BUDGET,ALLOC6)
        result=evaluate_budget(val,margin(vs,cuts),0.,"candidate","validation",BUDGET)
        a=next(x for x in result if x["attack"]=="all")
        name=f"rf{trees}_leaf{leaf}"
        val_results.append(dict(candidate=name,**a))
        candidates[name]=(m,roles,stats,vs)
    # Fixed independent budgets, choose only by validation feasibility/recall.
    selected=max(val_results,key=lambda x:(x["false_alerts_per_day"]<=BUDGET+1e-9,
        x["episode_recall"],-x["false_alerts_per_day"],x["ap"]))["candidate"]
    print("SELECTED",selected,flush=True)
    m,roles,stats,vs=candidates[selected]
    ts=normalized(test,score(test,m,roles),stats)
    rows=[]
    for budget in (20.,100.,500.):
        evaluate_variant(rows,"v4_more_calibration",val,test,ov,ot,budget,ALLOC)
        cuts,tm=evaluate_variant(rows,"v6_hierarchy",val,test,vs,ts,budget,ALLOC6)
        evaluate_variant(rows,"v6_local_ablation",val,test,vs,ts,budget,{k:.2 for k in KINDS})
        evaluate_variant(rows,"v6_graph_ablation",val,test,vs,ts,budget,{"zone":1.})
        if budget==BUDGET:
            test[["world","nid","t","role","attack","episode"]].assign(
                score=tm,alarm=tm>0).join(ts.add_prefix("channel_")).to_csv(out/"predictions.csv",index=False)
    res=pd.DataFrame(rows); res.to_csv(out/"metrics.csv",index=False)
    pd.DataFrame(curves).to_csv(out/"training_diagnostics.csv",index=False)
    pd.DataFrame(val_results).to_csv(out/"validation_selection.csv",index=False)
    pd.DataFrame(overhead).to_csv(out/"kie_overhead.csv",index=False)
    (out/"manifest.json").write_text(json.dumps(dict(args=vars(args),plan=plan,
        selected=selected,features_local=L6,features_graph=G6,allocation=ALLOC6,
        confirmation_windows=K,primary_budget=BUDGET,
        scope="pilot; aggregate proxies; simulated KIE; no live validation"),indent=2))
    print("PRIMARY",flush=True)
    print(res[(res.split=="test")&(res.budget==BUDGET)].to_string(index=False),flush=True)
    print("VALIDATION",flush=True)
    print(pd.DataFrame(val_results).to_string(index=False),flush=True)
    print("TRAINING DIAGNOSTICS",flush=True)
    print(pd.DataFrame(curves).to_string(index=False),flush=True)
    print("BUDGET SENSITIVITY",flush=True)
    print(res[(res.split=="test")&(res.attack=="all")].to_string(index=False),flush=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--seed",type=int,required=True)
    p.add_argument("--days",type=float,default=.16)
    p.add_argument("--repetitions",type=int,default=2)
    p.add_argument("--out",required=True)
    a=p.parse_args(); run(a)
if __name__=="__main__": main()
