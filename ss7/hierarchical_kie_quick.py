"""Short SS7 KIE experiment: class-specific experts, causal baseline and graph fusion.
Isolated aggregate effects only; no live signaling, no new protocol semantics claimed.
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
from ss7.hierarchical_kie_v2 import (
    L1,L2,TRACK_LOCAL,TRACK_GRAPH,stack,xmat,group_weights,pooled_threshold,evaluate
)
from ss7.hierarchical_kie_v3 import (
    observe_v3,features_v3,LOCAL3,GRAPH3,ALL3,EXTERNAL,COMPROMISE,
    fit_rf,rf_scores,fit_compromise,valid,evaluate_budget
)
WARMUP=30
KINDS=list(CLASSES[1:])
# Opcode counts, local aggregate proportions and graph residuals already in KIE.
PROFILE=list(dict.fromkeys(["log_count_"+m for m in MSG_TYPES]+[
    "log_total","log_dialogues","external_od_log","external_od_share",
    "external_od_concentration","external_od_distinct","dialogue_gap",
    "map_share","isup_share","dest_entropy","in_out_ratio"]))
PROFILE_GRAPH=list(dict.fromkeys(["relative_link_mean","relative_link_min",
    "relative_link_rms","relative_od_mean","relative_od_min","relative_od_rms",
    "link_absmax","od_absmax","peer_coverage"]))
P=PROFILE+PROFILE_GRAPH
LNEW=list(dict.fromkeys(LOCAL3+[c+s for c in PROFILE for s in ["_base_z","_base_fast","_base_slow"]]))
GNEW=list(dict.fromkeys(GRAPH3+[c+s for c in PROFILE_GRAPH for s in ["_base_z","_base_fast","_base_slow"]]))
ALLNEW=list(dict.fromkeys(ALL3+LNEW+GNEW))
ALLOC={k:.16 for k in KINDS}
ALLOC["zone"]=.20
BUDGETS=(20.,100.,500.)
def features_quick(d):
    d=features_v3(d)
    d=d.sort_values(["nid","t"]).reset_index(drop=True)
    # First 30 intervals are an explicit commissioning period. Scheduling
    # begins after interval 40 for all worlds. Nothing from later intervals
    # or labels is used to establish this fixed background reference.
    b=d.loc[d.t<WARMUP].groupby("nid")[P]
    med=b.median().reindex(d.nid).set_axis(d.index)
    q25=b.quantile(.25).reindex(d.nid).set_axis(d.index)
    q75=b.quantile(.75).reindex(d.nid).set_axis(d.index)
    spread=(q75-q25).clip(lower=.12)
    z=(d[P]-med)/spread
    # Commissioning windows excluded from training/evaluation below.
    z.loc[d.t<WARMUP,:]=np.nan
    g=z.groupby(d.nid,sort=False)
    fast=g.transform(lambda s:s.rolling(4,min_periods=2).mean())
    slow=g.transform(lambda s:s.rolling(10,min_periods=4).mean())
    z.columns=[c+"_base_z" for c in P]
    fast.columns=[c+"_base_fast" for c in P]
    slow.columns=[c+"_base_slow" for c in P]
    return pd.concat([d,z,fast,slow],axis=1)

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
    frame=features_quick(pd.DataFrame(obs)).merge(pd.DataFrame(truth),on=["nid","t"],validate="one_to_one")
    totals["days"]=cfg.n_intervals/INTERVALS_PER_DAY
    totals["bytes_per_second"]=totals["offered_bytes"]/(cfg.n_intervals*cfg.interval_s)
    totals["application_to_signaling_link_bytes"]=totals["offered_bytes"]/max(totals["signaling_link_bytes"],1)
    frame['world']=seed
    return frame,effects,totals




def trim(d):
    return d.loc[d.t>=WARMUP].reset_index(drop=True)

def fit_experts(train,seed,cols):
    models={}; roles={}
    for i,k in enumerate(KINDS):
        # Role applicability is inferred solely from training episodes.
        roles[k]=sorted(train.loc[train.attack.eq(k),"role"].unique())
        g=train.loc[train.role.isin(roles[k])].reset_index(drop=True)
        if k==COMPROMISE:
            models[k]=("hgb",fit_compromise(g,seed+i),L1)
        else:
            models[k]=("rf",fit_rf(g,cols,seed+i,binary=g.attack.eq(k),n=70),cols)
    return models,roles

def score_experts(d,models,roles):
    s={}
    for k,(typ,m,cols) in models.items():
        p=m.decision_function(xmat(d,cols)) if typ=="hgb" else rf_scores(d,cols,m)[True].to_numpy()
        s[k]=np.where(valid(d)&d.role.isin(roles[k]),p,np.nan)
    return pd.DataFrame(s,index=d.index)

def zone_features(d,s):
    q=d[GNEW].copy()
    for c in s:
        q["local_"+c]=s[c]
        gp=s[c].groupby([d.world,d.t,d.zone],sort=False)
        q["peer_"+c]=(gp.transform("sum")-s[c])/(gp.transform("count")-1).replace(0,np.nan)
    return q

def metadata(q,d):
    return q.assign(world=d.world,episode=d.episode,attack=d.attack)

def role_reference(d,s):
    # Transform scores by the empirical *upper tail* of benign data for each
    # applicable role. Extrapolate beyond observed extremes, without clipping.
    stats={}
    for k in s:
        for role in d.role.unique():
            a=s.loc[d.attack.eq("normal")&d.role.eq(role),k].dropna().to_numpy()
            if len(a)<20: continue
            q=np.quantile(a,[.50,.90,.99])
            stats[k,role]=(q[1],max(q[2]-q[1],.1*(q[1]-q[0]),1e-5))
    return stats

def normalize(d,s,stats):
    x=pd.DataFrame(np.nan,index=d.index,columns=s.columns)
    for (k,role),(center,scale) in stats.items():
        mask=d.role.eq(role)
        x.loc[mask,k]=(s.loc[mask,k]-center)/scale
    return x

def causal_scores(d,s):
    # Three-window median suppresses isolated spikes without using future.
    # Require the current report to be accepted; do not impute positive alarms.
    g=s.groupby([d.world,d.nid],sort=False)
    m=g.transform(lambda v:v.rolling(3,min_periods=2).median())
    return m.where(s.notna()).fillna(-np.inf)

def cuts_for(d,s,budget,alloc):
    out={}
    for k,w in alloc.items():
        cut=pooled_threshold(d,s[k].to_numpy(),budget*w)
        # No available score => disabled expert, never an alarm.
        out[k]=float(cut) if np.isfinite(cut) else float("inf")
    return out

def margin(s,cuts):
    return np.max(np.column_stack([s[k].to_numpy()-v for k,v in cuts.items()]),axis=1)

def run(args):
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    kw=dict(days=args.days,repetitions=args.repetitions,loss=.01)
    na,_,_=generate(args.seed+9900,days=.10,repetitions=1,zero=True)
    nb,_,_=generate(args.seed+9900,days=.10,repetitions=1,clean=True)
    np.testing.assert_allclose(na[ALLNEW],nb[ALLNEW],atol=1e-10,rtol=0,equal_nan=True)
    (out/"null_gate.json").write_text(json.dumps({"passed":True,"rows":len(na)}))
    roles=["train"]*3+["fusion"]+["reference"]+["validation"]+["test"]*2
    data={k:[] for k in set(roles)}; overhead=[]
    for i,role in enumerate(roles):
        seed=args.seed+1009*i
        print("generate",role,seed,flush=True)
        d,ep,s=generate(seed,**kw)
        if min(e["start"] for e in ep)<WARMUP: raise AssertionError("commissioning contamination")
        d=trim(d)
        if set(d.attack)!=set(CLASSES): raise AssertionError("incomplete classes")
        data[role].append(d)
        overhead.append(dict(role=role,seed=seed,**s))
        pd.DataFrame(ep).to_csv(out/f"episodes_{role}_{seed}.csv",index=False)
    train,fusion,ref,val,test=(stack(data[k]) for k in ["train","fusion","reference","validation","test"])
    experts,applies=fit_experts(train,args.seed,LNEW)
    f=score_experts(fusion,experts,applies)
    g=zone_features(fusion,f); cols=list(g.columns)
    zone=fit_rf(metadata(g,fusion),cols,args.seed+19,binary=fusion.attack.ne("normal"),n=80)
    def score(d):
        s=score_experts(d,experts,applies)
        s["zone"]=rf_scores(zone_features(d,s),cols,zone)[True].where(valid(d),np.nan)
        return s
    sr,sv,st=map(score,[ref,val,test])
    stats=role_reference(ref,sr)
    v=causal_scores(val,normalize(val,sv,stats))
    t=causal_scores(test,normalize(test,st,stats))
    # Legacy feature expert ablation, independently refit on same training worlds.
    old,oldroles=fit_experts(train,args.seed,LOCAL3)
    orr,ov,ot=[score_experts(d,old,oldroles) for d in [ref,val,test]]
    os=role_reference(ref,orr)
    ov=causal_scores(val,normalize(val,ov,os)); ot=causal_scores(test,normalize(test,ot,os))
    full=stack([train,fusion])
    flat=fit_rf(full,LOCAL,args.seed+41,n=90)
    fv=1-rf_scores(val,LOCAL,flat)["normal"].to_numpy()
    ft=1-rf_scores(test,LOCAL,flat)["normal"].to_numpy()
    rows=[]; threshold_records=[]
    for budget in BUDGETS:
        for name,a,b,alloc in [
          ("quick_hierarchy",v,t,ALLOC),
          ("quick_local",v,t,{k:.2 for k in KINDS}),
          ("legacy_features_local",ov,ot,{k:.2 for k in KINDS})]:
            cut=cuts_for(val,a,budget,alloc)
            threshold_records.extend(dict(model=name,budget=budget,channel=k,threshold=z) for k,z in cut.items())
            vm,tm=margin(a,cut),margin(b,cut)
            rows+=evaluate_budget(val,vm,0.,name,"validation",budget)
            rows+=evaluate_budget(test,tm,0.,name,"test",budget)
            if budget==BUDGET and name=="quick_hierarchy":
                test[["world","nid","t","role","attack","episode"]].assign(
                    score=tm,alarm=tm>0).join(t.add_prefix("channel_")).to_csv(out/"predictions.csv",index=False)
        cut=pooled_threshold(val,fv,budget)
        rows+=evaluate_budget(val,fv,cut,"flat_rf","validation",budget)
        rows+=evaluate_budget(test,ft,cut,"flat_rf","test",budget)
    # Channel discrimination before global thresholding; not classification accuracy.
    diagnostic=[]
    for k in KINDS:
        mask=test.attack.isin(["normal",k]) & test.role.isin(applies[k])
        y=test.loc[mask,"attack"].eq(k); x=t.loc[mask,k].to_numpy()
        finite=np.isfinite(x); x=np.where(finite,x,np.min(x[finite])-1 if finite.any() else -1e12)
        diagnostic.append(dict(channel=k,roles=",".join(applies[k]),
            auc=roc_auc_score(y,x) if y.nunique()==2 else np.nan,
            ap=average_precision_score(y,x) if y.nunique()==2 else np.nan))
    result=pd.DataFrame(rows); result.to_csv(out/"metrics.csv",index=False)
    pd.DataFrame(diagnostic).to_csv(out/"channel_diagnostics.csv",index=False)
    pd.DataFrame(threshold_records).to_csv(out/"thresholds.csv",index=False)
    pd.DataFrame(overhead).to_csv(out/"kie_overhead.csv",index=False)
    (out/"manifest.json").write_text(json.dumps(dict(args=vars(args),roles=roles,
        seeds=[args.seed+1009*i for i in range(len(roles))],allocation=ALLOC,
        warmup_windows=WARMUP,role_applicability=applies,confirmation_windows=K,
        scope="pilot only; aggregate effects; synthetic KIE; no protocol semantics added"),indent=2))
    print("PRIMARY",flush=True)
    print(result[(result.split=="test")&(result.budget==BUDGET)].to_string(index=False),flush=True)
    print("BUDGET SENSITIVITY",flush=True)
    print(result[(result.split=="test")&(result.attack=="all")].to_string(index=False),flush=True)
    print("CHANNEL DIAGNOSTICS",flush=True)
    print(pd.DataFrame(diagnostic).to_string(index=False),flush=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--seed",type=int,required=True)
    p.add_argument("--days",type=float,default=.16)
    p.add_argument("--repetitions",type=int,default=2)
    p.add_argument("--out",required=True)
    a=p.parse_args()
    run(a)
if __name__=="__main__": main()
