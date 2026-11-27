"""V10: heterogeneous KIE experts, validation-only allocation and temporal diagnostics."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from ss7.hierarchical_kie import K,BUDGET,CLASSES,INTERVALS_PER_DAY
from ss7.hierarchical_kie_v2 import pooled_threshold
from ss7.hierarchical_kie_v3 import evaluate_budget,valid
from ss7.hierarchical_kie_quick import KINDS,score_experts,role_reference,normalize,causal_scores,cuts_for,margin
from ss7.hierarchical_kie_v6 import train_v6
from ss7.hierarchical_kie_v7 import graph_compromise,graph_score,prepare,ALLOCATIONS
from ss7.hierarchical_kie_v9 import read_worlds,profiles,train_local,fit_graph,raw,diagnostic
COMP="slave_compromise"
CHANNELS=KINDS+["zone"]
ALLOC=dict(ALLOCATIONS["local_priority"])

def runs(a,t):
    n=best=0; last=None
    for hit,ti in zip(a,t):
        n=(n+1 if last is not None and ti==last+1 else 1) if hit else 0
        best=max(n,best); last=ti
    return best

def episode_diagnostic(d,raw_score,smoothed,cut,name):
    f=d[["world","nid","t","attack","episode"]].copy()
    f["accepted"]=valid(d)
    f["raw_above"]=np.isfinite(raw_score)&(raw_score>cut)
    f["smooth_above"]=np.isfinite(smoothed)&(smoothed>cut)
    rows=[]
    for (world,ep,nid),g in f[f.attack.ne("normal")].groupby(["world","episode","nid"]):
        g=g.sort_values("t")
        rows.append(dict(channel=name,world=int(world),episode=int(ep),nid=int(nid),
            attack=g.attack.iloc[0],windows=len(g),accepted=int(g.accepted.sum()),
            raw_above=int(g.raw_above.sum()),smooth_above=int(g.smooth_above.sum()),
            longest_raw_run=runs(g.raw_above,g.t),longest_smooth_run=runs(g.smooth_above,g.t)))
    return rows

def select_heads(val,scored):
    # All choices made on validation before test scores are calculated.
    # Per-class episode recall uses the same fixed confirmation as full model.
    choices={}; audit=[]
    for c in KINDS:
        for name,s in scored.items():
            cut=cuts_for(val,s,BUDGET,{c:ALLOC[c]})[c]
            rows=evaluate_budget(val,margin(s,{c:cut}),0.,name,"validation",BUDGET)
            a=next(x for x in rows if x["attack"]==c)
            # tie-break by AP of raw channel on applicable accepted rows
            from sklearn.metrics import average_precision_score
            m=val.attack.isin(["normal",c])&np.isfinite(s[c])
            ap=average_precision_score(val.loc[m,"attack"].eq(c),s.loc[m,c]) if val.loc[m,"attack"].nunique()==2 else 0.
            audit.append(dict(channel=c,model=name,episode_recall=a["episode_recall"],
                false_alerts_per_day=a["false_alerts_per_day"],channel_ap=float(ap)))
        eligible=[x for x in audit if x["channel"]==c]
        choices[c]=max(eligible,key=lambda a:(a["episode_recall"],-a["false_alerts_per_day"],a["channel_ap"]))["model"]
    return choices,audit

def combine(scores,choices,zone_model):
    d=pd.DataFrame({c:scores[choices[c]][c] for c in KINDS})
    d["zone"]=scores[zone_model]["zone"]
    return d

def allocation_grid():
    # fixed finite grid, zone always positive; no test-dependent tuning.
    a={"local_priority":ALLOC,"equal":dict(ALLOCATIONS["equal"]),
       "compromise_priority":dict(ALLOCATIONS["compromise_priority"])}
    for c in KINDS:
        a["prioritize_"+c]={k:(.30 if k==c else .15) for k in KINDS}
        a["prioritize_"+c]["zone"]=.10
    return a

def budget_search(val,s,budget):
    audit=[]; best=None; bestkey=None
    for name,alloc in allocation_grid().items():
        cuts=cuts_for(val,s,budget,alloc)
        a=next(x for x in evaluate_budget(val,margin(s,cuts),0.,name,"validation",budget) if x["attack"]=="all")
        audit.append(dict(allocation=name,**a))
        key=(a["false_alerts_per_day"]<=budget+1e-9,a["episode_recall"],-a["false_alerts_per_day"],a["ap"])
        if bestkey is None or key>bestkey: best=(name,cuts); bestkey=key
    return best,audit

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--seed",type=int,default=216101)
    p.add_argument("--input",default="worlds")
    p.add_argument("--out",required=True)
    args=p.parse_args(); out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    data=read_worlds(args)
    train,fusion,ref,val,test=(data[k] for k in ["train","fusion","reference","validation","test"])
    frames={}
    for k,d in data.items(): frames[k],cols=profiles(d,False)
    tr,fu,re,va,te=(frames[k] for k in ["train","fusion","reference","validation","test"])
    configs={}
    m,roles=train_v6(train,args.seed,70,20)
    g,gc=graph_compromise(fusion,args.seed+19)
    def old(d):
        s=score_experts(d,m,roles); s["zone"]=graph_score(d,g,gc); return s
    configs["v7"]=(old,ref,val,test)
    gm,gcols=fit_graph(fu,False,args.seed+19)
    for kind in ("extra","hgb"):
        lm,lr=train_local(tr,cols,args.seed,kind)
        def scorer(d,lm=lm,lr=lr):
            return raw(d,lm,lr,gm,gcols,False)
        configs[kind]=(scorer,re,va,te)
    # The reference scaler is frozen separately for each classifier/channel.
    refs={}; vr={}; vs={}; stats={}
    for name,(scorer,r,v,t) in configs.items():
        refs[name]=scorer(r); vr[name]=scorer(v)
        stats[name]=role_reference(r,refs[name])
        vs[name]=prepare(v,vr[name],stats[name],"median",False)
    choices,audit=select_heads(val,vs)
    # Zone choice by validation compromise episode recall at equal budget.
    zone_audit=[]
    for name in ("v7","extra"):
        z=vs[name]; cut=cuts_for(val,z,BUDGET,{"zone":ALLOC["zone"]})
        a=next(x for x in evaluate_budget(val,margin(z,cut),0.,name,"validation",BUDGET) if x["attack"]==COMP)
        zone_audit.append(dict(model=name,recall=a["episode_recall"],false_alerts_per_day=a["false_alerts_per_day"],ap=a["ap"]))
    zone_model=max(zone_audit,key=lambda a:(a["recall"],-a["false_alerts_per_day"]))["model"]
    v=combine(vs,choices,zone_model)
    selected,tuning=budget_search(val,v,BUDGET)
    alloc_name,primary=selected
    print("HEADS",json.dumps(choices),flush=True)
    print("ZONE",zone_model,"ALLOCATION",alloc_name,flush=True)
    # All thresholds fixed before obtaining test scores.
    allcuts={}
    for budget in (20.,100.,500.):
        allcuts[budget]=cuts_for(val,v,budget,allocation_grid()[alloc_name])
    test_raw={}; ts={}
    for name,(scorer,r,vv,t) in configs.items():
        test_raw[name]=scorer(t)
        ts[name]=prepare(t,test_raw[name],stats[name],"median",False)
    t=combine(ts,choices,zone_model)
    tz=combine({n:normalize(configs[n][3],test_raw[n],stats[n]) for n in configs},choices,zone_model)
    rows=[]
    for budget in (20.,100.,500.):
        for name in configs:
            c=cuts_for(val,vs[name],budget,ALLOC)
            rows+=evaluate_budget(val,margin(vs[name],c),0.,name,"validation",budget)
            rows+=evaluate_budget(test,margin(ts[name],c),0.,name,"test",budget)
        for name,c in [("v10_fixed",cuts_for(val,v,budget,ALLOC)),("v10_selected",allcuts[budget])]:
            rows+=evaluate_budget(val,margin(v,c),0.,name,"validation",budget)
            rows+=evaluate_budget(test,margin(t,c),0.,name,"test",budget)
        # Reallocate ablations to the same total budget, not used for selection.
        for name,a in [("v10_local",{k:.2 for k in KINDS}),("v10_zone",{"zone":1.})]:
            c=cuts_for(val,v,budget,a)
            rows+=evaluate_budget(test,margin(t,c),0.,name,"test",budget)
    dr=[]
    for c in CHANNELS:
        dr+=episode_diagnostic(test,tz[c].to_numpy(),t[c].to_numpy(),primary[c],c)
    ddf=pd.DataFrame(dr); ddf.to_csv(out/"episode_channel_diagnostics.csv",index=False)
    # Counterfactual diagnostic at unchanged operating threshold: not a new
    # detector, does not alter primary three-window definition or budget.
    q=ddf[ddf.apply(lambda a:a.channel==a.attack or (a.channel=="zone" and a.attack==COMP),axis=1)].copy()
    q["any_raw"]=q.raw_above.gt(0); q["any_smooth"]=q.smooth_above.gt(0)
    q["confirmed"]=q.longest_smooth_run.ge(K)
    summary=q.groupby(["channel","attack"])[["any_raw","any_smooth","confirmed"]].sum().reset_index()
    res=pd.DataFrame(rows); res.to_csv(out/"metrics.csv",index=False)
    pd.DataFrame(audit).to_csv(out/"head_selection.csv",index=False)
    pd.DataFrame(tuning).to_csv(out/"allocation_selection.csv",index=False)
    summary.to_csv(out/"confirmation_diagnostics.csv",index=False)
    (out/"manifest.json").write_text(json.dumps(dict(seed=args.seed,source_run=46037193622,
        heads=choices,zone=zone_model,allocation=alloc_name,primary_budget=BUDGET,
        confirmation_windows=K,scope="reused development group; no new confirmatory evidence"),indent=2))
    print("PRIMARY",flush=True)
    print(res[(res.split=="test")&(res.budget==BUDGET)].to_string(index=False),flush=True)
    print("HEAD SELECTION",flush=True)
    print(pd.DataFrame(audit).to_string(index=False),flush=True)
    print("ALLOCATION SELECTION",flush=True)
    print(pd.DataFrame(tuning).to_string(index=False),flush=True)
    print("CONFIRMATION DIAGNOSTICS",flush=True)
    print(summary.to_string(index=False),flush=True)
    print("BUDGET SENSITIVITY",flush=True)
    print(res[(res.split=="test")&(res.attack=="all")].to_string(index=False),flush=True)

if __name__=="__main__": main()
