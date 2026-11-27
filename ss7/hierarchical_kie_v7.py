"""V7 short defensive KIE pilot: graph compromise specialist and temporal evidence.
All traffic and KIE delivery remain simulated; no live protocol activity.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from ss7.hierarchical_kie import CLASSES,BUDGET,K
from ss7.hierarchical_kie_v2 import stack
from ss7.hierarchical_kie_v3 import rf_scores,valid,evaluate_budget
from ss7.hierarchical_kie_quick import (
    trim,KINDS,ALLOC,score_experts,metadata,role_reference,normalize,
    causal_scores,cuts_for,margin
)
from ss7.hierarchical_kie_v6 import (
    generate,ALL6,train_v6,graph_only,fit_regularized,build_graph,score_graph
)
COMP="slave_compromise"
BUDGETS=(20.,100.,500.)
# Fixed design grid; allocations always sum to one. No test tuning.
ALLOCATIONS={
    "equal":dict(ALLOC),
    "local_priority":{**{k:.18 for k in KINDS},"zone":.10},
    "compromise_priority":{**{k:.15 for k in KINDS if k!=COMP},COMP:.25,"zone":.15},
}
def temporal(d,z,mode):
    if mode=="median": return causal_scores(d,z)
    a=z.to_numpy(float); out=np.full_like(a,-np.inf)
    times=d.t.to_numpy()
    for _,idx in d.groupby(["world","nid"],sort=False).indices.items():
        idx=np.asarray(idx); idx=idx[np.argsort(times[idx])]
        state=np.zeros(a.shape[1]); last=None
        for j in idx:
            if last is not None and times[j]!=last+1: state[:]=0.
            ok=np.isfinite(a[j])
            # Leaky one-sided evidence; bounded increments avoid one outlier
            # driving a sustained alarm. Missing reports decay state.
            ev=np.where(ok,np.clip(a[j]-.25,-2.,3.),0.)
            state=np.maximum(0.,.85*state+ev)
            # Keep current-report evidence in the causal sequence; no alarm
            # on a missing or inapplicable report.
            out[j,ok]=state[ok]
            last=times[j]
    return pd.DataFrame(out,index=d.index,columns=z.columns)

def node_adjust(d,z):
    # Small, causal adjustment of positive background drift, based only on
    # earlier observed scores. Bound the adjustment, not the output score.
    out=z.copy()
    for c in z:
        g=z[c].groupby([d.world,d.nid],sort=False)
        past=g.transform(lambda s:s.shift(10).rolling(40,min_periods=12).median())
        out[c]=z[c]-.5*past.clip(lower=0.,upper=2.).fillna(0.)
    return out

def graph_compromise(fusion,seed):
    # Graph-only observations, independent fusion worlds.
    q=graph_only(fusion); cols=list(q.columns)
    return fit_regularized(metadata(q,fusion),cols,
        fusion.attack.eq(COMP).to_numpy(),seed,80,20),cols

def graph_score(d,model,cols):
    return rf_scores(graph_only(d),cols,model)[True].where(valid(d),np.nan)

def scores(d,models,roles,graph,cols):
    p=score_experts(d,models,roles)
    p["zone"]=graph_score(d,graph,cols)
    return p

def prepare(d,s,stats,mode,adapt):
    z=normalize(d,s,stats)
    if adapt: z=node_adjust(d,z)
    return temporal(d,z,mode)

def fit_eval(d,s,budget,alloc,name):
    cuts=cuts_for(d,s,budget,alloc)
    m=margin(s,cuts)
    rows=evaluate_budget(d,m,0.,name,"validation",budget)
    return cuts,rows

def key(rows):
    a=next(x for x in rows if x["attack"]=="all")
    return (a["false_alerts_per_day"]<=BUDGET+1e-9,a["episode_recall"],
            -a["false_alerts_per_day"],a["ap"])

def add_eval(rows,name,val,test,v,t,budget,alloc):
    cuts,rv=fit_eval(val,v,budget,alloc,name)
    m=margin(t,cuts)
    rows.extend(rv)
    rows.extend(evaluate_budget(test,m,0.,name,"test",budget))
    return m,cuts

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
    # Fix local model complexity for speed and a paired comparison.
    models,roles=train_v6(train,args.seed,70,20)
    old,gcols=build_graph(fusion,args.seed+19)
    graph,ccols=graph_compromise(fusion,args.seed+19)
    rr,vr=[scores(d,models,roles,old,gcols) for d in [ref,val]]
    rc,vc=[scores(d,models,roles,graph,ccols) for d in [ref,val]]
    oldstats=role_reference(ref,rr); stats=role_reference(ref,rc)
    baseline=prepare(val,vr,oldstats,"median",False)
    # Choose only among full hierarchies. All variants retain zone channel.
    candidates={}; tuning=[]
    for mode in ("median","cusum"):
        for adapt in (False,True):
            v=prepare(val,vc,stats,mode,adapt)
            for allocation,alloc in ALLOCATIONS.items():
                name=f"{mode}_adapt{int(adapt)}_{allocation}"
                cuts,rows=fit_eval(val,v,BUDGET,alloc,name)
                candidates[name]=(mode,adapt,allocation,v,cuts,rows)
                tuning.extend(rows)
    selected=max(candidates,key=lambda k:key(candidates[k][-1]))
    mode,adapt,allocation,v,_,_=candidates[selected]
    alloc=ALLOCATIONS[allocation]
    print("SELECTED",selected,flush=True)
    # Test scoring begins only after validation selection.
    tr=scores(test,models,roles,old,gcols)
    tc=scores(test,models,roles,graph,ccols)
    bt=prepare(test,tr,oldstats,"median",False)
    ct=prepare(test,tc,stats,mode,adapt)
    # Fixed graph-specialist ablation, isolates graph replacement from temporal
    # processing and allocation.
    cv=prepare(val,vc,stats,"median",False)
    cb=prepare(test,tc,stats,"median",False)
    rows=[]; threshold_records=[]
    for budget in BUDGETS:
        add_eval(rows,"v6_fixed",val,test,baseline,bt,budget,ALLOC)
        add_eval(rows,"v7_graph_specialist",val,test,cv,cb,budget,ALLOC)
        m,cuts=add_eval(rows,"v7_hierarchy",val,test,v,ct,budget,alloc)
        add_eval(rows,"v7_local_ablation",val,test,v,ct,budget,{k:.2 for k in KINDS})
        add_eval(rows,"v7_graph_ablation",val,test,v,ct,budget,{"zone":1.})
        threshold_records.extend(dict(budget=budget,channel=k,threshold=x) for k,x in cuts.items())
        if budget==BUDGET:
            test[["world","nid","t","role","attack","episode"]].assign(
                score=m,alarm=m>0).join(ct.add_prefix("channel_")).to_csv(out/"predictions.csv",index=False)
    res=pd.DataFrame(rows); res.to_csv(out/"metrics.csv",index=False)
    pd.DataFrame(tuning).to_csv(out/"validation_selection.csv",index=False)
    pd.DataFrame(threshold_records).to_csv(out/"thresholds.csv",index=False)
    pd.DataFrame(overhead).to_csv(out/"kie_overhead.csv",index=False)
    (out/"manifest.json").write_text(json.dumps(dict(args=vars(args),plan=plan,
        selected=selected,mode=mode,adapt=adapt,allocation=alloc,
        local_rf_trees=70,local_rf_min_leaf=20,confirmation_windows=K,
        primary_budget=BUDGET,scope="short aggregate-effect pilot; synthetic KIE"),indent=2))
    print("PRIMARY",flush=True)
    print(res[(res.split=="test")&(res.budget==BUDGET)].to_string(index=False),flush=True)
    print("VALIDATION",flush=True)
    print(pd.DataFrame(tuning).query("attack == 'all'").to_string(index=False),flush=True)
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
