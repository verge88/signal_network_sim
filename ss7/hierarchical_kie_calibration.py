"""V5 short calibration pilot; defensive aggregate SS7 KIE simulation only."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from ss7.hierarchical_kie import CLASSES, LOCAL, BUDGET, K, INTERVALS_PER_DAY
from ss7.hierarchical_kie_v2 import stack, pooled_threshold
from ss7.hierarchical_kie_v3 import fit_rf,rf_scores,valid,evaluate_budget
from ss7.hierarchical_kie_quick import (
    generate,trim,ALLNEW,LNEW,GNEW,KINDS,ALLOC,fit_experts,score_experts,
    zone_features,metadata,role_reference,normalize,causal_scores,cuts_for,margin
)

BUDGETS=(20.,100.,500.)
def robust_quantiles(a):
    a=np.asarray(a,dtype=float); a=a[np.isfinite(a)]
    if len(a)<20: return None
    # Positive monotone tail normalization. Distinct knots prevent division by zero.
    q=np.quantile(a,[.50,.90,.97,.99,.999])
    span=max(q[-1]-q[1],.1*(q[1]-q[0]),1e-5)
    return q,span

def tail_fit(d,s):
    stats={}
    for c in s:
        for role in d.role.unique():
            a=s.loc[d.attack.eq("normal")&d.role.eq(role),c]
            x=robust_quantiles(a)
            if x is not None: stats[c,role]=x
    return stats

def tail_transform(d,s,stats):
    x=pd.DataFrame(np.nan,index=d.index,columns=s.columns)
    # Continuous upper-tail extrapolation; no artificial upper ceiling.
    for (c,role),(q,span) in stats.items():
        mask=d.role.eq(role)
        a=s.loc[mask,c].to_numpy(float)
        # Standardize relative to high benign quantile; score 0 at q99.
        x.loc[mask,c]=(a-q[3])/span
    return x

def node_adjust(d,s,strength=.5):
    """Lagged, robust node reference; no current/future score or label."""
    out=s.copy()
    for c in s:
        g=s[c].groupby([d.world,d.nid],sort=False)
        past=g.transform(lambda a:a.shift(8).rolling(48,min_periods=12).median())
        # Remove only positive background drift, partially, to reduce masking.
        out[c]=s[c]-strength*past.clip(lower=0).fillna(0.)
    return out

def smooth(d,s,mode):
    if mode=="median":
        return causal_scores(d,s)
    if mode!="ewma": raise ValueError(mode)
    g=s.groupby([d.world,d.nid],sort=False)
    m=g.transform(lambda a:a.ewm(span=5,adjust=False,min_periods=2,ignore_na=True).mean())
    return m.where(s.notna()).fillna(-np.inf)

def joint(s,columns):
    return s[columns].max(axis=1).fillna(-np.inf).to_numpy()

def safe_threshold(d,score,budget):
    cut=pooled_threshold(d,score,budget)
    if not np.isfinite(cut): raise ValueError("insufficient finite validation scores")
    return cut

def worst_world_threshold(d,score,budget):
    vals=[]
    for _,g in d.groupby("world",sort=False):
        vals.append(safe_threshold(g,np.asarray(score)[g.index],budget))
    return max(vals)

def val_key(rows):
    a=next(x for x in rows if x["attack"]=="all")
    # Prefer feasible validation performance; then recall, then fewer alerts.
    return (a["false_alerts_per_day"]<=BUDGET+1e-9,a["episode_recall"],
            -a["false_alerts_per_day"],a["ap"])

def add_eval(rows,d,s,cut,name,split,budget):
    rows.extend(evaluate_budget(d,s,cut,name,split,budget))

def run(args):
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    kw=dict(days=args.days,repetitions=args.repetitions,loss=.01)
    na,_,_=generate(args.seed+9900,days=.10,repetitions=1,zero=True)
    nb,_,_=generate(args.seed+9900,days=.10,repetitions=1,clean=True)
    np.testing.assert_allclose(na[ALLNEW],nb[ALLNEW],atol=1e-10,rtol=0,equal_nan=True)
    (out/"null_gate.json").write_text(json.dumps({"passed":True,"rows":len(na)}))
    # Preserve the V4 world assignments. Additional reference/validation worlds
    # have distinct IDs. On the fresh group even the original V4 seeds are unseen.
    plan=[("train",0),("train",1),("train",2),("fusion",3),
          ("reference",4),("validation",5),("test",6),("test",7),
          ("reference_extra",8),("validation_extra",9),("validation_extra",10),
          ("validation_extra",11)]
    data={k:[] for k,_ in plan}; overhead=[]
    for role,i in plan:
        seed=args.seed+1009*i
        print("generate",role,seed,flush=True)
        d,ep,s=generate(seed,**kw); d=trim(d)
        if set(d.attack)!=set(CLASSES): raise AssertionError("incomplete classes")
        data[role].append(d)
        overhead.append(dict(role=role,seed=seed,**s))
        pd.DataFrame(ep).to_csv(out/f"episodes_{role}_{seed}.csv",index=False)
    train,fusion,ref0,val0,test=(stack(data[k]) for k in ["train","fusion","reference","validation","test"])
    ref=stack([ref0]+data["reference_extra"]); val=stack([val0]+data["validation_extra"])
    experts,applies=fit_experts(train,args.seed,LNEW)
    f=score_experts(fusion,experts,applies)
    g=zone_features(fusion,f); cols=list(g.columns)
    zone=fit_rf(metadata(g,fusion),cols,args.seed+19,binary=fusion.attack.ne("normal"),n=80)
    def score(d):
        s=score_experts(d,experts,applies)
        s["zone"]=rf_scores(zone_features(d,s),cols,zone)[True].where(valid(d),np.nan)
        return s
    sr0,sv0,sr,sv=map(score,[ref0,val0,ref,val])
    # Choose the variant with validation data only, before scoring test.
    candidates={}
    stats=tail_fit(ref,sr)
    for mode in ("median","ewma"):
        for strength in (0.,.5):
            for include_zone in (True,False):
                name=f"joint_{mode}_drift{int(strength*100)}_"+("zone" if include_zone else "local")
                channels=KINDS+["zone"] if include_zone else KINDS
                z=tail_transform(val,sv,stats)
                if strength: z=node_adjust(val,z,strength)
                a=joint(smooth(val,z,mode),channels)
                cut=safe_threshold(val,a,BUDGET)
                rs=evaluate_budget(val,a,cut,name,"validation",BUDGET)
                candidates[name]=(mode,strength,channels,cut,a,rs)
    selected=max(candidates,key=lambda k:val_key(candidates[k][-1]))
    print("SELECTED ON VALIDATION",selected,flush=True)
    # Only now calculate test scores; no test-dependent model selection.
    st=score(test)
    rows=[]; tuning=[]
    for name,(mode,strength,channels,cut,a,rs) in candidates.items():
        tuning.extend(rs)
    pd.DataFrame(tuning).to_csv(out/"validation_selection.csv",index=False)
    mode,strength,channels,_,a,_=candidates[selected]
    z=tail_transform(test,st,stats)
    if strength: z=node_adjust(test,z,strength)
    b=joint(smooth(test,z,mode),channels)
    # Exact V4 baseline with its original single reference and validation worlds.
    old=role_reference(ref0,sr0)
    ov=causal_scores(val0,normalize(val0,sv0,old))
    ot=causal_scores(test,normalize(test,st,old))
    # Fixed allocation V4 with expanded calibration, separating added data from
    # changes in fusion and temporal scoring.
    oldwide=role_reference(ref,sr)
    wv=causal_scores(val,normalize(val,sv,oldwide))
    wt=causal_scores(test,normalize(test,st,oldwide))
    full=stack([train,fusion])
    flat=fit_rf(full,LOCAL,args.seed+41,n=90)
    fv=1-rf_scores(val,LOCAL,flat)["normal"].to_numpy()
    ft=1-rf_scores(test,LOCAL,flat)["normal"].to_numpy()
    for budget in BUDGETS:
        for name,dv,v,t in [
            ("v4_exact",val0,ov,ot),("v4_more_calibration",val,wv,wt)]:
            cuts=cuts_for(dv,v,budget,ALLOC)
            add_eval(rows,dv,margin(v,cuts),0.,name,"validation",budget)
            add_eval(rows,test,margin(t,cuts),0.,name,"test",budget)
        for name,cut in [
            ("v5_selected",safe_threshold(val,a,budget)),
            ("v5_worst_world",worst_world_threshold(val,a,budget))]:
            add_eval(rows,val,a,cut,name,"validation",budget)
            add_eval(rows,test,b,cut,name,"test",budget)
            if name=="v5_selected" and budget==BUDGET:
                test[["world","nid","t","role","attack","episode"]].assign(
                    score=b,alarm=b>cut).to_csv(out/"predictions.csv",index=False)
        cut=safe_threshold(val,fv,budget)
        add_eval(rows,val,fv,cut,"flat_rf","validation",budget)
        add_eval(rows,test,ft,cut,"flat_rf","test",budget)
    # Per-test-world diagnostic at frozen primary threshold, no recalibration.
    cut=safe_threshold(val,a,BUDGET)
    per=[]
    for world,g in test.groupby("world",sort=False):
        for row in evaluate_budget(g,b[g.index],cut,"v5_selected","test",BUDGET):
            per.append(dict(world=int(world),**row))
    res=pd.DataFrame(rows); res.to_csv(out/"metrics.csv",index=False)
    pd.DataFrame(per).to_csv(out/"per_world.csv",index=False)
    pd.DataFrame(overhead).to_csv(out/"kie_overhead.csv",index=False)
    (out/"manifest.json").write_text(json.dumps(dict(args=vars(args),plan=plan,
        selected=selected,mode=mode,drift_strength=strength,channels=channels,
        confirmation_windows=K,primary_budget=BUDGET,
        scope="pilot; aggregate proxies; simulated KIE; no live validation"),indent=2))
    print("PRIMARY",flush=True)
    print(res[(res.split=="test")&(res.budget==BUDGET)].to_string(index=False),flush=True)
    print("VALIDATION",flush=True)
    print(pd.DataFrame(tuning).query("attack == 'all'").to_string(index=False),flush=True)
    print("BUDGET SENSITIVITY",flush=True)
    print(res[(res.split=="test")&(res.attack=="all")].to_string(index=False),flush=True)
    print("PER WORLD",flush=True)
    print(pd.DataFrame(per).query("attack == 'all'").to_string(index=False),flush=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--seed",type=int,required=True)
    p.add_argument("--days",type=float,default=.16)
    p.add_argument("--repetitions",type=int,default=2)
    p.add_argument("--out",required=True)
    a=p.parse_args(); run(a)
if __name__=="__main__": main()
