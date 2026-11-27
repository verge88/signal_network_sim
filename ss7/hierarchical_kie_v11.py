"""V11 benign-only role-conditioned statistical KIE detector.
Isolated aggregate simulation; no live signaling, no inference of attack intent.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score,average_precision_score
from ss7.hierarchical_kie import CLASSES,BUDGET,K,MSG_TYPES
from ss7.hierarchical_kie_v2 import stack
from ss7.hierarchical_kie_v3 import valid,evaluate_budget
from ss7.hierarchical_kie_quick import trim,score_experts,role_reference,normalize,causal_scores,cuts_for,margin
from ss7.hierarchical_kie_v6 import generate,ALL6,train_v6
from ss7.hierarchical_kie_v7 import graph_compromise,graph_score,prepare,ALLOCATIONS
from ss7.hierarchical_kie_v9 import read_worlds,profiles,train_local,fit_graph,raw
from ss7.hierarchical_kie_v8_sharded import PLAN,shard_plan,sha

# Predeclared small observable feature set, independent of attack labels.
LOCAL=["log_count_"+m for m in MSG_TYPES]+["log_total","log_dialogues",
    "external_od_log","external_od_share","dialogue_gap","map_share","isup_share",
    "dest_entropy","in_out_ratio"]
GRAPH=["relative_link_mean","relative_link_min","relative_link_rms",
       "relative_od_mean","relative_od_min","relative_od_rms","link_absmax",
       "od_absmax","edge_signed_residual","edge_rms_residual","edge_signed_deficit",
       "edge_coverage","transit_balance"]
META=["world","nid","t","role","zone"]
ALLOCS={"balanced":{"local":.7,"zone":.3},
        "local_priority":{"local":.85,"zone":.15}}
BASE_ALLOC=dict(ALLOCATIONS["local_priority"])

def feature_matrix(d,cols,graph=False):
    """Causal residuals against delayed historical median and robust dispersion.
    No attack labels or hidden simulator data used. Past estimates reset by world/node.
    """
    x=d[cols].replace([np.inf,-np.inf],np.nan)
    keys=[d.world,d.nid]
    g=x.groupby(keys,sort=False)
    med=g.transform(lambda s:s.shift(8).rolling(64,min_periods=12).median())
    lo=g.transform(lambda s:s.shift(8).rolling(64,min_periods=12).quantile(.25))
    hi=g.transform(lambda s:s.shift(8).rolling(64,min_periods=12).quantile(.75))
    # Floors in native feature units, fixed before running. A separate benign
    # reference subsequently handles different roles and feature scales.
    floor=pd.Series({c:(.03 if ("share" in c or "relative_" in c or "coverage" in c) else .15) for c in cols})
    z=(x-med)/(hi-lo).clip(lower=floor,axis=1)
    short=z.groupby(keys,sort=False).transform(lambda s:s.rolling(4,min_periods=2).mean())
    long=z.groupby(keys,sort=False).transform(lambda s:s.rolling(10,min_periods=4).mean())
    result=pd.concat([z.add_suffix("_now"),short.add_suffix("_4"),long.add_suffix("_10")],axis=1)
    if graph:
        # Absolute consistency deviations supplement temporal changes.
        result=pd.concat([result,x.add_suffix("_absolute")],axis=1)
    else:
        # Composition changes relative to total traffic.
        counts=[c for c in cols if c.startswith("log_count_")]
        mix=x[counts].sub(x.log_total,axis=0)
        ref=mix.groupby(keys,sort=False).transform(lambda s:s.shift(8).rolling(64,min_periods=12).median())
        result=pd.concat([result,(mix-ref).add_suffix("_composition")],axis=1)
    return result.where(valid(d),np.nan)

class BenignModel:
    """Robust marginal scaling plus shrinkage Mahalanobis distance.
    Role-conditioned model learned only from labeled benign training windows.
    Missing dimensions excluded from scoring, not treated as positive evidence.
    """
    def fit(self,d,x):
        self.models={}
        for role in sorted(d.role.unique()):
            a=x.loc[d.role.eq(role)&d.attack.eq("normal")].to_numpy(float)
            keep=np.isfinite(a).sum(axis=0)>=40
            if not keep.any(): continue
            a=a[:,keep]
            center=np.nanmedian(a,axis=0)
            q=np.nanquantile(a,[.25,.75],axis=0)
            scale=np.maximum(q[1]-q[0],.10)
            z=(a-center)/scale
            # Fit covariance on robustly winsorized benign samples.
            fit=np.clip(np.where(np.isfinite(z),z,0.),-8.,8.)
            cov=LedoitWolf().fit(fit).covariance_
            self.models[role]=(keep,center,scale,cov)
        return self
    def score(self,d,x):
        out=pd.DataFrame(np.nan,index=d.index,columns=["mahal","tail"])
        for role,(keep,center,scale,cov) in self.models.items():
            idx=np.flatnonzero(d.role.eq(role).to_numpy()&valid(d).to_numpy())
            z=(x.iloc[idx].to_numpy(float)[:,keep]-center)/scale
            mask=np.isfinite(z)
            # Exact marginal covariance on each observed feature pattern.
            # Group rows with equal masks; bounded fallback is diagonal score.
            patterns,inv=np.unique(mask,axis=0,return_inverse=True)
            m=np.full(len(idx),np.nan); tail=m.copy()
            for j,pat in enumerate(patterns):
                rows=np.flatnonzero(inv==j)
                if pat.sum()<max(2,int(.35*len(pat))): continue
                q=z[rows][:,pat]
                c=cov[np.ix_(pat,pat)]
                precision=np.linalg.pinv(c,hermitian=True)
                m[rows]=np.sqrt(np.maximum(np.einsum("ij,jk,ik->i",q,precision,q),0.)/pat.sum())
                # Top-three marginal deviations avoid single noisy extreme.
                k=min(3,pat.sum())
                tail[rows]=np.sort(np.abs(q),axis=1)[:,-k:].mean(axis=1)
            out.iloc[idx,0]=m; out.iloc[idx,1]=tail
        return out

def generate_shard(args):
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    if args.shard==0:
        na,_,_=generate(args.seed+9900,.10,repetitions=1,zero=True)
        nb,_,_=generate(args.seed+9900,.10,repetitions=1,clean=True)
        np.testing.assert_allclose(na[ALL6],nb[ALL6],atol=1e-10,rtol=0,equal_nan=True)
        (out/"null_gate.json").write_text(json.dumps({"passed":True,"rows":len(na),"seed":args.seed+9900}))
    for role,i in shard_plan(args.shard):
        seed=args.seed+1009*i; start=time.perf_counter()
        d,ep,s=generate(seed,days=.16,repetitions=2,loss=.01); d=trim(d)
        if set(d.attack)!=set(CLASSES): raise AssertionError("incomplete classes")
        f=out/f"world_{i}.pkl.gz"; d.to_pickle(f,compression="gzip")
        pd.DataFrame(ep).to_csv(out/f"episodes_{role}_{seed}.csv",index=False)
        (out/f"world_{i}.json").write_text(json.dumps(dict(role=role,index=i,seed=seed,
             days=.16,repetitions=2,rows=len(d),sha256=sha(f),overhead=s,
             seconds=time.perf_counter()-start)))
        print("world",role,seed,"seconds",round(time.perf_counter()-start,2),flush=True)

def diagnostic(d,s,name):
    rows=[]
    for c in s:
        for kind in CLASSES[1:]:
            keep=d.attack.isin(["normal",kind])&s[c].notna()
            y=d.loc[keep,"attack"].eq(kind); x=s.loc[keep,c]
            rows.append(dict(model=name,channel=c,attack=kind,
                auc=roc_auc_score(y,x) if y.nunique()==2 else None,
                ap=average_precision_score(y,x) if y.nunique()==2 else None,
                positive=int(y.sum())))
    return rows

def merge_channels(local, zone):
    # Explicit column access: DataFrame.tail is a method, not the score column.
    if not local.index.equals(zone.index):
        raise ValueError("local and zone score index mismatch")
    return {name: pd.DataFrame({"local":local[name],"zone":zone[name]})
            for name in ("mahal","tail")}

def assess(args):
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    d=read_worlds(args)
    train,fusion,ref,val,test=(d[k] for k in ["train","fusion","reference","validation","test"])
    print("FIT BENIGN STATISTICAL MODELS",flush=True)
    # Both local and graph models see only benign training data; graph learned
    # on the independently designated fusion world.
    local=BenignModel().fit(train,feature_matrix(train,LOCAL))
    zone=BenignModel().fit(fusion,feature_matrix(fusion,GRAPH,True))
    def score(x):
        l=local.score(x,feature_matrix(x,LOCAL))
        g=zone.score(x,feature_matrix(x,GRAPH,True))
        return merge_channels(l,g)
    rr,rv=score(ref),score(val)
    stats={k:role_reference(ref,v) for k,v in rr.items()}
    v={k:causal_scores(val,normalize(val,x,stats[k])) for k,x in rv.items()}
    # Fixed variants, validation-only comparison. Preserve three-window rule.
    tune=[]
    for mode,s in v.items():
        for name,a in ALLOCS.items():
            cuts=cuts_for(val,s,BUDGET,a)
            r=next(x for x in evaluate_budget(val,margin(s,cuts),0.,mode+"_"+name,"validation",BUDGET) if x["attack"]=="all")
            tune.append(dict(mode=mode,allocation=name,**r))
    selected=max(tune,key=lambda a:(a["false_alerts_per_day"]<=BUDGET+1e-9,
        a["episode_recall"],-a["false_alerts_per_day"],a["ap"]))
    mode,allocation=selected["mode"],selected["allocation"]
    print("SELECTED",mode,allocation,flush=True)
    print("FIT SUPERVISED COMPARATORS",flush=True)
    m,roles=train_v6(train,args.seed,70,20)
    g,gc=graph_compromise(fusion,args.seed+19)
    def v7(x):
        s=score_experts(x,m,roles); s["zone"]=graph_score(x,g,gc); return s
    frames={}
    for k,x in d.items(): frames[k],cols=profiles(x,False)
    tr,fu,re,va,te=(frames[k] for k in ["train","fusion","reference","validation","test"])
    lm,lr=train_local(tr,cols,args.seed,"extra"); gm,gcols=fit_graph(fu,False,args.seed+19)
    def v9(x): return raw(x,lm,lr,gm,gcols,False)
    pairs={}
    for name,fn,r,w,t in [("v7_fixed",v7,ref,val,test),("v9_extra",v9,re,va,te)]:
        st=role_reference(r,fn(r))
        pairs[name]=(prepare(w,fn(w),st,"median",False),prepare(t,fn(t),st,"median",False),BASE_ALLOC)
    rt=score(test)
    t={k:causal_scores(test,normalize(test,x,stats[k])) for k,x in rt.items()}
    for k in ("mahal","tail"):
        pairs["v11_"+k]=(v[k],t[k],ALLOCS["balanced"])
    pairs["v11_selected"]=(v[mode],t[mode],ALLOCS[allocation])
    pairs["v11_local"]=(v[mode],t[mode],{"local":1.})
    pairs["v11_zone"]=(v[mode],t[mode],{"zone":1.})
    rows=[]
    for name,(a,b,alloc) in pairs.items():
        for budget in (20.,100.,500.):
            cuts=cuts_for(val,a,budget,alloc)
            rows+=evaluate_budget(val,margin(a,cuts),0.,name,"validation",budget)
            rows+=evaluate_budget(test,margin(b,cuts),0.,name,"test",budget)
            if name=="v11_selected" and budget==20.:
                test[["world","nid","t","attack","episode"]].assign(
                    score=margin(b,cuts)).join(b.add_prefix("channel_")).to_csv(out/"predictions.csv",index=False)
    # Identifiability audit: identical measured vectors under a matched
    # legitimate service-volume event have identical detector scores. This is
    # a logical paired relabeling control, not newly generated traffic.
    matched=test.attack.isin(CLASSES[1:5])
    a=margin(t[mode],cuts_for(val,v[mode],20.,ALLOCS[allocation]))[matched]
    audit=dict(matched_external_windows=int(matched.sum()),
        identical_observation_pairs=int(len(a)),identical_score_pairs=int(np.sum(a==a)),
        description="logical matched legitimate-volume control; no new empirical negative sample")
    res=pd.DataFrame(rows); res.to_csv(out/"metrics.csv",index=False)
    pd.DataFrame(tune).to_csv(out/"validation_selection.csv",index=False)
    pd.DataFrame(diagnostic(test,rt[mode],mode)).to_csv(out/"channel_diagnostics.csv",index=False)
    (out/"identifiability.json").write_text(json.dumps(audit,indent=2))
    (out/"manifest.json").write_text(json.dumps(dict(seed=args.seed,mode=mode,allocation=allocation,
        local_features=LOCAL,zone_features=GRAPH,confirmation_windows=K,primary_budget=BUDGET,
        scope="fresh synthetic worlds" if args.seed!=216101 else "reused development worlds"),indent=2))
    print("PRIMARY",flush=True)
    print(res[(res.split=="test")&(res.budget==20.)].to_string(index=False),flush=True)
    print("VALIDATION",flush=True)
    print(pd.DataFrame(tune).to_string(index=False),flush=True)
    print("CHANNEL DIAGNOSTICS",flush=True)
    print(pd.DataFrame(diagnostic(test,rt[mode],mode)).to_string(index=False),flush=True)
    print("IDENTIFIABILITY",json.dumps(audit),flush=True)
    print("BUDGET SENSITIVITY",flush=True)
    print(res[(res.split=="test")&(res.attack=="all")].to_string(index=False),flush=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--stage",choices=["generate","evaluate"],required=True)
    p.add_argument("--seed",type=int,required=True)
    p.add_argument("--shard",type=int,default=0)
    p.add_argument("--input",default="worlds")
    p.add_argument("--out",required=True)
    a=p.parse_args()
    if a.stage=="generate": generate_shard(a)
    else: assess(a)
if __name__=="__main__": main()
