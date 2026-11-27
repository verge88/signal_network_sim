"""V9 diagnostic on previously generated synthetic KIE worlds. No live traffic."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier,HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score,average_precision_score
from ss7.hierarchical_kie import CLASSES,BUDGET,K,MSG_TYPES
from ss7.hierarchical_kie_v2 import stack,xmat,group_weights
from ss7.hierarchical_kie_v3 import valid,evaluate_budget,rf_scores
from ss7.hierarchical_kie_quick import KINDS,role_reference,score_experts,cuts_for,margin
from ss7.hierarchical_kie_v6 import train_v6,graph_only,fit_regularized
from ss7.hierarchical_kie_v7 import graph_compromise,graph_score,prepare,ALLOCATIONS
from ss7.hierarchical_kie_v8 import OBS_LOCAL,OBS_GRAPH,COMP
from ss7.hierarchical_kie_v8_sharded import PLAN,sha
ALLOC=dict(ALLOCATIONS["local_priority"])
# Relative changes of locally reported counts; no IDs, roles or labels as inputs.
BASE=["log_count_"+m for m in MSG_TYPES]+["log_total","log_dialogues",
    "external_od_log","external_od_share","dialogue_gap","map_share","isup_share",
    "dest_entropy","in_out_ratio"]
def profiles(d,monitor=False):
    d=d.copy().reset_index(drop=True)
    names=BASE+(OBS_LOCAL if monitor else [])
    g=d.groupby(["world","nid"],sort=False)[names]
    past=g.transform(lambda s:s.shift(8).rolling(64,min_periods=12).median())
    recent=g.transform(lambda s:s.rolling(4,min_periods=2).mean())
    delta=d[names]-past; delta.columns=[c+"_p9" for c in names]
    mean=recent-past; mean.columns=[c+"_m9" for c in names]
    # Ratios derived from log1p counts; not inferred transaction authorization.
    ratios=pd.DataFrame({m+"_share9":d["log_count_"+m]-d.log_total for m in MSG_TYPES})
    return pd.concat([d,delta,mean,ratios],axis=1),list(delta.columns)+list(mean.columns)+list(ratios.columns)+["dialogue_gap","external_od_share","map_share","isup_share"]

def train_local(d,cols,seed,kind):
    mods={}; roles={}
    for i,c in enumerate(KINDS):
        roles[c]=sorted(d.loc[d.attack.eq(c),"role"].unique())
        g=d[d.role.isin(roles[c])].reset_index(drop=True); y=g.attack.eq(c).to_numpy()
        if kind=="hgb":
            m=HistGradientBoostingClassifier(max_iter=65,max_leaf_nodes=12,
                min_samples_leaf=35,l2_regularization=7.,learning_rate=.07,
                random_state=seed+i,early_stopping=False)
            mods[c]=("hgb",m.fit(xmat(g,cols),y,sample_weight=group_weights(g,y)),cols)
        else:
            x=xmat(g,cols); z=x[g.attack.eq("normal")]
            med=np.zeros(x.shape[1]); ok=np.isfinite(z).any(axis=0)
            med[ok]=np.nanmedian(z[:,ok],axis=0)
            xx=np.concatenate([np.where(np.isfinite(x),x,med),~np.isfinite(x)],axis=1)
            m=ExtraTreesClassifier(n_estimators=70,min_samples_leaf=20,max_depth=12,
                max_features="sqrt",random_state=seed+i,n_jobs=2)
            mods[c]=("rf",(m.fit(xx,y,sample_weight=group_weights(g,y)),med),cols)
    return mods,roles

def graph_frame(d,monitor):
    g=graph_only(d)
    if monitor:
        g=pd.concat([g,d[OBS_GRAPH]],axis=1)
    return g

def fit_graph(d,monitor,seed):
    g=graph_frame(d,monitor); cols=list(g.columns)
    return fit_regularized(g.assign(world=d.world,episode=d.episode,attack=d.attack),
        cols,d.attack.eq(COMP).to_numpy(),seed,60,20),cols

def raw(d,m,roles,g,gcols,monitor):
    p=score_experts(d,m,roles)
    p["zone"]=rf_scores(graph_frame(d,monitor),gcols,g)[True].where(valid(d),np.nan)
    return p

def read_worlds(args):
    src=Path(args.input)
    n=json.loads((src/"null_gate.json").read_text())
    if not n.get("passed") or n["seed"]!=args.seed+9900: raise ValueError("null gate")
    data={k:[] for k,_ in PLAN}
    for role,i in PLAN:
        f=src/f"world_{i}.pkl.gz"; m=json.loads((src/f"world_{i}.json").read_text())
        if (m["role"],m["index"],m["seed"],m["sha256"])!=(role,i,args.seed+1009*i,sha(f)):
            raise ValueError("manifest mismatch")
        d=pd.read_pickle(f,compression="gzip")
        if len(d)!=m["rows"] or set(d.world)!={m["seed"]} or set(d.attack)!=set(CLASSES):
            raise ValueError("world mismatch")
        data[role].append(d)
    return {k:stack(v).sort_values(["world","nid","t"]).reset_index(drop=True) for k,v in data.items()}

def diagnostic(d,s,name,split):
    rows=[]
    for c in s:
        target=COMP if c=="zone" else c
        keep=d.attack.isin(["normal",target])&s[c].notna()
        y=d.loc[keep,"attack"].eq(target); x=s.loc[keep,c]
        rows.append(dict(model=name,split=split,channel=c,positive=int(y.sum()),
            availability=float(s[c].notna().mean()),
            auc=roc_auc_score(y,x) if y.nunique()==2 else None,
            ap=average_precision_score(y,x) if y.nunique()==2 else None))
    return rows

def compare(args):
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    data=read_worlds(args)
    train,fusion,ref,val,test=(data[k] for k in ["train","fusion","reference","validation","test"])
    configs={}; diag=[]; tuning=[]
    m,roles=train_v6(train,args.seed,70,20)
    g,gcols=graph_compromise(fusion,args.seed+19)
    def baseline(d):
        s=score_experts(d,m,roles)
        s["zone"]=graph_score(d,g,gcols)
        return s
    rr,rv=map(baseline,[ref,val])
    stats=role_reference(ref,rr); vs=prepare(val,rv,stats,"median",False)
    configs["v7_fixed"]=(baseline,stats,vs,test)
    for monitor in (False,True):
        frames={}
        for k,d in data.items():
            frames[k],cols=profiles(d,monitor)
        tr,fu,re,va,te=(frames[k] for k in ["train","fusion","reference","validation","test"])
        gm,gc=fit_graph(fu,monitor,args.seed+19)
        for kind in ("extra","hgb"):
            name=kind+("_observer" if monitor else "_kie")
            print("FIT",name,flush=True)
            lm,lr=train_local(tr,cols,args.seed,kind)
            def scorer(d,lm=lm,lr=lr,gm=gm,gc=gc,monitor=monitor):
                return raw(d,lm,lr,gm,gc,monitor)
            rr,rv=scorer(re),scorer(va); st=role_reference(re,rr)
            vs=prepare(va,rv,st,"median",False)
            diag+=diagnostic(va,rv,name,"validation")
            configs[name]=(scorer,st,vs,te)
    # Independent validation choice, no test scores before this point.
    for name,(_,_,v,_) in configs.items():
        cuts=cuts_for(val,v,BUDGET,ALLOC)
        tuning+=evaluate_budget(val,margin(v,cuts),0.,name,"validation",BUDGET)
    eligible=[x for x in tuning if x["attack"]=="all" and x["model"]!="v7_fixed"]
    best=max(eligible,key=lambda a:(a["false_alerts_per_day"]<=BUDGET+1e-9,
        a["episode_recall"],-a["false_alerts_per_day"],a["ap"]))["model"]
    print("SELECTED",best,flush=True)
    rows=[]
    # Development data only. Report candidates transparently, no confirmation claim.
    for name,(scorer,st,v,te) in configs.items():
        rt=scorer(te); t=prepare(te,rt,st,"median",False)
        diag+=diagnostic(te,rt,name,"test")
        for budget in (20.,100.,500.):
            alloc=ALLOC
            cuts=cuts_for(val,v,budget,alloc)
            vm,tm=margin(v,cuts),margin(t,cuts)
            rows+=evaluate_budget(val,vm,0.,name,"validation",budget)
            rows+=evaluate_budget(te,tm,0.,name,"test",budget)
            if name==best:
                for ab,al in [("selected_local",{c:.2 for c in KINDS}),("selected_zone",{"zone":1.})]:
                    cc=cuts_for(val,v,budget,al)
                    rows+=evaluate_budget(val,margin(v,cc),0.,ab,"validation",budget)
                    rows+=evaluate_budget(te,margin(t,cc),0.,ab,"test",budget)
    res=pd.DataFrame(rows); res.to_csv(out/"metrics.csv",index=False)
    pd.DataFrame(diag).to_csv(out/"channel_diagnostics.csv",index=False)
    pd.DataFrame(tuning).to_csv(out/"validation_selection.csv",index=False)
    (out/"manifest.json").write_text(json.dumps(dict(seed=args.seed,source_run=46037193622,
        selected=best,scope="reused development test worlds; no new independent confirmation",
        allocation=ALLOC,confirmation_windows=K,primary_budget=BUDGET),indent=2))
    print("PRIMARY",flush=True)
    print(res[(res.split=="test")&(res.budget==BUDGET)].to_string(index=False),flush=True)
    print("VALIDATION",flush=True)
    print(pd.DataFrame(tuning).query("attack == 'all'").to_string(index=False),flush=True)
    print("CHANNEL DIAGNOSTICS",flush=True)
    print(pd.DataFrame(diag).to_string(index=False),flush=True)
    print("BUDGET SENSITIVITY",flush=True)
    print(res[(res.split=="test")&(res.attack=="all")].to_string(index=False),flush=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--seed",type=int,default=216101)
    p.add_argument("--input",default="worlds")
    p.add_argument("--out",required=True)
    compare(p.parse_args())
if __name__=="__main__": main()
