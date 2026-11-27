"""Exact V8 rerun, independent parallel world generation with paired evaluation."""
import argparse, json, time, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score
from ss7.hierarchical_kie_v8 import (
    generate,ALL8,train_new,graph_fit,graph_new,ALLOC8,COMP
)
from ss7.hierarchical_kie_v6 import train_v6
from ss7.hierarchical_kie_v7 import graph_compromise,graph_score,prepare
from ss7.hierarchical_kie_quick import trim,KINDS,score_experts,role_reference,cuts_for,margin
from ss7.hierarchical_kie_v3 import rf_scores,valid,evaluate_budget
from ss7.hierarchical_kie_v2 import stack
from ss7.hierarchical_kie import CLASSES,BUDGET,K
PLAN=[("train",0),("train",1),("train",2),("fusion",3),
      ("reference",4),("validation",5),("test",6),("test",7),
      ("reference",8),("validation",9),("validation",10),("validation",11)]
SHARDS=4

def shard_plan(shard):
    if not 0<=shard<SHARDS: raise ValueError("invalid shard")
    return [p for p in PLAN if p[1]%SHARDS==shard]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def generate_shard(args):
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    if args.shard==0:
        na,_,_=generate(args.seed+9900,days=.10,repetitions=1,zero=True)
        nb,_,_=generate(args.seed+9900,days=.10,repetitions=1,clean=True)
        np.testing.assert_allclose(na[ALL8],nb[ALL8],atol=1e-10,rtol=0,equal_nan=True)
        (out/"null_gate.json").write_text(json.dumps({"passed":True,"rows":len(na),"seed":args.seed+9900}))
    for role,i in shard_plan(args.shard):
        seed=args.seed+1009*i; start=time.perf_counter()
        print("generate",role,seed,flush=True)
        d,ep,s=generate(seed,days=args.days,repetitions=args.repetitions,loss=.01)
        d=trim(d)
        if set(d.attack)!=set(CLASSES): raise AssertionError("incomplete classes")
        # Trusted artifacts produced by this same workflow; no external pickle.
        # Binary compression preserves dtypes and scores exactly.
        f=out/f"world_{i}.pkl.gz"
        d.to_pickle(f,compression="gzip")
        pd.DataFrame(ep).to_csv(out/f"episodes_{role}_{seed}.csv",index=False)
        (out/f"world_{i}.json").write_text(json.dumps(dict(role=role,index=i,seed=seed,
            days=args.days,repetitions=args.repetitions,rows=len(d),sha256=sha(f),
            seconds=time.perf_counter()-start,overhead=s)))
        print("world seconds",i,round(time.perf_counter()-start,2),flush=True)

def evaluate_all(args):
    src=Path(args.input); out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    null=json.loads((src/"null_gate.json").read_text())
    if not null.get("passed") or null["seed"]!=args.seed+9900: raise ValueError("null gate")
    (out/"null_gate.json").write_text(json.dumps(null))
    data={k:[] for k,_ in PLAN}; overhead=[]; timing=[]
    for role,i in PLAN:
        f=src/f"world_{i}.pkl.gz"; m=json.loads((src/f"world_{i}.json").read_text())
        seed=args.seed+1009*i
        if (m["role"],m["index"],m["seed"],m["days"],m["repetitions"],m["sha256"])!=(role,i,seed,args.days,args.repetitions,sha(f)):
            raise ValueError("world manifest mismatch")
        d=pd.read_pickle(f,compression="gzip")
        if len(d)!=m["rows"] or set(d.world)!={seed} or set(d.attack)!=set(CLASSES):
            raise ValueError("world data mismatch")
        data[role].append(d)
        overhead.append(dict(role=role,seed=seed,**m["overhead"]))
        timing.append(dict(role=role,seed=seed,seconds=m["seconds"]))
        ep=src/f"episodes_{role}_{seed}.csv"
        (out/ep.name).write_bytes(ep.read_bytes())
    pd.DataFrame(timing).to_csv(out/"generation_timing.csv",index=False)
    print("GENERATION TIMING",flush=True)
    print(pd.DataFrame(timing).to_string(index=False),flush=True)
    train,fusion,ref,val,test=(stack(data[k]) for k in ["train","fusion","reference","validation","test"])
    print('fit paired V7',flush=True)
    old,oroles=train_v6(train,args.seed,70,20)
    og,oc=graph_compromise(fusion,args.seed+19)
    print('fit V8',flush=True)
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
    rows=[]; diagnostic=[]
    for enhanced,name in [(False,"v7_fixed"),(True,"v8_hierarchy")]:
        rr,rv,rt=[raw(d,enhanced) for d in [ref,val,test]]
        # Diagnostic only, computed after fitting; not used for thresholds.
        for channel in rv:
            target=COMP if channel=="zone" else channel
            for split,d,raw_scores in [("validation",val,rv),("test",test,rt)]:
                keep=d.attack.isin(["normal",target]) & raw_scores[channel].notna()
                yy=d.loc[keep,"attack"].eq(target)
                xx=raw_scores.loc[keep,channel]
                diagnostic.append(dict(model=name,split=split,channel=channel,
                    rows=int(keep.sum()),positive=int(yy.sum()),
                    availability=float(raw_scores[channel].notna().mean()),
                    auc=roc_auc_score(yy,xx) if yy.nunique()==2 else None,
                    ap=average_precision_score(yy,xx) if yy.nunique()==2 else None))
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
    pd.DataFrame(diagnostic).to_csv(out/"channel_diagnostics.csv",index=False)
    print("CHANNEL DIAGNOSTICS",flush=True)
    print(pd.DataFrame(diagnostic).to_string(index=False),flush=True)
    print("KIE OVERHEAD",flush=True)
    print(over.to_string(index=False),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--stage",choices=["generate","evaluate"],required=True)
    p.add_argument("--seed",type=int,required=True)
    p.add_argument("--days",type=float,default=.16)
    p.add_argument("--repetitions",type=int,default=2)
    p.add_argument("--shard",type=int,default=0)
    p.add_argument("--input",default="worlds")
    p.add_argument("--out",required=True)
    a=p.parse_args()
    if a.stage=="generate": generate_shard(a)
    else: evaluate_all(a)
if __name__=="__main__": main()
