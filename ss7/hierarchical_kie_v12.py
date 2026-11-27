"""V12 outcome-conditioned synthetic signaling and matched benign load controls.

Isolated aggregate transaction cohorts, not live protocol traffic. The original
simulator remains untouched. Generic request/return stage direction is an
approximation, not a decoded MAP/TCAP or ISUP state machine.
"""
import argparse,json,time
from pathlib import Path
from typing import Dict,List,Tuple
from dataclasses import replace
import numpy as np
import pandas as pd
from ss7.fix import (NodeState,Episode,Flow,Service,SERVICE_TEMPLATE,TCAP_SERVICES,
    MSG_TYPES,MT_IDX,link_state,SimConfig,RngBook,KeyCustody,make_episodes,INTERVALS_PER_DAY)
from ss7.hierarchical_kie import AggregateWorld,schedule,deliver,CLASSES,BUDGET,K,FAMILIES
from ss7.hierarchical_kie_v6 import observe_v6,features_v6,ALL6,train_v6
from ss7.hierarchical_kie_v8_sharded import PLAN,shard_plan,sha
from ss7.hierarchical_kie_v2 import stack
from ss7.hierarchical_kie_v3 import evaluate_budget
from ss7.hierarchical_kie_quick import trim,score_experts,role_reference,cuts_for,margin
from ss7.hierarchical_kie_v7 import graph_compromise,graph_score,prepare,ALLOCATIONS
from ss7.hierarchical_kie_v9 import profiles,train_local,fit_graph,raw
ALLOC=dict(ALLOCATIONS["local_priority"])

def account_cohort(states,path,service,count,closed,rng,hidden=None):
    """Count each abstract stage once at both ends of each traversed hop.
    Initiation always occurs; later stages depend on outcome and a sampled
    partial completion count. No actual protocol sequence is implied.
    """
    if not 0<=closed<=count: raise ValueError("invalid cohort outcome")
    stages=[m for m,n in SERVICE_TEMPLATE[service].items() for _ in range(n)]
    remaining=count-closed
    for i,m in enumerate(stages):
        if i==0: volume=count
        else:
            remaining=int(rng.binomial(remaining,.45))
            volume=closed+remaining
        if volume<=0: continue
        # Abstract direction: initial stage forward, later stages alternate.
        # Not a literal representation of the operation's real wire direction.
        route=path if i%2==0 else path[::-1]
        for u,v in zip(route[:-1],route[1:]):
            edge=(min(u,v),max(u,v))
            states[u].mtype[MT_IDX[m]]+=volume
            states[v].mtype[MT_IDX[m]]+=volume
            states[u].link_out[edge]=states[u].link_out.get(edge,0)+volume
            states[v].link_in[edge]=states[v].link_in.get(edge,0)+volume
            if hidden is not None:
                nid=hidden["ep"].nid
                if nid==u or nid==v:
                    hidden["mtype"][MT_IDX[m]]+=volume
                    hidden["msgs"]+=volume
                    hidden["links"][edge]=hidden["links"].get(edge,0)+volume
    for i,nid in enumerate(path):
        st=states[nid]
        if i==0:
            st.originated+=count; st.dest[path[-1]]=st.dest.get(path[-1],0)+count
        elif i==len(path)-1:
            st.terminated+=count; st.dest[path[0]]=st.dest.get(path[0],0)+count
        else: st.transited+=count
        if i in (0,len(path)-1) and service in TCAP_SERVICES:
            st.dlg_begin+=count; st.dlg_end+=closed
        if hidden is not None and nid==hidden["ep"].nid:
            hidden["dest"][path[-1]]=hidden["dest"].get(path[-1],0)+count
            if i in (0,len(path)-1) and service in TCAP_SERVICES:
                hidden["dlg"]=(hidden["dlg"][0]+count,hidden["dlg"][1]+closed)

class OutcomeWorld(AggregateWorld):
    def step_true(self, t: int, episodes: List[Episode]
                  ) -> Tuple[Dict[int, NodeState], Dict[int, Dict], Dict]:
        cfg, topo = self.cfg, self.topo
        rng = self.book.child("traffic", t)
        dt = cfg.interval_s
        phase = 2.0 * np.pi * (t % INTERVALS_PER_DAY) / INTERVALS_PER_DAY
        diurnal = 1.0 + self.diurnal_amp * np.sin(phase - 1.2)

        # AR(1) по узлам — общий механизм для всех, без веток
        for n in topo.nodes:
            self._ar_x[n] = (self._ar_phi[n] * self._ar_x[n]
                             + rng.normal(0.0, self._ar_sig[n]))
        node_mult = {n: float(np.exp(self._ar_x[n])) for n in topo.nodes}

        # нормальные события (негативные контроли)
        ev_mult = {n: 1.0 for n in topo.nodes}
        failed_links = set()
        for ep in episodes:
            if not (ep.t0 <= t <= ep.t1) or ep.kind == "compromise":
                continue
            if ep.kind in ("traffic_spike", "traffic_dip", "scale_in", "campaign"):
                for n in (topo.nodes if ep.nid is None else [ep.nid]):
                    ev_mult[n] *= ep.mult
            elif ep.kind == "link_failure" and ep.nid is not None:
                for l in topo.links_of(ep.nid)[:1]:
                    failed_links.add(l)

        states = {n: NodeState(n, t, np.zeros(len(MSG_TYPES), np.int64))
                  for n in topo.nodes}
        offered: Dict[Tuple[int, int], float] = {}
        pending: List[Tuple] = []

        # --- первый проход: раскладка транзакций по маршрутам
        for f in self.flows:
            lam = f.rate * dt * diurnal * node_mult[f.src] * ev_mult[f.src]
            # внешние атаки на границе (SRI-SM flood / ATI tracking / IRSF)
            for ep in episodes:
                if (ep.kind == "ext_attack" and ep.t0 <= t <= ep.t1
                        and f.src == ep.nid and f.service in
                        (Service.ROAM_SRI, Service.SMS_MT, Service.LOCATION)):
                    lam *= ep.mult
            n_txn = int(rng.poisson(max(lam, 0.0)))
            if n_txn <= 0:
                continue
            r = topo.route(f.src, f.dst)
            if not r.paths:
                continue
            split = rng.multinomial(n_txn, r.shares)
            for path, cnt in zip(r.paths, split):
                if cnt <= 0 or any((min(u, v), max(u, v)) in failed_links
                                   for u, v in zip(path[:-1], path[1:])):
                    continue
                pending.append((f, path, int(cnt)))
                tmpl = SERVICE_TEMPLATE[f.service]
                bits = self._service_bits(tmpl)
                for u, v in zip(path[:-1], path[1:]):
                    k = (min(u, v), max(u, v))
                    offered[k] = offered.get(k, 0.0) + cnt * bits / dt

        # --- скрытая активность компрометированного узла
        hidden: Dict[int, Dict] = {}
        hidden_flow_ids = {}
        for ep in episodes:
            if ep.kind != "compromise" or not (ep.t0 <= t <= ep.t1):
                continue
            nid = ep.nid
            hidden_lambda = max(ep.hidden_rate * dt, 0.0)
            h = 0 if hidden_lambda == 0.0 else int(rng.poisson(hidden_lambda))
            # At h=0 the adversarial process is absent, not an empty
            # transaction. Keeping this branch side-effect free is required
            # for the counterfactual null test.
            if h <= 0:
                continue
            ext = int(rng.choice(topo.external))
            dstz = int(rng.choice(topo.hlr))
            r1 = topo.route(ext, nid)
            r2 = topo.route(nid, dstz)
            if not r1.paths or not r2.paths:
                continue
            f_h = Flow(ext, dstz, Service.SMS_MT, 0.0)
            path = tuple(list(r1.paths[0]) + list(r2.paths[0])[1:])
            pending.append((f_h, path, h))
            hidden_flow_ids[id(f_h)] = nid
            hidden[nid] = {"msgs": 0, "dest": {}, "links": {},
                           "mtype": np.zeros(len(MSG_TYPES), np.int64),
                           "dlg": (0, 0), "ep": ep, "path": path, "n": h,
                           "dst": dstz}
            tmpl = SERVICE_TEMPLATE[Service.SMS_MT]
            bits = self._service_bits(tmpl)
            for u, v in zip(path[:-1], path[1:]):
                k = (min(u, v), max(u, v))
                offered[k] = offered.get(k, 0.0) + h * bits / dt

        # --- состояние линков из ФАКТИЧЕСКОЙ нагрузки
        lstate = {}
        for k, lsx in topo.linksets.items():
            lstate[k] = link_state(offered.get(k, 0.0), lsx, self.mean_msg_bits)

        # Outcome-conditioned message accounting for abstract transaction cohorts.
        # All cohorts attempt an initiation. Only closed cohorts emit the full
        # remaining template. Failed cohorts may emit a partial suffix.
        # Offered full-template workload remains a queue-load approximation.
        stage_rng=np.random.default_rng([cfg.seed,t,12037])
        for f,path,cnt in pending:
            hop_keys=[(min(u,v),max(u,v)) for u,v in zip(path[:-1],path[1:])]
            p_ok=float(np.prod([1.0-lstate[k][2] for k in hop_keys]))**2
            p_ok=1.0-(1.0-p_ok)**(1+self.max_retry)
            closed=int(rng.binomial(cnt,np.clip(p_ok,0.,1.)))
            h=hidden.get(hidden_flow_ids.get(id(f)))
            # The abstract stage model applies identically to normal and
            # anomalous flows; no attack label is an input.
            account_cohort(states,path,f.service,cnt,closed,stage_rng,h)
        for st in states.values(): st.check_identities()
        meta = {"lstate": lstate, "offered": offered, "failed": failed_links}
        return states, hidden, meta

    # ---- отчёты ------------------------------------------------------------

def generate(seed,mode,days=.16,zero=False,clean=False,benign_only=False):
    cfg=SimConfig(seed=seed,days=days,key_custody=KeyCustody.NODE_SOFTWARE)
    book=RngBook(seed)
    sim=(OutcomeWorld if mode=="outcome" else AggregateWorld)(cfg,book)
    reps=1 if days<.15 else 2
    effects,comp=schedule(sim,cfg,np.random.default_rng(seed+19203),reps,zero)
    normal=make_episodes(sim,cfg,book["episode"],n_compromise=0,n_ext=0,n_normal=6*reps)
    if clean: effects,comp=[],[]
    # Independent negative-control worlds: same service-family load distribution
    # and same scenario generator, independent seed, no compromise, benign labels.
    # This intentionally tests ambiguity of aggregate volume effects; it does not
    # assert that real malicious and legitimate transactions are identical.
    if benign_only:
        effects=[e for e in effects if e["class"] in FAMILIES]; comp=[]
    obs=[]; truth=[]
    totals=dict(polls=0,reports=0,accepted=0,missing=0,invalid=0,offered_bytes=0,
        accepted_bytes=0,signaling_link_bytes=0.,accounted_link_messages=0)
    for t in range(cfg.n_intervals):
        states,hidden,meta=sim.aggregate_step(t,normal+comp,effects)
        reports=sim.step_reports(t,states,hidden,meta,normal+comp)
        rec,status,stats=deliver(reports,seed,t,np.random.default_rng([seed,t,981]),.01)
        for k,v in stats.items(): totals[k]+=v
        totals["signaling_link_bytes"]+=sum(meta["offered"].values())*cfg.interval_s/8
        totals["accounted_link_messages"]+=sum(sum(s.link_out.values()) for s in states.values())
        active={e["nid"]:e for e in effects if e["start"]<=t<=e["end"]}
        for nid in sim.topo.slaves:
            obs.append(dict(nid=nid,t=t,zone=sim.topo.nodes[nid].zone,
                role=sim.topo.nodes[nid].ntype.value,**observe_v6(nid,rec,status,sim.topo)))
            e=active.get(nid)
            truth.append(dict(nid=nid,t=t,attack=e["class"] if e and not benign_only else "normal",
                episode=e["id"] if e and not benign_only else -1,
                matched_benign=int(bool(e) and benign_only)))
    d=features_v6(pd.DataFrame(obs)).merge(pd.DataFrame(truth),on=["nid","t"],validate="one_to_one")
    d["world"]=seed
    totals["days"]=cfg.n_intervals/INTERVALS_PER_DAY
    totals["application_to_offered_link_bytes"]=totals["offered_bytes"]/max(totals["signaling_link_bytes"],1.)
    return d,effects,totals

# Add separately generated benign matched worlds to calibration and test.
# Same attack schedule distribution; disjoint random seeds and roles.
PLAN12=PLAN+[("validation_benign",12),("test_benign",13)]
def shard(args):
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    if args.shard==0:
        a,_,_=generate(args.seed+9900,args.mode,.10,zero=True)
        b,_,_=generate(args.seed+9900,args.mode,.10,clean=True)
        np.testing.assert_allclose(a[ALL6],b[ALL6],atol=1e-10,rtol=0,equal_nan=True)
        (out/"null_gate.json").write_text(json.dumps(dict(passed=True,seed=args.seed+9900)))
    for role,i in PLAN12:
        if i%4!=args.shard: continue
        seed=args.seed+1009*i; start=time.perf_counter()
        d,ep,s=generate(seed,args.mode,benign_only=role.endswith("_benign")); d=trim(d)
        expected={"normal"} if role.endswith("_benign") else set(CLASSES)
        if set(d.attack)!=expected: raise AssertionError("class coverage")
        f=out/f"world_{i}.pkl.gz"; d.to_pickle(f,compression="gzip")
        pd.DataFrame(ep).to_csv(out/f"episodes_{role}_{seed}.csv",index=False)
        (out/f"world_{i}.json").write_text(json.dumps(dict(role=role,index=i,seed=seed,
             mode=args.mode,rows=len(d),sha256=sha(f),overhead=s,
             seconds=time.perf_counter()-start)))
        print("world",role,seed,"seconds",round(time.perf_counter()-start,2),flush=True)

def load(args):
    src=Path(args.input)
    n=json.loads((src/"null_gate.json").read_text())
    if not n["passed"] or n["seed"]!=args.seed+9900: raise ValueError("null gate")
    data={k:[] for k,_ in PLAN12}; overhead=[]
    for role,i in PLAN12:
        f=src/f"world_{i}.pkl.gz"; m=json.loads((src/f"world_{i}.json").read_text())
        if (m["role"],m["index"],m["seed"],m["mode"],m["sha256"])!=(role,i,args.seed+1009*i,args.mode,sha(f)):
            raise ValueError("manifest mismatch")
        d=pd.read_pickle(f,compression="gzip")
        expected={"normal"} if role.endswith("_benign") else set(CLASSES)
        if len(d)!=m["rows"] or set(d.world)!={m["seed"]} or set(d.attack)!=expected:
            raise ValueError("world mismatch")
        data[role].append(d)
        overhead.append(dict(role=role,seed=m["seed"],**m["overhead"]))
    return {k:stack(v).sort_values(["world","nid","t"]).reset_index(drop=True) for k,v in data.items()},overhead

def matched_metrics(d,score,name,budget):
    q=d[["world","nid","t","matched_benign"]].assign(fire=np.asarray(score)>0)
    days=q[["world","t"]].drop_duplicates().shape[0]/INTERVALS_PER_DAY
    take=q.matched_benign.eq(1)
    return dict(model=name,budget=budget,windows=int(take.sum()),
        alarms=int(q.loc[take,"fire"].sum()),matched_window_fpr=float(q.loc[take,"fire"].mean()),
        all_benign_alarms_per_day=float(q.fire.sum()/days))

def assess(args):
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    data,over=load(args)
    train,fusion,ref=(data[k] for k in ["train","fusion","reference"])
    # Hard-negative validation world contributes to the same pooled budget.
    val=stack([data["validation"],data["validation_benign"]])
    test=data["test"]; hard=data["test_benign"]
    m,roles=train_v6(train,args.seed,70,20); g,gc=graph_compromise(fusion,args.seed+19)
    def v7(x):
        s=score_experts(x,m,roles); s["zone"]=graph_score(x,g,gc); return s
    frames={}
    for k,d in data.items(): frames[k],cols=profiles(d,False)
    tr,fu,re,te,ha=(frames[k] for k in ["train","fusion","reference","test","test_benign"])
    va=stack([frames["validation"],frames["validation_benign"]])
    lm,lr=train_local(tr,cols,args.seed,"extra"); gm,gcols=fit_graph(fu,False,args.seed+19)
    def v9(x): return raw(x,lm,lr,gm,gcols,False)
    rows=[]; controls=[]
    for name,fn,r,w,t,h in [("v7_fixed",v7,ref,val,test,hard),("v9_extra",v9,re,va,te,ha)]:
        st=role_reference(r,fn(r))
        vs=prepare(w,fn(w),st,"median",False)
        ts=prepare(t,fn(t),st,"median",False)
        hs=prepare(h,fn(h),st,"median",False)
        for budget in (20.,100.,500.):
            for variant,alloc in [(name,ALLOC),(name+"_local",{k:.2 for k in CLASSES[1:]})]:
                cuts=cuts_for(w,vs,budget,alloc)
                rows+=evaluate_budget(w,margin(vs,cuts),0.,variant,"validation",budget)
                rows+=evaluate_budget(t,margin(ts,cuts),0.,variant,"test",budget)
                controls.append(matched_metrics(h,margin(hs,cuts),variant,budget))
    res=pd.DataFrame(rows); res.to_csv(out/"metrics.csv",index=False)
    pd.DataFrame(controls).to_csv(out/"matched_benign.csv",index=False)
    pd.DataFrame(over).to_csv(out/"kie_overhead.csv",index=False)
    (out/"manifest.json").write_text(json.dumps(dict(seed=args.seed,mode=args.mode,
        plan=PLAN12,primary_budget=BUDGET,confirmation_windows=K,
        scope="generic outcome-conditioned cohort accounting; external proxy intent not identifiable"),indent=2))
    print("PRIMARY",flush=True)
    print(res[(res.split=="test")&(res.budget==20.)].to_string(index=False),flush=True)
    print("MATCHED BENIGN",flush=True)
    print(pd.DataFrame(controls).to_string(index=False),flush=True)
    print("BUDGET SENSITIVITY",flush=True)
    print(res[(res.split=="test")&(res.attack=="all")].to_string(index=False),flush=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--stage",choices=["generate","evaluate"],required=True)
    p.add_argument("--seed",type=int,default=256101)
    p.add_argument("--mode",choices=["legacy","outcome"],required=True)
    p.add_argument("--shard",type=int,default=0)
    p.add_argument("--input",default="worlds")
    p.add_argument("--out",required=True)
    a=p.parse_args()
    if a.stage=="generate": shard(a)
    else: assess(a)
if __name__=="__main__": main()
