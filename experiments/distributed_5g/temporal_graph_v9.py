"""Temporal-graph V9: specialized pair/k4/k5 rescue with supported fusion.

V9 is a prospective successor to frozen V8.  V8 confirmation seeds 90701-90706
are consumed development evidence only.

The V8 failure exposed two distinct mechanisms:
1) weak k=2 episodes can form a stable learned-probability pair even when a
   single benign third node temporarily narrows the top2/top3 boundary;
2) independent OR fusion of V4 and the group branch can exceed the operational
   false-positive budget even when each branch is individually acceptable.

V9 therefore uses three specialized permutation-equivariant rescue channels:
* k=2: robust pair contrast against the median of the remaining six zone members,
  accumulated while the same unordered pair persists;
* k=4: the original top-k boundary statistic with a conservative frozen floor;
* k=5: the original top-k boundary statistic.

The pair rescue is accepted only when the unchanged V4 raw evidence reaches a
sub-threshold support fraction.  V4 itself keeps its original direct threshold.
No adjacency or absolute within-zone numeric ordering is used.
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from experiments.distributed_5g.temporal_graph_v4 import CommonModeResidualTransformer, slot_number


class PairMedianStableRescue:
    """Persistent unordered top-2 rescue using robust contrast to the other six."""

    def __init__(self, model, history=8, decay=0.75):
        self.model=copy.deepcopy(model)
        self.history_size=int(history)
        self.decay=float(decay)
        if not 0 <= self.decay < 1:
            raise ValueError("decay must be in [0,1)")
        self.transformer=CommonModeResidualTransformer(history=self.history_size)
        self.reset()

    def get_params(self, deep=True):
        return {
            "history":self.history_size,
            "decay":self.decay,
            "candidate":"top-2 learned-probability set",
            "instantaneous_score":"(mean(top2)-median(rest6))*min(top2)/(1+4*std(top2))",
            "state":"evidence retained only while the same unordered pair persists",
            "membership_change":"hard reset to current instantaneous score",
            "uses_adjacency":False,
            "uses_numeric_nf_order":False,
            "permutation_equivariant_within_zone":True,
        }

    def reset(self):
        self.transformer.reset()
        self.state={}
        self.current_run=None

    def _window_scores(self, part, probability):
        part=pd.DataFrame(part).copy()
        probability=np.asarray(probability,dtype=float)
        if len(part)!=32 or part.nf_id.nunique()!=32 or len(probability)!=32:
            raise ValueError("V9 pair rescue requires exactly 32 NF probabilities")
        if not np.isfinite(probability).all():
            raise ValueError("Nonfinite learned probabilities")
        part["_p"]=probability
        output={str(nf):0.0 for nf in part.nf_id.astype(str)}
        for zone in range(4):
            slots=part.nf_id.map(slot_number)
            z=part.loc[(slots//8).eq(zone),["nf_id","_p"]].copy()
            if len(z)!=8:
                raise ValueError(f"Expected 8 NF in zone {zone}")
            z=z.sort_values("_p",ascending=False)
            p=z["_p"].to_numpy(float)
            members=tuple(sorted(z.nf_id.iloc[:2].astype(str).tolist()))
            top=p[:2]
            contrast=max(0.0,float(np.mean(top))-float(np.median(p[2:])))
            coherence=1.0/(1.0+4.0*float(np.std(top)))
            instantaneous=contrast*float(np.min(top))*coherence
            previous_members,previous_evidence=self.state.get(zone,((),0.0))
            evidence=(self.decay*float(previous_evidence)+instantaneous
                      if previous_members==members else instantaneous)
            self.state[zone]=(members,float(evidence))
            for nf in members:
                output[nf]=float(evidence)
        return output

    def decision_function(self, frame):
        frame=pd.DataFrame(frame).copy()
        required={"run_id","window_id","nf_id","request_count"}
        missing=required-set(frame.columns)
        if missing:
            raise ValueError(f"Missing V9 pair inputs: {sorted(missing)}")
        if frame.run_id.nunique()!=1:
            raise ValueError("V9 pair scoring call must contain one acquisition run")
        run_id=str(frame.run_id.iloc[0])
        if self.current_run is None:
            self.current_run=run_id
        elif self.current_run!=run_id:
            self.reset();self.current_run=run_id

        graph=self.transformer.transform_online(frame)
        probability=self.model.predict_proba(graph)[:,1]
        out=np.zeros(len(frame),dtype=float)
        absolute={idx:pos for pos,idx in enumerate(frame.index)}
        for _,part in frame.groupby("window_id",sort=False):
            pos=np.asarray([absolute[idx] for idx in part.index],dtype=int)
            scores=self._window_scores(part,probability[pos])
            for idx,row in part.iterrows():
                out[absolute[idx]]=scores[str(row.nf_id)]
        return out

    def clean_copy(self):
        clone=copy.deepcopy(self);clone.reset();return clone


class FixedKStableRescue:
    """Persistent unordered fixed-k rescue using the V7/V8 boundary statistic."""

    def __init__(self, model, k, history=8, decay=0.25):
        self.model=copy.deepcopy(model)
        self.k=int(k)
        self.history_size=int(history)
        self.decay=float(decay)
        if self.k not in (4,5):
            raise ValueError("V9 fixed-k rescue is defined only for k=4 or k=5")
        if not 0 <= self.decay < 1:
            raise ValueError("decay must be in [0,1)")
        self.transformer=CommonModeResidualTransformer(history=self.history_size)
        self.reset()

    def get_params(self,deep=True):
        return {
            "k":self.k,"history":self.history_size,"decay":self.decay,
            "candidate":"top-k learned-probability set",
            "instantaneous_score":"(p_k-p_k+1)*mean(top-k p)/(1+4*std(top-k p))",
            "membership_change":"hard reset to current instantaneous score",
            "uses_adjacency":False,"uses_numeric_nf_order":False,
            "permutation_equivariant_within_zone":True,
        }

    def reset(self):
        self.transformer.reset();self.state={};self.current_run=None

    def _window_scores(self,part,probability):
        part=pd.DataFrame(part).copy();probability=np.asarray(probability,dtype=float)
        if len(part)!=32 or part.nf_id.nunique()!=32 or len(probability)!=32:
            raise ValueError("V9 fixed-k rescue requires exactly 32 NF probabilities")
        part["_p"]=probability
        output={str(nf):0.0 for nf in part.nf_id.astype(str)}
        for zone in range(4):
            slots=part.nf_id.map(slot_number)
            z=part.loc[(slots//8).eq(zone),["nf_id","_p"]].copy().sort_values("_p",ascending=False)
            if len(z)!=8:
                raise ValueError(f"Expected 8 NF in zone {zone}")
            p=z["_p"].to_numpy(float)
            top=p[:self.k]
            gap=max(0.0,float(p[self.k-1]-p[self.k]))
            coherence=1.0/(1.0+4.0*float(np.std(top)))
            instantaneous=gap*float(np.mean(top))*coherence
            members=tuple(sorted(z.nf_id.iloc[:self.k].astype(str).tolist()))
            previous_members,previous_evidence=self.state.get(zone,((),0.0))
            evidence=(self.decay*float(previous_evidence)+instantaneous
                      if previous_members==members else instantaneous)
            self.state[zone]=(members,float(evidence))
            for nf in members:
                output[nf]=float(evidence)
        return output

    def decision_function(self,frame):
        frame=pd.DataFrame(frame).copy()
        if frame.run_id.nunique()!=1:
            raise ValueError("V9 fixed-k scoring call must contain one acquisition run")
        run_id=str(frame.run_id.iloc[0])
        if self.current_run is None:self.current_run=run_id
        elif self.current_run!=run_id:
            self.reset();self.current_run=run_id
        graph=self.transformer.transform_online(frame)
        probability=self.model.predict_proba(graph)[:,1]
        out=np.zeros(len(frame),dtype=float);absolute={idx:pos for pos,idx in enumerate(frame.index)}
        for _,part in frame.groupby("window_id",sort=False):
            pos=np.asarray([absolute[idx] for idx in part.index],dtype=int)
            scores=self._window_scores(part,probability[pos])
            for idx,row in part.iterrows():out[absolute[idx]]=scores[str(row.nf_id)]
        return out

    def clean_copy(self):
        clone=copy.deepcopy(self);clone.reset();return clone
