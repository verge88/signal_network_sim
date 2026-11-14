"""Temporal-graph V7: stable-set, FPR-constrained group aggregation.

V7 keeps the learned V4 residual-temporal probability model and the V6
permutation-invariant top-k candidate definition, but changes temporal evidence
from per-NF accumulation to per-(zone,k,unordered-member-set) accumulation.

The V6 prospective experiment (seeds 90501-90506) is consumed development
evidence for V7. It showed that most excess false positives were short post-attack
tails from per-node evidence after the coordinated set stopped being stable.
V7 therefore resets a candidate's accumulated evidence whenever the unordered
top-k membership changes. No adjacency or numeric NF ordering is used.

Runtime features remain label-free. NF identities are used only as opaque keys to
test whether the same unordered set persists across adjacent windows.
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from experiments.distributed_5g.temporal_graph_v4 import CommonModeResidualTransformer, slot_number


class StableSetGroupDetector:
    """Permutation-equivariant variable-k evidence with membership-change reset."""

    def __init__(self, model, history=8, min_group=2, max_group=5, decay=0.25):
        self.model=copy.deepcopy(model)
        self.history_size=int(history)
        self.min_group=int(min_group)
        self.max_group=int(max_group)
        self.decay=float(decay)
        if not 2 <= self.min_group <= self.max_group <= 7:
            raise ValueError("group sizes must satisfy 2 <= min_group <= max_group <= 7")
        if not 0 <= self.decay < 1:
            raise ValueError("decay must be in [0,1)")
        self.transformer=CommonModeResidualTransformer(history=self.history_size)
        self.reset()

    def get_params(self, deep=True):
        return {
            "history":self.history_size,
            "min_group":self.min_group,
            "max_group":self.max_group,
            "decay":self.decay,
            "candidate":"top-k learned-probability boundary",
            "group_score":"(p_k-p_k+1)*mean(top-k p)/(1+4*std(top-k p))",
            "state":"evidence is retained only while the same unordered top-k set persists",
            "membership_change":"hard reset to current instantaneous score",
            "uses_adjacency":False,
            "uses_numeric_nf_order":False,
            "permutation_equivariant_within_zone":True,
        }

    def reset(self):
        self.transformer.reset()
        self.group_state={}
        self.current_run=None

    def _window_scores(self, part, probability):
        part=pd.DataFrame(part).copy()
        probability=np.asarray(probability,dtype=float)
        if len(part)!=32 or part.nf_id.nunique()!=32 or len(probability)!=len(part):
            raise ValueError("V7 group detector requires exactly 32 NF probabilities")
        if not np.isfinite(probability).all():
            raise ValueError("Nonfinite learned probabilities")

        part["_p"]=probability
        output={str(nf):0.0 for nf in part.nf_id.astype(str)}
        for zone in range(4):
            slots=part.nf_id.map(slot_number)
            z=part.loc[(slots//8).eq(zone),["nf_id","_p"]].copy()
            if len(z)!=8:
                raise ValueError(f"Expected 8 NF in zone {zone}")
            p=z["_p"].to_numpy(float)
            ordered=np.sort(p)[::-1]

            for k in range(self.min_group,self.max_group+1):
                floor=float(ordered[k-1])
                outside=float(ordered[k])
                gap=max(0.0,floor-outside)
                key=(zone,k)
                if gap <= 0:
                    self.group_state[key]=((),0.0)
                    continue

                selected=p >= floor
                if int(selected.sum()) != k:
                    raise AssertionError("Positive probability boundary did not define k members")
                group=p[selected]
                members=tuple(sorted(z.loc[selected,"nf_id"].astype(str).tolist()))
                coherence=1.0/(1.0+4.0*float(np.std(group)))
                instantaneous=gap*float(np.mean(group))*coherence

                previous_members,previous_evidence=self.group_state.get(key,((),0.0))
                evidence=(self.decay*float(previous_evidence)+instantaneous
                          if previous_members==members else instantaneous)
                self.group_state[key]=(members,float(evidence))
                for nf in members:
                    output[nf]=max(output[nf],float(evidence))
        return output

    def decision_function(self, frame):
        frame=pd.DataFrame(frame).copy()
        required={"run_id","window_id","nf_id","request_count"}
        missing=required-set(frame.columns)
        if missing:
            raise ValueError(f"Missing V7 group inputs: {sorted(missing)}")
        if frame.run_id.nunique()!=1:
            raise ValueError("V7 group scoring call must contain one acquisition run")
        run_id=str(frame.run_id.iloc[0])
        if self.current_run is None:
            self.current_run=run_id
        elif self.current_run!=run_id:
            self.reset()
            self.current_run=run_id

        graph=self.transformer.transform_online(frame)
        probability=self.model.predict_proba(graph)[:,1]
        out=np.zeros(len(frame),dtype=float)
        positions={idx:pos for pos,idx in enumerate(frame.index)}
        absolute_positions={idx:pos for pos,idx in enumerate(frame.index)}

        for _,part in frame.groupby("window_id",sort=False):
            pos=np.asarray([absolute_positions[idx] for idx in part.index],dtype=int)
            scores=self._window_scores(part,probability[pos])
            for idx,row in part.iterrows():
                out[positions[idx]]=scores[str(row.nf_id)]
        return out

    def clean_copy(self):
        clone=copy.deepcopy(self)
        clone.reset()
        return clone
