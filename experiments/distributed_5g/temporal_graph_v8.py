"""Temporal-graph V8: pair-sensitive stable-set aggregation.

V8 is a prospective successor to frozen V7.  The V7 confirmation runs
90601-90606 are consumed development evidence for V8 and are never treated as
V8 confirmation.

The V7 group branch already eliminated the V6 post-episode false-positive tail,
but k=2 attacks had markedly lower window recall and one missed episode.  V8
changes only the temporal retention coefficient for the k=2 unordered candidate
set.  Candidate construction, learned probabilities, permutation equivariance,
k=3..5 behavior, and normal-only threshold calibration remain unchanged.

For k=2, evidence is accumulated with decay 0.55 while the same unordered pair
persists.  For k=3..5, the V7 decay 0.25 is retained.  Membership change still
causes a hard reset.  This strengthens persistent pair evidence without lowering
the alarm threshold or introducing adjacency / numeric-ID features.
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from experiments.distributed_5g.temporal_graph_v4 import CommonModeResidualTransformer, slot_number


class PairSensitiveStableSetDetector:
    """Variable-k stable-set detector with a dedicated persistent-pair decay."""

    def __init__(self, model, history=8, min_group=2, max_group=5,
                 pair_decay=0.55, group_decay=0.25):
        self.model=copy.deepcopy(model)
        self.history_size=int(history)
        self.min_group=int(min_group)
        self.max_group=int(max_group)
        self.pair_decay=float(pair_decay)
        self.group_decay=float(group_decay)
        if not 2 <= self.min_group <= self.max_group <= 7:
            raise ValueError("group sizes must satisfy 2 <= min_group <= max_group <= 7")
        for name,value in (("pair_decay",self.pair_decay),("group_decay",self.group_decay)):
            if not 0 <= value < 1:
                raise ValueError(f"{name} must be in [0,1)")
        self.transformer=CommonModeResidualTransformer(history=self.history_size)
        self.reset()

    def get_params(self, deep=True):
        return {
            "history":self.history_size,
            "min_group":self.min_group,
            "max_group":self.max_group,
            "pair_decay":self.pair_decay,
            "group_decay":self.group_decay,
            "candidate":"top-k learned-probability boundary",
            "group_score":"(p_k-p_k+1)*mean(top-k p)/(1+4*std(top-k p))",
            "state":"evidence retained only while the same unordered top-k set persists",
            "membership_change":"hard reset to current instantaneous score",
            "k2_specialization":"pair_decay only; score and threshold policy unchanged",
            "uses_adjacency":False,
            "uses_numeric_nf_order":False,
            "permutation_equivariant_within_zone":True,
        }

    def reset(self):
        self.transformer.reset()
        self.group_state={}
        self.current_run=None

    def _decay_for_k(self,k):
        return self.pair_decay if int(k)==2 else self.group_decay

    def _window_scores(self, part, probability):
        part=pd.DataFrame(part).copy()
        probability=np.asarray(probability,dtype=float)
        if len(part)!=32 or part.nf_id.nunique()!=32 or len(probability)!=len(part):
            raise ValueError("V8 group detector requires exactly 32 NF probabilities")
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
                decay=self._decay_for_k(k)
                evidence=(decay*float(previous_evidence)+instantaneous
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
            raise ValueError(f"Missing V8 group inputs: {sorted(missing)}")
        if frame.run_id.nunique()!=1:
            raise ValueError("V8 group scoring call must contain one acquisition run")
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
