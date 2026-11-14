"""Temporal-graph V6: permutation-invariant, variable-cardinality group aggregation.

V6 keeps the learned V4 residual-temporal probability model unchanged and adds a
new group branch that never uses NF adjacency or absolute slot identifiers.

For each eight-NF zone and every candidate cardinality k in [2, 5], the branch
looks for a positive probability boundary between the k-th and (k+1)-th ranked
nodes. Evidence is assigned to the whole top-k set and accumulated causally.
Consequently, a bijective relabeling of NF identifiers within a zone leaves the
score multiset unchanged, while coordinated groups of different sizes can be
represented by the same mechanism.

Runtime inputs remain the measured request-count stream transformed by the
already-defined V4 CommonModeResidualTransformer. Scenario labels, attack labels,
run IDs, window numbers, and NF numeric identifiers are not model features.
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from experiments.distributed_5g.temporal_graph_v4 import CommonModeResidualTransformer, slot_number


class PermutationInvariantGroupRescue:
    """Group evidence over learned V4 probabilities, invariant to within-zone IDs."""

    def __init__(self, model, history=8, min_group=2, max_group=5, decay=0.75):
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
            "aggregation":"top-k learned-probability boundary, k in inclusive range",
            "permutation_invariant_within_zone":True,
            "adjacency_used":False,
            "score":"(p_k-p_k+1)*mean(top-k p)/(1+4*std(top-k p))",
        }

    def reset(self):
        self.transformer.reset()
        self.evidence={}
        self.current_run=None

    def _window_scores(self, part, probability):
        """Return instantaneous group scores for one 32-NF window.

        The only use of slot numbers is to identify the pre-existing eight-NF zone
        partition. Ordering or adjacency inside a zone is never consulted.
        """
        part=pd.DataFrame(part).copy()
        probability=np.asarray(probability,dtype=float)
        if len(part)!=32 or part.nf_id.nunique()!=32 or len(probability)!=len(part):
            raise ValueError("V6 group rescue requires exactly 32 NF probabilities")
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
            node_score=np.zeros(len(z),dtype=float)
            for k in range(self.min_group,self.max_group+1):
                floor=float(ordered[k-1])
                outside=float(ordered[k])
                gap=max(0.0,floor-outside)
                if gap <= 0:
                    continue
                members=p >= floor
                # Positive boundary gap guarantees exactly k members even if there
                # are ties inside the selected group.
                if int(members.sum()) != k:
                    raise AssertionError("Positive probability boundary did not define k members")
                group=p[members]
                coherence=1.0/(1.0+4.0*float(np.std(group)))
                score=gap*float(np.mean(group))*coherence
                node_score[members]=np.maximum(node_score[members],score)
            for nf,score in zip(z.nf_id.astype(str),node_score):
                output[nf]=float(score)
        return output

    def decision_function(self, frame):
        frame=pd.DataFrame(frame).copy()
        required={"run_id","window_id","nf_id","request_count"}
        missing=required-set(frame.columns)
        if missing:
            raise ValueError(f"Missing V6 group inputs: {sorted(missing)}")
        if frame.run_id.nunique()!=1:
            raise ValueError("V6 group scoring call must contain one acquisition run")
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

        # transform_online preserves row order. Slice the probability vector using
        # the original frame positions for each window.
        absolute_positions={idx:pos for pos,idx in enumerate(frame.index)}
        for _,part in frame.groupby("window_id",sort=False):
            pos=np.asarray([absolute_positions[idx] for idx in part.index],dtype=int)
            scores=self._window_scores(part,probability[pos])
            for idx,row in part.iterrows():
                nf=str(row.nf_id)
                value=self.decay*self.evidence.get(nf,0.0)+scores[nf]
                self.evidence[nf]=value
                out[positions[idx]]=value
        return out

    def clean_copy(self):
        clone=copy.deepcopy(self)
        clone.reset()
        return clone
