"""Frozen cyclic-S5 rescue motif for temporal-graph V5.

Selected on consumed seeds 89901-90203:
- zone residual IQR <= 1.0
- temporal decay 0.8
- evidence threshold 2.0

The motif is label-free at runtime. It subtracts the per-zone median request count,
finds a contiguous three-node group whose every member is elevated above all
nodes outside the group, and accumulates that evidence causally.
"""
from collections import deque
import copy

import numpy as np
import pandas as pd


class CyclicTripleRescue:
    def __init__(self, iqr_limit=1.0, decay=0.8):
        self.iqr_limit=float(iqr_limit)
        self.decay=float(decay)
        if self.iqr_limit < 0:
            raise ValueError("iqr_limit must be nonnegative")
        if not 0 <= self.decay < 1:
            raise ValueError("decay must be in [0,1)")
        self.reset()

    def get_params(self, deep=True):
        return {
            "iqr_limit":self.iqr_limit,
            "decay":self.decay,
            "motif":"contiguous 3-node positive residual cluster after zone-median removal",
            "coherence":"1/(1+std(triple residuals))",
        }

    def reset(self):
        self.evidence={}
        self.current_run=None

    @staticmethod
    def _slot(nf_id):
        text=str(nf_id)
        if not text.startswith("slot-"):
            raise ValueError(f"Expected slot-N, got {text!r}")
        value=int(text.split("-",1)[1])
        if not 0 <= value < 32:
            raise ValueError(value)
        return value

    def _window_scores(self, part):
        if len(part)!=32 or part.nf_id.nunique()!=32:
            raise ValueError("Cyclic rescue requires exactly 32 NF rows per window")
        output={}
        for zone in range(4):
            z=part[part.nf_id.map(self._slot).floordiv(8).eq(zone)].copy()
            z["_slot"]=z.nf_id.map(self._slot)%8
            z=z.sort_values("_slot")
            if len(z)!=8:
                raise ValueError(f"Expected 8 NF in zone {zone}")
            counts=z.request_count.to_numpy(float)
            common=float(np.median(counts))
            residual=counts-common
            q75,q25=np.quantile(residual,[.75,.25])
            zone_iqr=float(q75-q25)
            node_score=np.zeros(8,dtype=float)
            if zone_iqr <= self.iqr_limit:
                for start in range(6):
                    triple=residual[start:start+3]
                    rest=np.concatenate([residual[:start],residual[start+3:]])
                    separation=float(np.min(triple)-max(0.0,float(np.max(rest))))
                    coherence=1.0/(1.0+float(np.std(triple)))
                    score=max(0.0,separation)*coherence
                    node_score[start:start+3]=np.maximum(node_score[start:start+3],score)
            for nf,score in zip(z.nf_id.astype(str),node_score):
                output[nf]=float(score)
        return output

    def decision_function(self, frame):
        frame=pd.DataFrame(frame).copy()
        required={"run_id","window_id","nf_id","request_count"}
        missing=required-set(frame.columns)
        if missing:
            raise ValueError(f"Missing cyclic rescue inputs: {sorted(missing)}")
        if frame.run_id.nunique()!=1:
            raise ValueError("Cyclic rescue scoring call must contain one acquisition run")
        run_id=str(frame.run_id.iloc[0])
        if self.current_run is None:
            self.current_run=run_id
        elif self.current_run!=run_id:
            self.reset()
            self.current_run=run_id

        out=np.zeros(len(frame),dtype=float)
        positions={idx:pos for pos,idx in enumerate(frame.index)}
        for _,part in frame.groupby("window_id",sort=False):
            scores=self._window_scores(part)
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
