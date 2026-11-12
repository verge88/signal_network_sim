"""Temporal-graph V3 runtime detector.

The detector learns S5-vs-normal separation from consumed historical development
captures, up-weights hard benign synchrony, converts point probabilities into a
localized multi-node graph motif score, and accumulates that structural evidence
causally. A separate fresh normal calibration stream sets the final evidence
threshold before held-out prospective testing.
"""
from collections import deque
import copy

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from experiments.distributed_5g.temporal_graph_v2 import FEATURES, LocalSynchronyTransformer


def slot_number(nf_id):
    text=str(nf_id)
    if not text.startswith("slot-"):
        raise ValueError(f"Expected slot-N, got {text!r}")
    value=int(text.split("-",1)[1])
    if not 0 <= value < 32:
        raise ValueError(value)
    return value


class TemporalGraphV3Detector:
    def __init__(self, seed=7143, history=8, cluster_order=3, decay=0.9,
                 hard_negative_quantile=0.98, hard_negative_weight=8.0):
        self.seed=int(seed)
        self.history_size=int(history)
        self.cluster_order=int(cluster_order)
        self.decay=float(decay)
        self.hard_negative_quantile=float(hard_negative_quantile)
        self.hard_negative_weight=float(hard_negative_weight)
        if self.cluster_order not in (2,3):
            raise ValueError("cluster_order must be 2 or 3")
        if not 0 < self.decay < 1:
            raise ValueError("decay must be in (0,1)")
        self.transformer=LocalSynchronyTransformer(history=self.history_size)
        self.model=None
        self.training_meta={}
        self.reset()

    def get_params(self, deep=True):
        return {
            "seed":self.seed,
            "history":self.history_size,
            "cluster_order":self.cluster_order,
            "decay":self.decay,
            "hard_negative_quantile":self.hard_negative_quantile,
            "hard_negative_weight":self.hard_negative_weight,
            "estimator":"StandardScaler + LogisticRegression",
            "features":list(FEATURES),
            "structural_score":"min(node_probability, zone_kth_probability) * max(0, zone_kth_probability-zone_median_probability) * clipped(1+localized_corr_contrast+triangle_contrast)",
            "temporal_evidence":"E_t = decay * E_(t-1) + structural_score",
        }

    def _new_model(self):
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0,class_weight="balanced",max_iter=1000,
                               random_state=self.seed))

    def fit_development(self, datasets, extra_normal=None):
        frames=[]
        for frame,labels in datasets:
            graph=LocalSynchronyTransformer(self.history_size).transform(frame)
            item=pd.DataFrame(graph,columns=FEATURES)
            item["attack"]=labels.attack.astype(int).to_numpy()
            item["scenario"]=labels.scenario.astype(str).to_numpy()
            frames.append(item)
        if extra_normal is not None and len(extra_normal):
            graph=LocalSynchronyTransformer(self.history_size).transform(extra_normal)
            item=pd.DataFrame(graph,columns=FEATURES)
            item["attack"]=0
            item["scenario"]="N1"
            frames.append(item)

        train=pd.concat(frames,ignore_index=True)
        target=train.scenario.eq("S5")
        keep=target | train.attack.eq(0)
        x=train.loc[keep,list(FEATURES)].to_numpy(float)
        y=target.loc[keep].astype(int).to_numpy()
        if not y.any() or y.all():
            raise ValueError("V3 development requires S5 and normal examples")

        stage1=self._new_model()
        stage1.fit(x,y)
        probability=stage1.predict_proba(x)[:,1]
        normal=probability[y==0]
        hard_cut=float(np.quantile(normal,self.hard_negative_quantile))
        sample_weight=np.ones(len(y),dtype=float)
        hard=(y==0)&(probability>=hard_cut)
        sample_weight[hard]=self.hard_negative_weight

        self.model=self._new_model()
        self.model.fit(x,y,logisticregression__sample_weight=sample_weight)
        self.training_meta={
            "training_rows":int(len(y)),
            "s5_rows":int(y.sum()),
            "hard_negative_rows":int(hard.sum()),
            "hard_negative_cut":hard_cut,
        }
        self.reset()
        return self

    def reset(self):
        self.transformer.reset()
        self.evidence={}
        self.current_run=None

    def _structural(self, frame, probability, graph):
        metadata=frame[["run_id","window_id","nf_id"]].reset_index(drop=True).copy()
        metadata["probability"]=np.asarray(probability,dtype=float)
        metadata["localized_corr_contrast"]=graph[:,FEATURES.index("localized_corr_contrast")]
        metadata["triangle_contrast"]=graph[:,FEATURES.index("triangle_contrast")]
        metadata["_pos"]=np.arange(len(metadata))
        output=np.zeros(len(metadata),dtype=float)
        for (_,window_id),window in metadata.groupby(["run_id","window_id"],sort=False):
            for zone in range(4):
                slots=window.nf_id.map(slot_number)
                part=window[(slots//8).eq(zone)]
                if len(part)!=8:
                    raise ValueError(f"Expected 8 NF in zone {zone}, window {window_id}")
                p=part.probability.to_numpy(float)
                kth=float(np.sort(p)[-self.cluster_order])
                median=float(np.median(p))
                cluster_contrast=max(0.0,kth-median)
                local=np.maximum(0.,part.localized_corr_contrast.to_numpy(float))
                triangle=np.maximum(0.,part.triangle_contrast.to_numpy(float))
                motif=np.clip(1.0+local+triangle,1.0,3.0)
                value=np.minimum(p,kth)*cluster_contrast*motif
                output[part._pos.to_numpy(int)]=value
        return output

    def decision_function(self, frame):
        if self.model is None:
            raise ValueError("TemporalGraphV3Detector must be fitted")
        frame=pd.DataFrame(frame).copy()
        if frame.run_id.nunique()!=1:
            raise ValueError("V3 scoring call must contain one acquisition run")
        run_id=str(frame.run_id.iloc[0])
        if self.current_run is None:
            self.current_run=run_id
        elif self.current_run!=run_id:
            self.reset()
            self.current_run=run_id

        graph=self.transformer.transform_online(frame)
        probability=self.model.predict_proba(graph)[:,1]
        structural=self._structural(frame,probability,graph)
        output=np.zeros(len(frame),dtype=float)
        for pos,nf_id in enumerate(frame.nf_id.astype(str).to_numpy()):
            value=self.decay*self.evidence.get(nf_id,0.0)+float(structural[pos])
            self.evidence[nf_id]=value
            output[pos]=value
        return output

    def clean_copy(self):
        clone=copy.deepcopy(self)
        clone.reset()
        return clone
