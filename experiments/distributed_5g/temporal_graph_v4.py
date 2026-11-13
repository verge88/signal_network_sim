"""Temporal-graph V4: common-mode rejection for S5 under cyclic load.

V4 removes the instantaneous zone-wide workload component before temporal
correlations are formed. This makes the representation insensitive to legitimate
common cyclic load while preserving a localized three-node residual waveform.
Runtime features contain no scenario, regime, window number, run ID, or labels.
"""
from collections import deque
import copy

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURES_V4=(
    "residual_level",
    "residual_delta",
    "residual_mean8",
    "residual_std8",
    "top2_residual_corr_mean",
    "top2_residual_corr_min",
    "median_residual_peer_corr",
    "localized_residual_corr_contrast",
    "residual_triangle_corr",
    "residual_triangle_contrast",
    "residual_delta_similarity",
    "residual_sign_cluster",
    "zone_common_delta_abs",
    "zone_residual_iqr",
)


def slot_number(nf_id):
    text=str(nf_id)
    if not text.startswith("slot-"):
        raise ValueError(f"Expected slot-N, got {text!r}")
    value=int(text.split("-",1)[1])
    if not 0<=value<32:
        raise ValueError(value)
    return value


def zone_number(nf_id):
    return slot_number(nf_id)//8


def _corr(a,b):
    a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float)
    if len(a)<4 or np.std(a)<1e-9 or np.std(b)<1e-9:
        return 0.0
    value=float(np.corrcoef(a,b)[0,1])
    return value if np.isfinite(value) else 0.0


class CommonModeResidualTransformer:
    def __init__(self,history=8):
        self.history_size=int(history)
        self.reset()

    def reset(self):
        self.residual_history={}
        self.zone_common_history={}

    def _window(self,part):
        if len(part)!=32 or part.nf_id.nunique()!=32:
            raise ValueError("Expected exactly 32 NF rows")
        rows={str(r.nf_id):r for r in part.itertuples(index=False)}
        residual_now={}
        common_now={}
        zone_iqr={}
        for zone in range(4):
            members=[nf for nf in rows if zone_number(nf)==zone]
            counts=np.asarray([float(rows[nf].request_count) for nf in members],dtype=float)
            common=float(np.median(counts))
            common_now[zone]=common
            q75,q25=np.quantile(counts,[.75,.25])
            zone_iqr[zone]=float(q75-q25)
            for nf in members:
                residual_now[nf]=float(rows[nf].request_count)-common

        for nf,value in residual_now.items():
            self.residual_history.setdefault(nf,deque(maxlen=self.history_size+1)).append(value)
        for zone,value in common_now.items():
            self.zone_common_history.setdefault(zone,deque(maxlen=self.history_size+1)).append(value)

        residual_deltas={}
        residual_series={}
        for nf,h in self.residual_history.items():
            values=np.asarray(h,dtype=float)
            residual_series[nf]=values
            residual_deltas[nf]=np.diff(values) if len(values)>=2 else np.zeros(1,dtype=float)

        result={}
        for nf,row in rows.items():
            zone=zone_number(nf)
            peers=[p for p in rows if p!=nf and zone_number(p)==zone]
            own=residual_deltas[nf]
            correlations=[]
            for peer in peers:
                width=min(len(own),len(residual_deltas[peer]),self.history_size)
                value=_corr(own[-width:],residual_deltas[peer][-width:]) if width>=4 else 0.0
                correlations.append((peer,value))
            ordered=sorted(correlations,key=lambda item:item[1],reverse=True)
            top=ordered[:2]
            corr_values=np.asarray([value for _,value in correlations],dtype=float)
            top_values=np.asarray([value for _,value in top],dtype=float)
            median_corr=float(np.median(corr_values))
            top_mean=float(top_values.mean()) if len(top_values) else 0.0
            top_min=float(top_values.min()) if len(top_values) else 0.0

            triangle=0.0
            if len(top)==2:
                p1,p2=top[0][0],top[1][0]
                width=min(len(residual_deltas[p1]),len(residual_deltas[p2]),self.history_size)
                pair=_corr(residual_deltas[p1][-width:],residual_deltas[p2][-width:]) if width>=4 else 0.0
                triangle=float((top[0][1]+top[1][1]+pair)/3.0)

            current_delta=float(own[-1]) if len(own) else 0.0
            peer_current=np.asarray([float(residual_deltas[p][-1]) if len(residual_deltas[p]) else 0.0 for p in peers])
            gaps=np.abs(peer_current-current_delta)
            nearest=np.sort(gaps)[:2] if len(gaps)>=2 else gaps
            similarity=float(np.exp(-nearest).mean()) if len(nearest) else 0.0

            sign=np.sign(current_delta)
            same=np.asarray([
                float(np.sign(value)==sign and abs(value)>1e-12 and abs(current_delta)>1e-12)
                for value in peer_current
            ],dtype=float)
            sign_cluster=float(np.sort(same)[-2:].mean()-np.median(same)) if len(same)>=2 else 0.0

            own_series=residual_series[nf]
            zone_history=self.zone_common_history[zone]
            common_delta=(float(zone_history[-1]-zone_history[-2]) if len(zone_history)>=2 else 0.0)
            result[nf]=np.asarray([
                float(residual_now[nf]),
                current_delta,
                float(np.mean(own_series[-self.history_size:])),
                float(np.std(own_series[-self.history_size:])),
                top_mean,
                top_min,
                median_corr,
                top_mean-median_corr,
                triangle,
                triangle-median_corr,
                similarity,
                sign_cluster,
                abs(common_delta),
                zone_iqr[zone],
            ],dtype=float)
        return result

    def _transform_one_run(self,frame,reset):
        frame=pd.DataFrame(frame).copy()
        required={"run_id","window_id","nf_id","request_count"}
        missing=required-set(frame)
        if missing:
            raise ValueError(f"Missing V4 inputs: {sorted(missing)}")
        if frame.run_id.nunique()!=1:
            raise ValueError("V4 online call must contain one acquisition run")
        if reset:
            self.reset()
        output=np.zeros((len(frame),len(FEATURES_V4)),dtype=float)
        positions={idx:pos for pos,idx in enumerate(frame.index)}
        for _,part in frame.groupby("window_id",sort=False):
            values=self._window(part)
            for idx,row in part.iterrows():
                output[positions[idx]]=values[str(row.nf_id)]
        if not np.isfinite(output).all():
            raise ValueError("Nonfinite V4 graph feature")
        return output

    def transform(self,frame):
        frame=pd.DataFrame(frame).copy()
        parts=[]
        for _,run in frame.groupby("run_id",sort=False):
            parts.append(self._transform_one_run(run,reset=True))
        return np.vstack(parts) if parts else np.empty((0,len(FEATURES_V4)))

    def transform_online(self,frame):
        return self._transform_one_run(frame,reset=False)


class TemporalGraphV4Detector:
    """Hard-negative logistic detector over common-mode residual graph features."""

    def __init__(self,seed=7143,history=8,cluster_order=2,decay=.75,
                 hard_negative_quantile=.98,hard_negative_weight=8.,
                 cyclic_positive_weight=3.,cyclic_normal_weight=2.):
        self.seed=int(seed)
        self.history_size=int(history)
        self.cluster_order=int(cluster_order)
        self.decay=float(decay)
        self.hard_negative_quantile=float(hard_negative_quantile)
        self.hard_negative_weight=float(hard_negative_weight)
        self.cyclic_positive_weight=float(cyclic_positive_weight)
        self.cyclic_normal_weight=float(cyclic_normal_weight)
        self.transformer=CommonModeResidualTransformer(history=history)
        self.model=None
        self.training_meta={}
        self.reset()

    def get_params(self,deep=True):
        return {
            "seed":self.seed,"history":self.history_size,
            "cluster_order":self.cluster_order,"decay":self.decay,
            "hard_negative_quantile":self.hard_negative_quantile,
            "hard_negative_weight":self.hard_negative_weight,
            "cyclic_positive_weight":self.cyclic_positive_weight,
            "cyclic_normal_weight":self.cyclic_normal_weight,
            "estimator":"StandardScaler + LogisticRegression",
            "features":list(FEATURES_V4),
            "common_mode":"per-window zone median removed before temporal correlations",
        }

    def _new_model(self):
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0,class_weight="balanced",max_iter=1000,
                               random_state=self.seed))

    def fit_development(self,datasets,extra_normal=None):
        frames=[]
        for frame,labels in datasets:
            graph=CommonModeResidualTransformer(self.history_size).transform(frame)
            item=pd.DataFrame(graph,columns=FEATURES_V4)
            item["attack"]=labels.attack.astype(int).to_numpy()
            item["scenario"]=labels.scenario.astype(str).to_numpy()
            # Development-only regime label from window position. It is used only
            # for sample weighting and never enters model features or runtime.
            item["cyclic"]=frame.window_id.astype(int).between(1800,2699).to_numpy()
            frames.append(item)
        if extra_normal is not None and len(extra_normal):
            graph=CommonModeResidualTransformer(self.history_size).transform(extra_normal)
            item=pd.DataFrame(graph,columns=FEATURES_V4)
            item["attack"]=0;item["scenario"]="N1"
            item["cyclic"]=extra_normal.window_id.astype(int).between(1800,2699).to_numpy()
            frames.append(item)

        train=pd.concat(frames,ignore_index=True)
        target=train.scenario.eq("S5")
        keep=target|train.attack.eq(0)
        x=train.loc[keep,list(FEATURES_V4)].to_numpy(float)
        y=target.loc[keep].astype(int).to_numpy()
        cyclic=train.loc[keep,"cyclic"].astype(bool).to_numpy()
        if not y.any() or y.all():
            raise ValueError("V4 requires S5 and normal development examples")

        stage1=self._new_model();stage1.fit(x,y)
        probability=stage1.predict_proba(x)[:,1]
        normal=probability[y==0]
        hard_cut=float(np.quantile(normal,self.hard_negative_quantile))
        weights=np.ones(len(y),dtype=float)
        weights[(y==0)&(probability>=hard_cut)]=self.hard_negative_weight
        weights[(y==1)&cyclic]*=self.cyclic_positive_weight
        weights[(y==0)&cyclic]*=self.cyclic_normal_weight

        self.model=self._new_model()
        self.model.fit(x,y,logisticregression__sample_weight=weights)
        self.training_meta={
            "training_rows":int(len(y)),"s5_rows":int(y.sum()),
            "cyclic_s5_rows":int(((y==1)&cyclic).sum()),
            "cyclic_normal_rows":int(((y==0)&cyclic).sum()),
            "hard_negative_rows":int(((y==0)&(probability>=hard_cut)).sum()),
            "hard_negative_cut":hard_cut,
        }
        self.reset();return self

    def reset(self):
        self.transformer.reset();self.evidence={};self.current_run=None

    def _structural(self,frame,probability,graph):
        meta=frame[["run_id","window_id","nf_id"]].reset_index(drop=True).copy()
        meta["probability"]=np.asarray(probability,dtype=float)
        meta["local"]=graph[:,FEATURES_V4.index("localized_residual_corr_contrast")]
        meta["triangle"]=graph[:,FEATURES_V4.index("residual_triangle_contrast")]
        meta["_pos"]=np.arange(len(meta))
        out=np.zeros(len(meta),dtype=float)
        for (_,window_id),window in meta.groupby(["run_id","window_id"],sort=False):
            for zone in range(4):
                slots=window.nf_id.map(slot_number)
                part=window[(slots//8).eq(zone)]
                p=part.probability.to_numpy(float)
                kth=float(np.sort(p)[-self.cluster_order])
                median=float(np.median(p))
                contrast=max(0.,kth-median)
                local=np.maximum(0.,part["local"].to_numpy(float))
                triangle=np.maximum(0.,part["triangle"].to_numpy(float))
                motif=np.clip(1.+local+triangle,1.,3.)
                out[part._pos.to_numpy(int)]=np.minimum(p,kth)*contrast*motif
        return out

    def decision_function(self,frame):
        if self.model is None:
            raise ValueError("V4 must be fitted")
        frame=pd.DataFrame(frame).copy()
        if frame.run_id.nunique()!=1:
            raise ValueError("V4 scoring call must contain one run")
        run_id=str(frame.run_id.iloc[0])
        if self.current_run is None:
            self.current_run=run_id
        elif self.current_run!=run_id:
            self.reset();self.current_run=run_id
        graph=self.transformer.transform_online(frame)
        probability=self.model.predict_proba(graph)[:,1]
        structural=self._structural(frame,probability,graph)
        out=np.zeros(len(frame),dtype=float)
        for pos,nf in enumerate(frame.nf_id.astype(str).to_numpy()):
            value=self.decay*self.evidence.get(nf,0.)+float(structural[pos])
            self.evidence[nf]=value;out[pos]=value
        return out

    def clean_copy(self):
        clone=copy.deepcopy(self);clone.reset();return clone
