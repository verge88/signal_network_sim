"""Invariant temporal-graph features for localized coordinated drift.

The representation intentionally discards absolute load level. It measures whether
a small subset of peers becomes unusually synchronized relative to the rest of its
zone over a short causal history. This distinguishes a localized three-node motif
from legitimate zone-wide cyclic load changes.
"""
from collections import defaultdict, deque
import numpy as np
import pandas as pd

FEATURES = (
    "top2_corr_mean",
    "top2_corr_min",
    "median_peer_corr",
    "localized_corr_contrast",
    "high_corr_peer_count",
    "triangle_corr",
    "triangle_contrast",
    "delta_rank_contrast",
    "sign_cluster_contrast",
    "localized_delta_similarity",
)


def slot_number(nf_id):
    text=str(nf_id)
    if not text.startswith("slot-"):
        raise ValueError(f"Expected slot-N identity, got {text!r}")
    value=int(text.split("-",1)[1])
    if not 0 <= value < 32:
        raise ValueError(value)
    return value


def zone_number(nf_id):
    return slot_number(nf_id)//8


def _corr(a,b):
    a=np.asarray(a,dtype=float); b=np.asarray(b,dtype=float)
    if len(a)<4 or np.std(a)<1e-9 or np.std(b)<1e-9:
        return 0.0
    value=float(np.corrcoef(a,b)[0,1])
    return value if np.isfinite(value) else 0.0


class LocalSynchronyTransformer:
    def __init__(self, history=8):
        self.history_size=int(history)
        self.reset()

    def reset(self):
        self.count_history=defaultdict(lambda: deque(maxlen=self.history_size+1))

    def _features_for_window(self, part):
        if len(part)!=32 or part.nf_id.nunique()!=32:
            raise ValueError("Expected exactly 32 NF rows")
        rows={str(r.nf_id):r for r in part.itertuples(index=False)}
        # Append current counts before forming causal histories; every vector ends at t.
        for nf,row in rows.items():
            self.count_history[nf].append(float(row.request_count))

        deltas={}
        for nf,h in self.count_history.items():
            values=np.asarray(h,dtype=float)
            deltas[nf]=np.diff(values) if len(values)>=2 else np.zeros(1,dtype=float)

        result={}
        for nf,row in rows.items():
            zone=zone_number(nf)
            peers=[p for p in rows if p!=nf and zone_number(p)==zone]
            own=deltas[nf]
            peer_corr=[]
            for p in peers:
                width=min(len(own),len(deltas[p]),self.history_size)
                corr=_corr(own[-width:],deltas[p][-width:]) if width>=4 else 0.0
                peer_corr.append((p,corr))
            ordered=sorted(peer_corr,key=lambda x:x[1],reverse=True)
            top=ordered[:2]
            corr_values=np.asarray([v for _,v in peer_corr],dtype=float)
            top_values=np.asarray([v for _,v in top],dtype=float)
            median_corr=float(np.median(corr_values))
            top2_mean=float(top_values.mean()) if len(top_values) else 0.0
            top2_min=float(top_values.min()) if len(top_values) else 0.0

            # Triangle synchrony: the node plus its two strongest peers.
            triangle=0.0
            if len(top)==2:
                p1,p2=top[0][0],top[1][0]
                width=min(len(deltas[p1]),len(deltas[p2]),self.history_size)
                peer_pair=_corr(deltas[p1][-width:],deltas[p2][-width:]) if width>=4 else 0.0
                triangle=float((top[0][1]+top[1][1]+peer_pair)/3.0)

            current_delta=float(own[-1]) if len(own) else 0.0
            peer_current=np.asarray([float(deltas[p][-1]) if len(deltas[p]) else 0.0 for p in peers])
            abs_gap=np.abs(peer_current-current_delta)
            nearest=np.sort(abs_gap)[:2] if len(abs_gap)>=2 else abs_gap
            local_similarity=float(np.exp(-nearest).mean()) if len(nearest) else 0.0

            same_sign=np.asarray([
                np.sign(x)==np.sign(current_delta) and abs(x)>1e-12 and abs(current_delta)>1e-12
                for x in peer_current
            ],dtype=float)
            sign_cluster=float(np.sort(same_sign)[-2:].mean() - np.median(same_sign)) if len(same_sign)>=2 else 0.0

            abs_peer=np.abs(peer_current)
            delta_rank=float((abs_peer < abs(current_delta)).mean()) if len(abs_peer) else 0.0
            peer_rank_center=float(abs(np.median(abs_peer)-abs(current_delta)))
            delta_rank_contrast=delta_rank/(1.0+peer_rank_center)

            result[nf]=np.asarray([
                top2_mean,
                top2_min,
                median_corr,
                top2_mean-median_corr,
                float((corr_values>0.75).sum()),
                triangle,
                triangle-median_corr,
                delta_rank_contrast,
                sign_cluster,
                local_similarity,
            ],dtype=float)
        return result

    def transform(self, frame):
        frame=pd.DataFrame(frame).copy()
        required={"run_id","window_id","nf_id","request_count"}
        missing=required-set(frame)
        if missing:
            raise ValueError(f"Missing columns: {sorted(missing)}")
        self.reset()
        output=np.zeros((len(frame),len(FEATURES)),dtype=float)
        for _,run in frame.groupby("run_id",sort=False):
            self.reset()
            for _,part in run.groupby("window_id",sort=False):
                features=self._features_for_window(part)
                for idx,row in part.iterrows():
                    output[frame.index.get_loc(idx)]=features[str(row.nf_id)]
        if not np.isfinite(output).all():
            raise ValueError("Nonfinite graph feature")
        return output
