import unittest
import numpy as np
import pandas as pd
from ss7.hierarchical_kie_v11 import LOCAL,GRAPH,feature_matrix,BenignModel
class V11Tests(unittest.TestCase):
    def test_feature_causality(self):
        n=100
        d=pd.DataFrame({c:np.arange(n,dtype=float) for c in LOCAL})
        d=d.assign(world=[1]*50+[2]*50,nid=[0]*n,t=list(range(50))*2,missing=0,invalid=0)
        a=feature_matrix(d,LOCAL)
        d.loc[40:49,LOCAL]=10000
        b=feature_matrix(d,LOCAL)
        np.testing.assert_allclose(a.iloc[:40],b.iloc[:40],equal_nan=True)
        np.testing.assert_allclose(a.iloc[50:],b.iloc[50:],equal_nan=True)
    def test_benign_only_and_missing(self):
        rng=np.random.default_rng(42); x=pd.DataFrame(rng.normal(size=(100,4)))
        d=pd.DataFrame(dict(role=["SP"]*100,attack=["normal"]*80+["irsf"]*20,missing=0,invalid=0))
        m=BenignModel().fit(d,x); x2=x.copy(); x2.iloc[80:]=100000
        m2=BenignModel().fit(d,x2)
        np.testing.assert_allclose(m.models["SP"][3],m2.models["SP"][3])
        d.loc[0,"missing"]=1
        self.assertTrue(m.score(d,x).iloc[0].isna().all())
if __name__=="__main__": unittest.main()
