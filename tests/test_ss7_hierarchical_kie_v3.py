import unittest
import numpy as np
import pandas as pd
from ss7.hierarchical_kie_v3 import Scale, temporal, channel_thresholds, union_margin, ALLOC, LOCAL3, GRAPH3

class V3Tests(unittest.TestCase):
    def test_scale_has_no_ceiling(self):
        d=pd.DataFrame({"role":["SP"]*100,"attack":["normal"]*100})
        x=pd.DataFrame({"a":np.arange(100.)})
        s=Scale().fit(d,x)
        a=s.transform(d.iloc[:2],pd.DataFrame({"a":[1000.,2000.]}))
        self.assertGreater(a.a.iloc[1],a.a.iloc[0])
        self.assertGreater(a.a.iloc[0],30.)
    def test_union_independence(self):
        s=pd.DataFrame({"service":[2.,2.],"compromise":[-5.,-5.],"corroborated":[-10.,100.]})
        m=union_margin(s,{"service":1.,"compromise":1.,"corroborated":1.})
        self.assertTrue(np.all(m>0))
    def test_temporal_missing_and_world_reset(self):
        d=pd.DataFrame({"world":[1,1,1,2],"nid":[0]*4,"t":[0,1,2,0]})
        z=pd.DataFrame({"a":[100.,100.,np.nan,0.]})
        s=temporal(d,z)
        self.assertTrue(np.isneginf(s.a.iloc[2]))
        self.assertEqual(s.a.iloc[3],0.)
    def test_budget_union(self):
        d=pd.DataFrame({"world":[1]*2880,"t":np.arange(2880),"attack":["normal"]*2880})
        s=pd.DataFrame({c:np.roll(np.arange(2880.),i*100) for i,c in enumerate(ALLOC)})
        cut=channel_thresholds(d,s,20.)
        self.assertLessEqual(np.sum(union_margin(s,cut)>0),20)
    def test_no_metadata_features(self):
        self.assertFalse({"world","nid","t","attack","episode"} & set(LOCAL3+GRAPH3))
if __name__=="__main__": unittest.main()
