import unittest
import numpy as np
import pandas as pd
from ss7.hierarchical_kie_v2 import TailScale, accumulate, pooled_threshold, evaluate, group_weights

class V2Tests(unittest.TestCase):
    def test_role_scale_extrapolates(self):
        d=pd.DataFrame({"role":["SP"]*100,"attack":["normal"]*100})
        x=pd.DataFrame({"a":np.arange(100.)})
        s=TailScale().fit(d,x)
        v=s.transform(d.iloc[:2],pd.DataFrame({"a":[101.,110.]}))
        self.assertGreater(v.a.iloc[1],v.a.iloc[0])
    def test_causal_and_world_reset(self):
        d=pd.DataFrame({"world":[1,1,1,2],"nid":[0]*4,"t":[0,1,2,0]})
        a=pd.DataFrame({"a":[5.,5.,np.nan,0.]})
        s=accumulate(d,a)
        self.assertEqual(s.a.iloc[2],-12.)
        self.assertEqual(s.a.iloc[3],0.)
        b=a.copy(); b.iloc[2]=30
        np.testing.assert_allclose(s.iloc[:2],accumulate(d,b).iloc[:2])
    def test_pooled_threshold(self):
        d=pd.DataFrame({"world":[1,1,2,2],"t":[0,1,0,1],"attack":["normal"]*4})
        self.assertEqual(pooled_threshold(d,np.arange(4.)),3.)
    def test_group_weights(self):
        d=pd.DataFrame({"world":[1,1,2,2,2],"episode":[-1,0,-1,1,1]})
        w=group_weights(d,np.array([0,1,0,1,1]))
        self.assertAlmostEqual(w[1],w[3]+w[4])
        self.assertAlmostEqual(w[0],w[2])
    def test_confirmation_world_boundary(self):
        d=pd.DataFrame({"world":[1,1,2,2,2,2],"nid":[0,0,0,0,0,1],
                        "t":[0,1,2,3,4,0],"attack":["sms_intercept"]*5+["normal"],
                        "episode":[0]*5+[-1]})
        x=evaluate(d,np.ones(len(d)),.5,"test","test")
        self.assertEqual(x[0]["episodes"],2)
        self.assertEqual(x[0]["detected"],1)
if __name__=="__main__": unittest.main()
