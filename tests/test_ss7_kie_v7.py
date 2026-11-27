import unittest
import numpy as np
import pandas as pd
from ss7.hierarchical_kie_v7 import temporal,node_adjust,ALLOCATIONS
class V7Tests(unittest.TestCase):
    def test_allocations(self):
        for a in ALLOCATIONS.values():
            self.assertAlmostEqual(sum(a.values()),1.)
            self.assertGreater(a["zone"],0.)
    def test_cusum_causal_and_world_reset(self):
        d=pd.DataFrame({"world":[1,1,1,2],"nid":[0]*4,"t":[0,1,2,0]})
        s=pd.DataFrame({"a":[2.,2.,np.nan,0.]})
        x=temporal(d,s,"cusum")
        self.assertGreater(x.a.iloc[1],x.a.iloc[0])
        self.assertTrue(np.isneginf(x.a.iloc[2]))
        self.assertEqual(x.a.iloc[3],0.)
        s.iloc[2]=100.
        np.testing.assert_allclose(x.iloc[:2],temporal(d,s,"cusum").iloc[:2])
    def test_node_adjust_causal(self):
        d=pd.DataFrame({"world":[1]*70,"nid":[0]*70})
        s=pd.DataFrame({"a":[1.]*70})
        x=node_adjust(d,s)
        s.iloc[50:]=100.
        np.testing.assert_allclose(x.iloc[:50],node_adjust(d,s).iloc[:50])
        self.assertEqual(x.a.iloc[40],.5)
if __name__=="__main__": unittest.main()
