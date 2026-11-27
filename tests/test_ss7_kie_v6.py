import unittest
import numpy as np
import pandas as pd
from ss7.hierarchical_kie_v6 import normalized_residual,L6,G6,GP,graph_only,ALLOC6
class V6Tests(unittest.TestCase):
    def test_symmetric_residual(self):
        self.assertEqual(normalized_residual(0,0),0)
        self.assertAlmostEqual(normalized_residual(10,20),-normalized_residual(20,10))
        self.assertTrue(np.isfinite(normalized_residual(1e9,0)))
    def test_no_metadata_predictors(self):
        self.assertFalse({"world","nid","t","attack","episode"}&set(L6+G6))
    def test_zone_leave_one_out_and_world_isolation(self):
        d=pd.DataFrame({c:[1.,3.,100.,300.] for c in G6})
        d=d.assign(world=[1,1,2,2],t=[0]*4,zone=[0]*4)
        g=graph_only(d)
        np.testing.assert_allclose(g[GP[0]+"_zone_centered"],[-2.,2.,-200.,200.])
    def test_budget(self):
        self.assertAlmostEqual(sum(ALLOC6.values()),1.)
if __name__=="__main__": unittest.main()
