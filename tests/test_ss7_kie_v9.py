import unittest
import numpy as np
import pandas as pd
from ss7.hierarchical_kie_v9 import BASE,profiles
class V9Tests(unittest.TestCase):
    def test_causal_and_world_isolation(self):
        n=100
        d=pd.DataFrame({c:np.arange(n,dtype=float) for c in BASE})
        d=d.assign(world=[1]*50+[2]*50,nid=[0]*n,t=list(range(50))*2)
        a,cols=profiles(d)
        d.loc[40:49,BASE]=10000
        b,_=profiles(d)
        np.testing.assert_allclose(a.loc[:39,cols],b.loc[:39,cols],equal_nan=True)
        np.testing.assert_allclose(a.loc[50:,cols],b.loc[50:,cols],equal_nan=True)
        self.assertFalse({"world","nid","t","attack","episode"}&set(cols))
if __name__=="__main__": unittest.main()
