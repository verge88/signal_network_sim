import unittest
import numpy as np
import pandas as pd
from ss7.hierarchical_kie_quick import causal_scores,cuts_for,margin,ALLOC,LNEW,GNEW

class QuickTests(unittest.TestCase):
    def test_causal(self):
        d=pd.DataFrame({"world":[1,1,1,2],"nid":[0]*4})
        s=pd.DataFrame({"a":[1.,3.,100.,7.]})
        a=causal_scores(d,s)
        self.assertEqual(a.a.iloc[1],2.)
        self.assertTrue(np.isneginf(a.a.iloc[3]))
        s.iloc[2]=np.nan
        self.assertTrue(np.isneginf(causal_scores(d,s).a.iloc[2]))
    def test_budget(self):
        d=pd.DataFrame({"world":[1]*2880,"t":np.arange(2880),"attack":["normal"]*2880})
        s=pd.DataFrame({c:np.roll(np.arange(2880.),i*100) for i,c in enumerate(ALLOC)})
        self.assertLessEqual(np.sum(margin(s,cuts_for(d,s,20.,ALLOC))>0),20)
    def test_no_metadata(self):
        self.assertFalse({"world","nid","t","attack","episode"} & set(LNEW+GNEW))
if __name__=="__main__": unittest.main()
