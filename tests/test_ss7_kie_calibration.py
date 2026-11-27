import unittest
import numpy as np
import pandas as pd
from ss7.hierarchical_kie_calibration import node_adjust,joint,safe_threshold,worst_world_threshold,tail_fit,tail_transform,smooth
class CalibrationTests(unittest.TestCase):
    def test_future_does_not_change_past(self):
        d=pd.DataFrame({"world":[1]*80,"nid":[0]*80})
        s=pd.DataFrame({"a":np.linspace(0,4,80)})
        a=node_adjust(d,s)
        s.iloc[65:]=100
        np.testing.assert_allclose(a.iloc[:65],node_adjust(d,s).iloc[:65])
    def test_missing_is_not_alarm(self):
        s=pd.DataFrame({"a":[np.nan,1.],"b":[np.nan,2.]})
        self.assertTrue(np.isneginf(joint(s,["a","b"])[0]))
    def test_global_budget(self):
        d=pd.DataFrame({"world":[1]*2880,"t":np.arange(2880),"attack":["normal"]*2880})
        x=np.arange(2880.)
        cut=safe_threshold(d,x,20.)
        self.assertEqual((x>cut).sum(),20)
    def test_worst_world_budget(self):
        d=pd.DataFrame({"world":np.repeat([1,2],2880),"t":np.tile(np.arange(2880),2),"attack":["normal"]*5760})
        x=np.r_[np.arange(2880.),np.arange(2880.)+1000]
        cut=worst_world_threshold(d,x,20.)
        self.assertLessEqual((x[:2880]>cut).sum(),20)
        self.assertLessEqual((x[2880:]>cut).sum(),20)
    def test_ewma_missing(self):
        d=pd.DataFrame({"world":[1]*4,"nid":[0]*4})
        s=pd.DataFrame({"a":[1.,2.,np.nan,3.]})
        self.assertTrue(np.isneginf(smooth(d,s,"ewma").a.iloc[2]))
if __name__=="__main__": unittest.main()
