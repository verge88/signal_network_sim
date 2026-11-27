import unittest
from types import SimpleNamespace
import numpy as np
import pandas as pd
from ss7.fix import NodeState, MSG_TYPES
from ss7.hierarchical_kie import pack,unpack,key_for,deliver,peer_mean,metrics,threshold

class KieTests(unittest.TestCase):
    def test_wire_and_stale(self):
        st=NodeState(3,7,np.zeros(len(MSG_TYPES),dtype=np.int64))
        r=SimpleNamespace(state=st,integrity_ok=True); k=key_for(1,3)
        w=pack(r,k)
        self.assertEqual(st,unpack(w,k,3,7))
        with self.assertRaises(ValueError): unpack(w,k,3,8)
        r.integrity_ok=False
        with self.assertRaises(ValueError): unpack(pack(r,k),k,3,7)
    def test_missing_is_not_valid(self):
        st=NodeState(3,7,np.zeros(len(MSG_TYPES),dtype=np.int64))
        rec,status,stats=deliver({3:SimpleNamespace(state=st,integrity_ok=True)},1,7,np.random.default_rng(42),loss=1.)
        self.assertEqual(rec,{})
        self.assertEqual(status[3],"missing")
        self.assertEqual(stats["accepted"],0)
        self.assertGreater(stats["offered_bytes"],0)
    def test_leave_one_out(self):
        d=pd.DataFrame({"t":[1,1,1],"zone":[0,0,0]}); p=pd.DataFrame({"score":[1.,3.,5.]})
        np.testing.assert_allclose(peer_mean(d,p,["t","zone"]).score,[4,3,2])
    def test_confirmation_does_not_cross_gaps(self):
        d=pd.DataFrame({"nid":[1,1,1,1,2],"t":[0,2,3,5,0],"attack":["sms_intercept"]*4+["normal"],
                        "episode":[0,0,0,0,-1]})
        self.assertEqual(metrics(d,np.ones(len(d)),.5,"test","test")[0]["detected"],0)
        self.assertEqual(threshold(d,np.ones(len(d))),1.)

if __name__=="__main__": unittest.main()
