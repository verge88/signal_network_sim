import unittest
from types import SimpleNamespace
import numpy as np
from ss7.fix import Service
from ss7.hierarchical_kie_v8 import EventObserver,DIM,FIDX,pack_observer,unpack_observer,L8,G8
class V8Tests(unittest.TestCase):
    def test_cohort_conservation(self):
        topo=SimpleNamespace(nodes={1:SimpleNamespace(external=False),2:SimpleNamespace(external=True)})
        o=EventObserver(topo,41,capture=1.)
        o.begin(0); o.on_batch((1,2),Service.SMS_MT,10,7)
        i=FIDX[Service.SMS_MT,True]
        self.assertEqual(o.rows[1][i,0],10)
        complete=int(o.rows[1][i,1]); expired=0
        self.assertEqual(o.open[1][i],10-complete)
        for t in (1,2,3):
            o.begin(t); complete+=int(o.rows[1][i,1]); expired+=int(o.rows[1][i,2])
        self.assertEqual((complete,expired,int(o.open[1][i])),(7,3,0))
    def test_wire_freshness_and_integrity(self):
        a=np.zeros((DIM,4),dtype=int)
        w=pack_observer(1,2,3,a,0)
        np.testing.assert_array_equal(unpack_observer(1,2,3,w)[0],a)
        with self.assertRaises(ValueError): unpack_observer(1,2,4,w)
        with self.assertRaises(ValueError): unpack_observer(2,2,3,w)
    def test_no_metadata(self):
        self.assertFalse({"world","nid","t","attack","episode"}&set(L8+G8))
if __name__=="__main__": unittest.main()
