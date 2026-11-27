import unittest
from ss7.hierarchical_kie_v10 import runs,allocation_grid,CHANNELS
class V10Tests(unittest.TestCase):
    def test_runs_break_on_missing_or_time_gap(self):
        self.assertEqual(runs([True,True,False,True,True,True],[0,1,2,3,4,6]),2)
        self.assertEqual(runs([True,True,True],[0,1,2]),3)
    def test_allocations(self):
        for a in allocation_grid().values():
            self.assertEqual(set(a),set(CHANNELS))
            self.assertAlmostEqual(sum(a.values()),1.)
            self.assertGreater(a["zone"],0.)
if __name__=="__main__": unittest.main()
