import unittest
from types import SimpleNamespace
import numpy as np
from ss7.fix import NodeState,MSG_TYPES,MT_IDX,Service
from ss7.hierarchical_kie_v12 import account_cohort,PLAN12
class V12Tests(unittest.TestCase):
    def states(self): return {n:NodeState(n,0,np.zeros(len(MSG_TYPES),dtype=np.int64)) for n in (1,2,3)}
    def test_exact_direction_and_conservation(self):
        s=self.states()
        account_cohort(s,(1,2,3),Service.SMS_MT,10,10,np.random.default_rng(42))
        for st in s.values(): st.check_identities()
        for u,v in [(1,2),(2,3)]:
            k=(u,v)
            self.assertEqual(s[u].link_out[k],s[v].link_in[k])
            self.assertEqual(s[v].link_out[k],s[u].link_in[k])
        self.assertEqual(s[2].total(),40)
        self.assertEqual(s[1].dlg_end,10)
    def test_incomplete_cohort(self):
        s=self.states()
        account_cohort(s,(1,2,3),Service.SMS_MT,100,0,np.random.default_rng(42))
        self.assertEqual(s[1].mtype[MT_IDX["SRI_SM"]],100)
        self.assertLess(s[1].mtype[MT_IDX["MT_FSM"]],100)
        self.assertEqual(s[3].dlg_end,0)
        for st in s.values(): st.check_identities()
    def test_disjoint_control_roles(self):
        self.assertEqual(len({i for _,i in PLAN12}),14)
        self.assertEqual([r for r,i in PLAN12 if r.endswith("_benign")],["validation_benign","test_benign"])
if __name__=="__main__": unittest.main()
