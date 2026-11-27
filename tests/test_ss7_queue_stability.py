import unittest
import numpy as np
from ss7.fix import mm1k_loss, link_state, Linkset

class QueueStabilityTests(unittest.TestCase):
    def test_reference_stationary_distribution(self):
        for k in (1, 5, 40, 400):
            for rho in (0., .1, .9, .9999999999, 1., 1.0000000001, 1.1, 2., 10., 12., 1e4):
                if rho == 0.:
                    expected = 0.
                else:
                    logw = np.arange(k+1)*np.log(rho)
                    w = np.exp(logw-logw.max())
                    expected = w[-1]/w.sum()
                with self.subTest(k=k,rho=rho):
                    self.assertAlmostEqual(mm1k_loss(rho,k),expected,places=12)
    def test_overflow_reproduction(self):
        for rho in (6., 8., 10., 12., 12.01, 1e100):
            p=mm1k_loss(rho,400)
            self.assertTrue(np.isfinite(p))
            self.assertAlmostEqual(p,1.-1./rho,places=12)
    def test_monotonic_and_limits(self):
        for k in (1, 10, 400, 10000):
            v=[mm1k_loss(x,k) for x in (0., .5, .99, 1., 1.01, 2., 12., 1e100, float("inf"))]
            self.assertTrue(all(0<=p<=1 for p in v))
            self.assertTrue(all(a<=b for a,b in zip(v,v[1:])))
        self.assertEqual(mm1k_loss(0.,0),1.)
        self.assertEqual(mm1k_loss(1.,400),1./401)
    def test_link_state_at_overload(self):
        ls=Linkset(1,2,1,buffer_msgs=400)
        rho,delay,p=link_state(8.*ls.capacity_bps,ls,800.)
        self.assertEqual(rho,8.)
        self.assertTrue(np.isfinite(delay))
        self.assertAlmostEqual(p,.875,places=12)
if __name__=="__main__": unittest.main()
