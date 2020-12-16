import unittest
import numpy as np
import sys
sys.path.append("./sir/")
import sir_ode
import sir_discrete

tol = 1e-8

class TestSirDiscrete(unittest.TestCase):

    def setUp(self):
        pass

    def test_sir_discrete(self):
        """
        Test s(t) + i(t) + r(t) = 1 for all t.
        """
        np.random.seed()
        N = np.random.randint(low = 1, high = 1000)
        k = np.random.rand()
        b = np.random.randint(low = 1, high = 10)
        t = 10
        i0 = np.random.rand()
        s0 = 1 - i0

        s, i, r = sir_discrete.simulation(b, k, t, N, s0, i0)
        # Check that s+i+r = 1 for all time points
        self.assertTrue(np.all(np.abs(np.array(s) + np.array(i) + np.array(r) - np.ones(t + 1)) < tol))


class TestSirOde(unittest.TestCase):

    def setUp(self):
        pass

    def test_sir_ode(self):
        """
        Test s(t) + i(t) + r(t) = 1 for all t.
        """
        N = np.random.randint(low=1)
        k = np.random.rand()
        b = np.random.randint(low=1)
        i0 = np.random.rand()

        ode = sir_ode.SIR_ODE(k,b,i0,N)
        teval = np.linspace(0, 10, 100)
        (sol, obj) = ode.simulate(teval, normalized_plot=False)

        # Check that s+i+r = 1 for all time points
        self.assertTrue(np.all(np.abs(np.array(sol.y[0]+sol.y[1]+sol.y[2]) - np.ones(100)) < tol))

if __name__ == '__main__':
    unittest.main()
