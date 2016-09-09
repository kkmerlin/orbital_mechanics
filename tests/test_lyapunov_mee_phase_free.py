"""Created on Wed Sep 08 2016 13:11.

@author: Nathan Budd
"""
import unittest
import numpy as np
import numpy.matlib as npm
from ..lyapunov_mee_phase_free import LyapunovMEEPhaseFree
from ..perturb_zero import PerturbZero
from ..dynamics_mee import DynamicsMEE
from ...orbital_mech.orbit import Orbit
from ...orbital_mech.element_sets.orb_coe import OrbCOE
from ...orbital_mech.element_sets.orb_mee import OrbMEE
from ...mcpi.mcpi import MCPI
from ...mcpi.mcpi_approx import MCPIapprox


class TestLyapunovMEEPhaseFree(unittest.TestCase):
    """Test class for LyapunovMEEPhaseFree."""

    def setUp(self):
        """."""
        xref = np.matrix([[1., 1., 1., 1., 1.]])
        self.lepf = LyapunovMEEPhaseFree({'mu': 1., 'xref': xref})

    def test_instantiation(self):
        """."""
        self.assertIsInstance(self.lepf, LyapunovMEEPhaseFree)

    def test_getattr(self):
        """."""
        self.assertEqual(self.lepf.mu, 1)

    def test_setattr(self):
        """."""
        self.lepf.mu = 2.
        self.assertEqual(self.lepf.mu, 2)

    def test_control(self):
        """."""
        x = np.matrix([[2., .5, 1., .1, .1, 0.],
                       [4., .5, 1., .1, .1, 0.],
                       [8., .5, 1., .1, .1, 0.]])
        t = np.matrix([[0.], [1.], [2.]])

        U = self.lepf(t, x)
        self.assertEqual(len(U), 3)

    def test_control_example(self):
        """."""
        X0 = Orbit(OrbCOE({'p': 2., 'e': .1, 'i': .1, 'W': 0., 'w': 0.,
                           'nu': 0.})).mee().list()[:-1]
        Xref = [X0[0]*1.] + X0[1:-1]

        mu = 1.
        ly_mee = LyapunovMEEPhaseFree({'mu': mu, 'xref': np.matrix(Xref)})
        dynmee = DynamicsMEE({'mu': mu, 'a_d': ly_mee})

        def X_guess_func(t):
            return t * npm.ones((1, 6)) + 0.1

        domains = (0., 30.)
        N = 20,
        X0 = Orbit(OrbCOE({'p': 2., 'e': 0., 'i': .5, 'W': 0., 'w': 0.,
                           'nu': 0.})).mee().list()[:-1]
        tol = 1e-10

        mcpi = MCPI(dynmee, domains, N, X_guess_func, X0, tol)
        X_approx = mcpi.solve_serial()
        print(mcpi.iterations)

        T_step = 0.1
        T = np.arange(domains[0], domains[1]+T_step, T_step).tolist()
        x_approx = X_approx(T)
        plt_p, = plt.plot(T, [row[0] for row in x_approx], label='p')
        plt_f, = plt.plot(T, [row[1] for row in x_approx], label='f')
        plt_g, = plt.plot(T, [row[2] for row in x_approx], label='g')
        plt_h, = plt.plot(T, [row[3] for row in x_approx], label='h')
        plt_k, = plt.plot(T, [row[4] for row in x_approx], label='k')
        plt_L, = plt.plot(T, [row[5] for row in x_approx], label='L')
        plt.legend(handles=[plt_p, plt_f, plt_g, plt_h, plt_k, plt_L])
        plt.show()
        self.assertIsInstance(X_approx, MCPIapprox)
