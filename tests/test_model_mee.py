"""Created on Wed Sep 07 2016 12:09.

@author: Nathan Budd
"""
import unittest
import numpy as np
import numpy.matlib as npm
import matplotlib.pyplot as plt
from ..model_mee import ModelMEE
from ..perturb_zero import PerturbZero
from ...mcpi.mcpi import MCPI
from ...mcpi.mcpi_approx import MCPIapprox
from ...orbital_mech.orbit import Orbit
from ...orbital_mech.element_sets.orb_coe import OrbCOE
from ...orbital_mech.element_sets.orb_mee import OrbMEE


class TestModelMEE(unittest.TestCase):
    """Test class for ModelMEE."""

    def setUp(self):
        """."""
        mu = 1.
        self.dmee = ModelMEE(mu)

    def test_instantiation(self):
        """."""
        self.assertIsInstance(self.dmee, ModelMEE)

    def test_getattr(self):
        """."""
        self.assertEqual(self.dmee.mu, 1)

    def test_setattr(self):
        """."""
        self.dmee.mu = 2.
        self.assertEqual(self.dmee.mu, 2)

    def test_dynamics(self):
        x = np.matrix([[2., .5, 1., .1, .1, 0.],
                       [4., .5, 1., .1, .1, 0.],
                       [8., .5, 1., .1, .1, 0.]])
        t = np.matrix([[0.], [1.], [2.]])

        xdot = self.dmee(t, x)
        print(self.dmee.Xdot)
        self.assertEqual(xdot.shape, (3, 6))

    def test_dynamics_integration(self):
        def X_guess_func(t):
            return t * npm.ones((1, 6)) + 0.1

        domains = (0., 30.)
        N = 20,
        X0 = Orbit(OrbCOE({'p': 2., 'e': 0., 'i': .5, 'W': 0., 'w': 0.,
                           'nu': 0.})).mee().list()[:-1]
        tol = 1e-10

        mcpi = MCPI(self.dmee, domains, N, X_guess_func, X0, tol)
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
