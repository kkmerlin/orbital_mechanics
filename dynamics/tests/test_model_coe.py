"""Created on Wed Sep 07 2016 12:09.

@author: Nathan Budd
"""
import unittest
import numpy as np
import matplotlib.pyplot as plt
from ..model_coe import ModelCOE
from ..perturb_zero import PerturbZero
from ..reference_coe import ReferenceCOE
from ..warm_start_constant import WarmStartConstant
from ...mcpi.mcpi import MCPI
from ...mcpi.mcpi_approx import MCPIapprox
from ...orbital_mech.element_sets.orb_coe import OrbCOE


class TestModelCOE(unittest.TestCase):
    """Test class for ModelCOE."""

    def setUp(self):
        """."""
        mu = 1.
        self.mcoe = ModelCOE(mu)

    def test_instantiation(self):
        """."""
        self.assertIsInstance(self.mcoe, ModelCOE)

    def test_getattr(self):
        """."""
        self.assertEqual(self.mcoe.mu, 1)

    def test_setattr(self):
        """."""
        self.mcoe.mu = 2.
        self.assertEqual(self.mcoe.mu, 2)

    def test_dynamics(self):
        x = np.array([[2., .5, 1., .1, .1, 0.],
                      [4., .5, 1., .1, .1, 0.],
                      [8., .5, 1., .1, .1, 0.]])
        t = np.array([[0.], [1.], [2.]])

        xdot = self.mcoe(t, x)
        print(self.mcoe.Xdot)
        self.assertEqual(xdot.shape, (3, 6))

    def test_dynamics_integration(self):
        domains = (0., 6.)
        N = 20,
        X0 = [2., .1, .1, 0., 0., 0.]
        tol = 1e-10

        mcpi = MCPI(self.mcoe, domains, N, WarmStartConstant(), X0, tol)
        X_approx = mcpi.solve_serial()
        print(mcpi.iterations)

        T = np.linspace(domains[0], domains[1], 100).reshape((100, 1))
        x_approx = X_approx(T)
        plt_p, = plt.plot(T, [row[0] for row in x_approx], label='p')
        plt_e, = plt.plot(T, [row[1] for row in x_approx], label='e')
        plt_i, = plt.plot(T, [row[2] for row in x_approx], label='i')
        plt_W, = plt.plot(T, [row[3] for row in x_approx], label='W')
        plt_w, = plt.plot(T, [row[4] for row in x_approx], label='w')
        plt_f, = plt.plot(T, [row[5] for row in x_approx], label='f')
        plt.legend(handles=[plt_p, plt_e, plt_i, plt_W, plt_w, plt_f])
        plt.show()
        self.assertIsInstance(X_approx, MCPIapprox)

    def test_reference(self):
        domains = (0., 1.)
        T = np.linspace(domains[0], domains[1], 100).reshape((100, 1))
        X0 = np.array([[2., .1, .1, 0., 0., 0.]])
        X = ReferenceCOE(X0, 1.)(T).tolist()

        plt_p, = plt.plot(T, [row[0] for row in X], label='p')
        plt_e, = plt.plot(T, [row[1] for row in X], label='e')
        plt_i, = plt.plot(T, [row[2] for row in X], label='i')
        plt_W, = plt.plot(T, [row[3] for row in X], label='W')
        plt_w, = plt.plot(T, [row[4] for row in X], label='w')
        plt_f, = plt.plot(T, [row[5] for row in X], label='f')
        plt.legend(handles=[plt_p, plt_e, plt_i, plt_W, plt_w, plt_f])
        plt.show()
        self.assertEqual(X[-1][0], X0[0,0])
