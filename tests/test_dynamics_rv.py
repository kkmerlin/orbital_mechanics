"""Created on Wed Sep 07 2016 12:09.

@author: Nathan Budd
"""
import unittest
import numpy as np
import numpy.matlib as npm
import matplotlib.pyplot as plt
from ..dynamics_rv import DynamicsRV
from ..perturb_zero import PerturbZero
from ...mcpi.mcpi import MCPI
from ...mcpi.mcpi_approx import MCPIapprox
from ...orbital_mech.orbit import Orbit
from ...orbital_mech.element_sets.orb_coe import OrbCOE
from ...orbital_mech.element_sets.orb_rv import OrbRV


class TestDynamicsRV(unittest.TestCase):
    """Test class for DynamicsRV."""

    def setUp(self):
        """."""
        mu = 1.
        self.drv = DynamicsRV(mu)

    def test_instantiation(self):
        """."""
        self.assertIsInstance(self.drv, DynamicsRV)

    def test_getattr(self):
        """."""
        self.assertEqual(self.drv.mu, 1)

    def test_setattr(self):
        """."""
        self.drv.mu = 2.
        self.assertEqual(self.drv.mu, 2)

    def test_dynamics(self):
        r = np.matrix([[1., 0., 0.],
                       [0., 1., 0.],
                       [0., 0., 1.]])
        t = np.matrix([[0.], [1.], [2.]])

        rdot = self.drv(t, r)
        print(self.drv.Xdot)
        self.assertEqual(rdot.shape, (3, 3))

    def test_dynamics_integration(self):
        def X_guess_func(t):
            return t * npm.ones((1, 6)) + 0.1

        domains = (0., 30.)
        N = 20,
        X0 = Orbit(OrbCOE({'p': 2., 'e': 0., 'i': .5, 'W': 0., 'w': 0.,
                           'nu': 0.})).rv().list()[:-1]
        tol = 1e-10

        mcpi = MCPI(self.drv, domains, N, X_guess_func, X0, tol)
        X_approx = mcpi.solve_serial()
        print(mcpi.iterations)

        T_step = 0.1
        T = np.arange(domains[0], domains[1]+T_step, T_step).tolist()
        x_approx = X_approx(T)
        plt_rx, = plt.plot(T, [row[0] for row in x_approx], label='r_x')
        plt_ry, = plt.plot(T, [row[1] for row in x_approx], label='r_y')
        plt_rz, = plt.plot(T, [row[2] for row in x_approx], label='r_z')
        plt_vx, = plt.plot(T, [row[3] for row in x_approx], label='v_x')
        plt_vy, = plt.plot(T, [row[4] for row in x_approx], label='v_y')
        plt_vz, = plt.plot(T, [row[5] for row in x_approx], label='v_z')
        plt.legend(handles=[plt_rx, plt_ry, plt_rz, plt_vx, plt_vy, plt_vz])
        plt.show()
        self.assertIsInstance(X_approx, MCPIapprox)
