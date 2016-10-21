"""Created on Sat Sep 10 2016 14:04.

@author: Nathan Budd
"""
import unittest
import numpy as np
import numpy.matlib as np
import matplotlib.pyplot as plt
from ..thrust_constant import ThrustConstant
from ..model_coe import ModelCOE
from ..gauss_lagrange_planetary_eqns import GaussLagrangePlanetaryEqns
from ..perturb_zero import PerturbZero
from ...mcpi.mcpi import MCPI
from ...mcpi.mcpi_approx import MCPIapprox
from ...orbital_mech.element_sets.orb_coe import OrbCOE


class TestThrustConstant(unittest.TestCase):
    """Test class for ModelCOE."""

    def setUp(self):
        """."""
        mu = 1.
        self.vector = np.array([[0., 1., 0.]]).T
        stm = GaussLagrangePlanetaryEqns(mu).coe
        self.thrust = ThrustConstant(self.vector, stm)

    def test_instantiation(self):
        """."""
        self.assertIsInstance(self.thrust, ThrustConstant)

    def test_getattr(self):
        """."""
        self.assertEqual(self.thrust.vector.shape, (3, 1))

    def test_setattr(self):
        """."""
        self.thrust.vector = np.array([[1.]])
        self.assertEqual(self.thrust.vector.shape, (1, 1))

    def test_dynamics(self):
        x = np.array([[2., .5, 1., .1, .1, 0.],
                      [4., .5, 1., .1, .1, 0.],
                      [8., .5, 1., .1, .1, 0.]])
        t = np.array([[0.], [1.], [2.]])

        Xdot = self.thrust(t, x)
        print(self.thrust.Xdot)
        self.assertEqual(Xdot.shape, (3, 6))
