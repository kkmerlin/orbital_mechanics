"""Created on Wed Sep 08 2016 13:11.

@author: Nathan Budd
"""
import unittest
import numpy as np
import numpy.matlib as npm
from ..lyapunov_element_steering_optimal import LyapunovElementSteeringOptimal
from ..perturb_zero import PerturbZero
from ..model_mee import ModelMEE
from ...orbital_mech.orbit import Orbit
from ...orbital_mech.element_sets.orb_coe import OrbCOE
from ...orbital_mech.element_sets.orb_mee import OrbMEE
from ...mcpi.mcpi import MCPI
from ...mcpi.mcpi_approx import MCPIapprox


class TestLyapunovElementSteeringOptimal(unittest.TestCase):
    """Test class for LyapunovElementSteeringOptimal."""

    def setUp(self):
        """."""
        mu = 1.
        a_t = 1e-6
        xref = ModelMEE(mu).reference
        self.lmo = LyapunovElementSteeringOptimal(mu, a_t, xref)

    def test_instantiation(self):
        """."""
        self.assertIsInstance(self.lmo, LyapunovElementSteeringOptimal)

    def test_getattr(self):
        """."""
        self.assertEqual(self.lmo.mu, 1)

    def test_setattr(self):
        """."""
        self.lmo.mu = 2.
        self.assertEqual(self.lmo.mu, 2)

    def test_control(self):
        """."""
        x = np.matrix([[2., .5, 1., .1, .1, 0.],
                       [4., .5, 1., .1, .1, 0.],
                       [8., .5, 1., .1, .1, 0.]])
        t = np.matrix([[0.], [1.], [2.]])

        U = self.lmo(t, x)
        self.assertEqual(len(U), 3)
