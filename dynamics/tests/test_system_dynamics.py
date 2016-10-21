"""Created on Wed Sep 08 2016 13:11.

@author: Nathan Budd
"""
import unittest
import numpy as np
import numpy.matlib as npm
from ..system_dynamics import SystemDynamics
from ..model_coe import ModelCOE
from ..perturb_zero import PerturbZero
from ...orbital_mech.orbit import Orbit
from ...orbital_mech.element_sets.orb_coe import OrbCOE
from ...orbital_mech.element_sets.orb_mee import OrbMEE
from ...mcpi.mcpi import MCPI
from ...mcpi.mcpi_approx import MCPIapprox


class TestSystemDynamics(unittest.TestCase):
    """Test class for SystemDynamics."""

    def setUp(self):
        """."""
        mu = 1.
        self.sys = SystemDynamics(ModelCOE(mu))

    def test_instantiation(self):
        """."""
        self.assertIsInstance(self.sys, SystemDynamics)

    def test_getattr(self):
        """."""
        self.assertIsInstance(self.sys.plant, ModelCOE)

    def test_setattr(self):
        """."""
        self.sys.control = PerturbZero()
        self.assertIsInstance(self.sys.control, PerturbZero)

    def test_control(self):
        """."""
        x = np.matrix([[2., .5, 1., .1, .1, 0.],
                       [4., .5, 1., .1, .1, 0.],
                       [8., .5, 1., .1, .1, 0.]])
        t = np.matrix([[0.], [1.], [2.]])

        Xdot = self.sys(t, x)
        self.assertEqual(Xdot.shape, (3, 6))
