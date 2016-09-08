"""Created on Wed Sep 07 2016 12:09.

@author: Nathan Budd
"""
import unittest
import numpy as np
from ..dynamics_coe import DynamicsCOE
from ..perturb_zero import PerturbZero


class TestDynamicsCOE(unittest.TestCase):
    """Test class for DynamicsCOE."""

    def setUp(self):
        """."""
        self.dcoe = DynamicsCOE({'mu': 1., 'a_d': PerturbZero()})

    def test_instantiation(self):
        """."""
        self.assertIsInstance(self.dcoe, DynamicsCOE)

    def test_getattr(self):
        """."""
        self.assertEqual(self.dcoe.mu, 1)

    def test_setattr(self):
        """."""
        self.dcoe.mu = 2.
        self.assertEqual(self.dcoe.mu, 2)

    def test_dynamics(self):
        x = np.matrix([[2., .5, 1., .1, .1, 0.],
                       [4., .5, 1., .1, .1, 0.],
                       [8., .5, 1., .1, .1, 0.]])
        t = np.matrix([[0.], [1.], [2.]])

        xdot = self.dcoe(t, x)
        print(xdot)
        self.assertEqual(xdot.shape, (3, 6))
