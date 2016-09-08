"""Created on Wed Sep 08 2016 13:11.

@author: Nathan Budd
"""
import unittest
import numpy as np
from ..lyapunov_elements_phase_free import LyapunovElementsPhaseFree
from ..perturb_zero import PerturbZero


class TestLyapunovElementsPhaseFree(unittest.TestCase):
    """Test class for LyapunovElementsPhaseFree."""

    def setUp(self):
        """."""
        self.lepf = LyapunovElementsPhaseFree({})

    def test_instantiation(self):
        """."""
        self.assertIsInstance(self.lepf, LyapunovElementsPhaseFree)

    def test_getattr(self):
        """."""
        self.assertEqual(self.lepf.mu, 1)

    def test_setattr(self):
        """."""
        self.lepf.mu = 2.
        self.assertEqual(self.lepf.mu, 2)

    def test_dynamics(self):
        x = np.matrix([[2., .5, 1., .1, .1, 0.],
                       [4., .5, 1., .1, .1, 0.],
                       [8., .5, 1., .1, .1, 0.]])
        t = np.matrix([[0.], [1.], [2.]])

        xdot = self.lepf(t, x)
        print(xdot)
        self.assertEqual(xdot.shape, (3, 6))
