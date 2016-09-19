"""Created on Wed Sep 08 2016 22:54.

@author: Nathan Budd
"""
import unittest
import numpy as np
from ..gauss_lagrange_planetary_eqns import GaussLagrangePlanetaryEqns


class TestGaussLagrangePlanetaryEqns(unittest.TestCase):
    """Test class for GaussLagrangePlanetaryEqns."""

    def setUp(self):
        """."""
        self.glpe = GaussLagrangePlanetaryEqns(1.)

    def test_instantiation(self):
        """."""
        self.assertIsInstance(self.glpe, GaussLagrangePlanetaryEqns)

    def test_getattr(self):
        """."""
        self.assertEqual(self.glpe.mu, 1)

    def test_setattr(self):
        """."""
        self.glpe.mu = 2.
        self.assertEqual(self.glpe.mu, 2)

    def test_mee(self):
        x = np.array([[2., .5, 1., .1, .1, 0.]])

        M = self.glpe.mee(x)
        print(M)
        self.assertEqual(M[0].shape, (6, 3))

    def test_mee_multiple(self):
        X = np.array([[2., .5, 1., .1, .1, 0.]]*10)

        M = self.glpe.mee(X)
        self.assertEqual(len(M), 10)
