"""Created on Wed Sep 07 2016 12:09.

@author: Nathan Budd
"""
import unittest
import numpy as np
from ..dynamics_rv_2body import DynamicsRV2body


class TestDynamicsRV2body(unittest.TestCase):
    """Test class for DynamicsAbstract."""

    def setUp(self):
        """."""
        self.drv = DynamicsRV2body({'mu': 1.})

    def test_instantiation(self):
        """."""
        self.assertIsInstance(self.drv, DynamicsRV2body)

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
        print(rdot)
        self.assertEqual(rdot.shape, (3, 3))
