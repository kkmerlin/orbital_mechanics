"""Created on Wed Sep 21 2016 14:50.

@author: Nathan Budd
"""
import unittest
from math import pi
import numpy as np
import orbit as orb


class TestOrbit(unittest.TestCase):
    """Test class for Orbit."""

    def setUp(self):
        pass

    def test_M2f_ellipse(self):
        coe_no_M_row = np.array([[2., .1, 10., 0., 0.]])
        coe_no_M = np.tile(coe_no_M_row, (10, 1))
        M = np.linspace(0, 2*pi, 10).reshape((10, 1))
        coe_M = np.concatenate((coe_no_M, M), 1)

        coe_f = orb.M2f(coe_M)
        self.assertEqual(coe_f[0, -1], 0.)

    def test_M2E(self):
        coe_no_M_row = np.array([[2., .1, 10., 0., 0.]])
        coe_no_M = np.tile(coe_no_M_row, (10, 1))
        M = np.linspace(0, 2*pi, 10).reshape((10, 1))
        coe_M = np.concatenate((coe_no_M, M), 1)

        coe_E = orb.M2E(coe_M)
        self.assertEqual(coe_E[0, -1], 0.)

    def test_E2f_ellipse(self):
        coe_no_E_row = np.array([[2., .1, 10., 0., 0.]])
        coe_no_E = np.tile(coe_no_E_row, (10, 1))
        E = np.linspace(0, 2*pi, 10).reshape((10, 1))
        coe_E = np.concatenate((coe_no_E, E), 1)

        coe_f = orb.E2f(coe_E)
        self.assertEqual(coe_f[0, -1], 0.)
