"""Created on Wed Sep 21 2016 14:50.

@author: Nathan Budd
"""
import unittest
from math import pi
import numpy as np
import numpy.random as npr
import ..orbit as orb
from .diff_elements import diff_elements


class TestOrbit(unittest.TestCase):
    """Test class for Orbit."""

    def setUp(self):
        pass

    def test_rv2mee2rv(self):
        tol = 1e-9

        m = 10000
        p = npr.rand(m, 1) * 10
        e = npr.rand(m, 1)
        i = npr.rand(m, 1) * np.pi
        W = npr.rand(m, 1) * 2*np.pi
        w = npr.rand(m, 1) * 2*np.pi
        f = npr.rand(m, 1) * 2*np.pi
        COE_0 = np.concatenate((p, e, i, W, w, f), 1)

        RV_1 = orb.coe2rv(COE_0)

        MEE = orb.rv2mee(RV_1)
        RV_2 = orb.mee2rv(MEE)

        RV_diff = RV_1 - RV_2
        print(RV_diff)

        self.assertTrue((np.fabs(RV_diff) < tol).all())

    def test_mee2rv2mee(self):
        tol = 1e-12

        m = 10000
        p = npr.rand(m, 1) * 10
        e = npr.rand(m, 1)
        i = npr.rand(m, 1) * np.pi*.95
        W = npr.rand(m, 1) * 2*np.pi
        w = npr.rand(m, 1) * 2*np.pi
        f = npr.rand(m, 1) * 2*np.pi
        COE_0 = np.concatenate((p, e, i, W, w, f), 1)

        MEE_1 = orb.coe2mee(COE_0)
        RV = orb.mee2rv(MEE_1)
        MEE_2 = orb.rv2mee(RV)

        MEE_diff = MEE_1 - MEE_2
        print(MEE_diff)

        self.assertTrue((np.fabs(MEE_diff) < tol).all())

    def test_mee2coe2mee(self):
        tol = 1e-12

        m = 10000
        p = npr.rand(m, 1) * 10
        e = npr.rand(m, 1)
        i = npr.rand(m, 1) * np.pi
        W = npr.rand(m, 1) * 2*np.pi
        w = npr.rand(m, 1) * 2*np.pi
        f = npr.rand(m, 1) * 2*np.pi
        COE_0 = np.concatenate((p, e, i, W, w, f), 1)
        MEE_1 = orb.coe2mee(COE_0)

        COE = orb.mee2coe(MEE_1)
        MEE_2 = orb.coe2mee(COE)

        MEE_diff = diff_elements(MEE_1, MEE_2, angle_idx=[5])
        print(MEE_diff)

        self.assertTrue((np.fabs(MEE_diff) < tol).all())

    def test_coe2mee2coe(self):
        tol = 1e-14

        m = 10000
        p = npr.rand(m, 1) * 10
        e = npr.rand(m, 1)
        i = npr.rand(m, 1) * np.pi
        W = npr.rand(m, 1) * 2*np.pi
        w = npr.rand(m, 1) * 2*np.pi
        f = npr.rand(m, 1) * 2*np.pi
        COE_1 = np.concatenate((p, e, i, W, w, f), 1)

        MEE = orb.coe2mee(COE_1)
        COE_2 = orb.mee2coe(MEE)

        COE_diff = diff_elements(COE_1, COE_2, angle_idx=[2, 3, 4, 5])
        print(COE_diff)

        self.assertTrue((np.fabs(COE_diff) < tol).all())

    def test_coe2rv2coe(self):
        tol = 1e-11

        m = 10000
        p = npr.rand(m, 1) * 10
        e = npr.rand(m, 1)
        i = npr.rand(m, 1) * np.pi
        W = npr.rand(m, 1) * 2*np.pi
        w = npr.rand(m, 1) * 2*np.pi
        f = npr.rand(m, 1) * 2*np.pi
        COE_1 = np.concatenate((p, e, i, W, w, f), 1)

        RV = orb.coe2rv(COE_1)
        COE_2 = orb.rv2coe(RV)

        COE_diff = diff_elements(COE_1, COE_2, angle_idx=[2, 3, 4, 5])
        print(COE_diff)

        self.assertTrue((np.fabs(COE_diff) < tol).all())

    def test_rv2coe2rv(self):
        tol = 1e-9

        m = 10000
        p = npr.rand(m, 1) * 10
        e = npr.rand(m, 1)
        i = npr.rand(m, 1) * np.pi
        W = npr.rand(m, 1) * 2*np.pi
        w = npr.rand(m, 1) * 2*np.pi
        f = npr.rand(m, 1) * 2*np.pi
        COE_0 = np.concatenate((p, e, i, W, w, f), 1)

        RV_1 = orb.coe2rv(COE_0)

        COE = orb.rv2coe(RV_1)
        RV_2 = orb.coe2rv(COE)

        RV_diff = RV_1 - RV_2
        print(RV_diff)

        self.assertTrue((np.fabs(RV_diff) < tol).all())

    def test_euler_sequence(self):
        tol = 1e-14

        axes = [1, 2, 3]
        a = np.array([[np.pi/2], [0.], [0.]])
        b = np.array([[0.], [np.pi/2], [0.]])
        c = np.array([[0.], [0.], [np.pi/2]])

        C = orb.euler_sequence(axes, *[a, b, c])
        print(C)

        C0 = np.array([[1., 0., 0.],
                       [0., 0., 1.],
                       [0., -1., 0.]])
        C1 = np.array([[0., 0., -1.],
                       [0., 1., 0.],
                       [1., 0., 0.]])
        C2 = np.array([[0., 1., 0.],
                       [-1., 0., 0.],
                       [0., 0., 1.]])
        result = (np.fabs(C[0] - C0) < tol).all()
        result = result and (np.fabs(C[1] - C1) < tol).all()
        result = result and (np.fabs(C[2] - C2) < tol).all()
        self.assertTrue(result)
