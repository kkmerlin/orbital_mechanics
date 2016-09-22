"""Created on Wed Sep 21 2015 14:23.

@author: Nathan Budd
"""
import numpy as np
from math import sin
from math import cos
from math import fabs


class Orbit():
    """A collection of methods related to orbital elements.

    Methods are all intended for batch operations on a history or trajectory
    of orbital element states.

    Instance Members
    -------
    mu : float
    The standard gravitational parameter
    """

    def __init__(self, mu):
        """."""
        self.mu = mu

    def M2f_ellipse(self, coe_M):
        """Convert elliptic mean anomaly, M,  to true anomaly, f.

        Input
        -----
        coe_M : ndarray
        mx6 array of classical orbital elements [p e i W w M]. m is the number
        of samples and 6 is the dimension of the element set.

        Output
        ------
        coe_f : ndarray
        mx6 array of classical orbital elements [p e i W w f]. m is the number
        of samples and 6 is the dimension of the element set.
        """
        coe_E = self.M2E(coe_M)
        coe_f = self.E2f(coe_E)
        return coe_f

    def M2E(self, coe_M):
        """Convert elliptic mean anomaly, M,  to eccentric anomaly, E.

        Input
        -----
        coe_M : ndarray
        mx6 array of classical orbital elements [p e i W w M]. m is the number
        of samples and 6 is the dimension of the element set.

        Output
        ------
        coe_E : ndarray
        mx6 array of classical orbital elements [p e i W w E]. m is the number
        of samples and 6 is the dimension of the element set.
        """
        tol = 1e-14
        e = coe_M[0:, 1:2]
        M = coe_M[0:, -1:]
        E0 = M + np.sign(np.sin(M))*e
        E1 = E0 + 1.

        while np.max(np.absolute(E1 - E0)) > tol:
            E0 = E1
            E1 = E0 - (M - E0 + e*np.sin(E0)) / (-1. + e*np.cos(E0))

        coe_E = np.concatenate((coe_M[0:, 0:-1], E1), 1)
        return coe_E

    def E2f(self, coe_E):
        """Convert eccentric anomaly, E, to true anomaly, f

        Input
        -----
        coe_E : ndarray
        mx6 array of classical orbital elements [p e i W w E]. m is the number
        of samples and 6 is the dimension of the element set.

        Output
        ------
        coe_f : ndarray
        mx6 array of classical orbital elements [p e i W w f]. m is the number
        of samples and 6 is the dimension of the element set.
        """
        e = coe_E[0:, 1:2]
        E = coe_E[0:, -1:]
        tan_f_by_2 = ((1.+e)/(1.-e))**.5 * np.tan(E/2)
        f = 2 * np.arctan(tan_f_by_2)
        coe_f = np.concatenate((coe_E[0:, 0:-1], f), 1)
        return coe_f
