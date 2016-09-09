"""Created on Wed Sep 08 2015 22:27.

@author: Nathan Budd

A collection of methods for Gauss's form of Lagrange's Planetary Equations in
different element sets.
"""
import numpy as np
from math import cos, sin


class GaussLagrangePlanetaryEqns():
    """MEE dynamics.

    Instance Members
    -------
    mu : float
    The standard gravitational parameter
    """

    def __init__(self, mu):
        """."""
        self.mu = mu

    def mee(self, elements):
        """Gauss's form of Lagrange's Planetary Equations for MEEs.

        Input
        -----
        elements : numpy.matrix
        Row vector of MEE elements [p f g h k L mu]

        Output
        ------
        M : numpy.matrix
        A 6x3 matrix of each element's time derivative as a result of
        disturbances in the r, theta, and angular momentum directions.
        """
        p = elements[0, 0]
        g = elements[0, 2]
        f = elements[0, 1]
        h = elements[0, 3]
        k = elements[0, 4]
        L = elements[0, 5]

        s = (1. + h**2 + k**2)**.5
        sL = sin(L)
        cL = cos(L)
        w = 1. + f*cL + g*sL
        rt_p_mu = (p/self.mu)**.5

        pdot = [0., 2*p/w*rt_p_mu, 0.]
        fdot = [sL, ((w+1self.)*cL + f)/w, -g*(h*sL - k*cL)/w] * rt_p_mu
        gdot = [-cL, ((w+1.)*cL + g)/w, -f*(h*sL - k*cL)/w] * rt_p_mu
        hdot = [0., 0., s*s*cL/2/w] * rt_p_mu
        kdot = [0., 0., s*s*sL/2/w] * rt_p_mu
        Ldot = [0., 0., (h*sL - k*cL)/w] * rt_p_mu

        M = np.matrix([pdot, fdot, gdot, hdot, kdot, Ldot])
        return M
