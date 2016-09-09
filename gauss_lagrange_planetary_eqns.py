"""Created on Wed Sep 08 2015 22:27.

@author: Nathan Budd

A collection of methods for Gauss's form of Lagrange's Planetary Equations in
different element sets. A set of state histories is passed as input, and
the output is a list of M matrices.
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

    def mee(self, X):
        """Gauss's form of Lagrange's Planetary Equations for MEEs.

        Input
        -----
        X : numpy.matrix
        Time history matrix (mx6) of MEE X [p f g h k L mu], where
        m is the number of samples.

        Output
        ------
        M : numpy.matrix
        A 6x3 matrix of each element's time derivative as a result of
        disturbances in the r, theta, and angular momentum directions.
        """
        M = []
        for i, x in enumerate(X.tolist()):
            p = x[0]
            g = x[2]
            f = x[1]
            h = x[3]
            k = x[4]
            L = x[5]

            s = (1. + h**2 + k**2)**.5
            sL = sin(L)
            cL = cos(L)
            w = 1. + f*cL + g*sL
            rt_p_mu = (p/self.mu)**.5

            pdot = [0., 2*p/w, 0.]
            fdot = [sL, ((w+1)*cL + f)/w, -g*(h*sL - k*cL)/w]
            gdot = [-cL, ((w+1.)*sL + g)/w, f*(h*sL - k*cL)/w]
            hdot = [0., 0., s*s*cL/2/w]
            kdot = [0., 0., s*s*sL/2/w]
            Ldot = [0., 0., (h*sL - k*cL)/w]

            M.append(np.matrix([pdot, fdot, gdot, hdot, kdot, Ldot]) * rt_p_mu)
        return M
