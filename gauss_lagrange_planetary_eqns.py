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
        Time history matrix (mx6) of MEE [p f g h k L], where
        m is the number of samples.

        Output
        ------
        G : numpy.matrix
        A 6x3 matrix of each element's time derivative as a result of
        disturbances in the r, theta, and angular momentum directions.
        """
        G = []
        for i, x in enumerate(X.tolist()):
            p = x[0]
            g = x[2]
            f = x[1]
            h = x[3]
            k = x[4]
            L = x[5]

            try:
                s = (1. + h**2 + k**2)**.5
            except OverflowError:
                import pdb;pdb.set_trace()
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

            G.append(np.matrix([pdot, fdot, gdot, hdot, kdot, Ldot]) * rt_p_mu)
        return G

    def coe(self, X):
        """Gauss's form of Lagrange's Planetary Equations for COEs.

        Input
        -----
        X : numpy.matrix
        Time history matrix (mx6) of COE [p e i W w f], where
        m is the number of samples.

        Output
        ------
        G : numpy.matrix
        A 6x3 matrix of each element's time derivative as a result of
        disturbances in the r, theta, and angular momentum directions.
        """
        G = []
        for i, x in enumerate(X.tolist()):
            p = x[0]
            e = x[2]
            i = x[1]
            W = x[3]
            w = x[4]
            f = x[5]

            sf = sin(f)
            cf = cos(f)
            st = sin(f + w)
            ct = cos(f + w)
            si = sin(i)
            ci = cos(i)
            r = p / (1. + e*cf)
            h = (self.mu * p)**.5

            adot = np.matrix([e*sf, p/r, 0.]) * 2*a**2/h
            edot = np.matrix([p*sf, (p+r)*cf + r*e, 0.]) / h
            pdot = adot*(1-e**2) - 2*a*e*edot
            idot = np.matrix([0., 0., r*ct/h])
            Wdot = np.matrix([0., 0., r*st/h/si])
            wdot = np.matrix([-p*cf/e, (p+r)*sf/e, -r*st*ci/si]) / h
            fdot = np.matrix([p*cf, -(p+r)*sf, 0.]) /h/e

            G.append(np.matrix([pdot, edot, idot, Wdot, wdot, fdot]))
        return G
