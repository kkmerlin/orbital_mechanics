"""Created on Wed Sep 08 2015 22:27.

@author: Nathan Budd
"""
import numpy as np
import numpy.linalg as npl
from math import cos, sin


class GaussVariationalEqns():
    """Generate the Gauss Variational Equations.

    A collection of methods for the Gauss Variational Equations
    in different element sets. A set of state histories is passed as input, and
    the output is a list of GVE matrices, mapping LVLH frame
    accelerations into orbital element derivatives.

    Instance Members
    -------
    mu : float
        The standard gravitational parameter
    element_set : string
        See two_body.py for more details.
    """

    def __init__(self, mu, element_set):
        """."""
        self.mu = mu
        self.element_set = element_set

    def __call__(self, X):
        """Gauss Variational Equations

        Input
        -----
        X : ndarray
        Time history array (mx6) of elements, where
        m is the number of samples.

        Output
        ------
        G : list of ndarray
        A 6x3 array of each element's time derivative as a result of
        disturbances in the r, theta, and angular momentum directions.
        """
        G_funcs = {'mee': self._mee,
                   'coe': self._coe,
                   'rv': self._rv}

        return G_funcs[self.element_set](X)

    def _mee(self, X):
        """Gauss Variational Equations for MEEs.

        Input
        -----
        X : ndarray
        Time history array (mx6) of MEE [p f g h k L], where
        m is the number of samples.

        Output
        ------
        G : list of ndarray
        A 6x3 array of each element's time derivative as a result of
        disturbances in the r, theta, and angular momentum directions.
        """
        G = [np.array([[]]) for x in X]
        for i, x in enumerate(X):
            p = x[0]
            f = x[1]
            g = x[2]
            h = x[3]
            k = x[4]
            L = x[5]

            sL = sin(L)
            cL = cos(L)
            s = (1. + h**2 + k**2)**.5
            w = 1. + f*cL + g*sL
            rt_p_mu = (p/self.mu)**.5

            pdot = [0., 2*p/w, 0.]
            fdot = [sL, ((w+1)*cL + f)/w, -g*(h*sL - k*cL)/w]
            gdot = [-cL, ((w+1.)*sL + g)/w, f*(h*sL - k*cL)/w]
            hdot = [0., 0., s*s*cL/2/w]
            kdot = [0., 0., s*s*sL/2/w]
            Ldot = [0., 0., (h*sL - k*cL)/w]

            G[i] = np.array([pdot, fdot, gdot, hdot, kdot, Ldot]) * rt_p_mu
        return G

    def _coe(self, X):
        """Gauss Variational Equations for COEs.

        Input
        -----
        X : ndarray
        Time history array (mx6) of COE [p e i W w f], where
        m is the number of samples.

        Output
        ------
        G : list of ndarray
        A 6x3 array of each element's time derivative as a result of
        disturbances in the r, theta, and angular momentum directions.
        """
        G = [np.zeros((6, 3)) for x in X]
        for k, x in enumerate(X):
            p = x[0]
            e = x[1]
            i = x[2]
            W = x[3]
            w = x[4]
            f = x[5]

            sf = sin(f)
            cf = cos(f)
            st = sin(f + w)
            ct = cos(f + w)
            si = sin(i)
            ci = cos(i)
            a = p / (1. - e**2)
            r = p / (1. + e*cf)
            h = (self.mu * p)**.5

            adot = np.array([e*sf, p/r, 0.]) * 2*a**2/h
            edot = np.array([p*sf, (p+r)*cf + r*e, 0.]) / h
            pdot = adot*(1-e**2) - 2*a*e*edot
            idot = np.array([0., 0., r*ct/h])
            Wdot = np.array([0., 0., r*st/h/si])
            wdot = np.array([-p*cf/e, (p+r)*sf/e, -r*st*ci/si]) / h
            fdot = np.array([p*cf, -(p+r)*sf, 0.]) / h / e

            G[k][0] = pdot
            G[k][1] = edot
            G[k][2] = idot
            G[k][3] = Wdot
            G[k][4] = wdot
            G[k][5] = fdot
        return G

    def _rv(self, X):
        """Gauss Variational Equations for RV.

        Input
        -----
        X : ndarray
        Time history array (mx6) of COE [p e i W w f], where
        m is the number of samples.

        Output
        ------
        G : list of ndarray
        A 6x3 array of each element's time derivative as a result of
        disturbances in the r, theta, and angular momentum directions.
        """
        G = [np.zeros((6, 3)) for x in X]
        for k, x in enumerate(X):
            r = x[0:3].reshape((3, 1))
            v = x[3:6].reshape((3, 1))
            h = np.cross(r, v, axis=0)

            i_r = r / npl.norm(r)
            i_h = h / npl.norm(h)
            i_theta = np.cross(i_h, i_r, axis=0)

            G[k][3:6, 0:] = np.concatenate((i_r, i_theta, i_h), 1)

        return G

        def __repr__(self):
            """Printable represenation of the object."""
            return 'GaussVariationalEqns({})'.format(self.mu)

        def __str__(self):
            """Human readable represenation of the object."""
            return 'GaussVariationalEqns(mu={})'.format(self.mu)
