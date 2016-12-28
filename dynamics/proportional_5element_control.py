"""Created on Wed Sep 08 2015 15:10.

@author: Nathan Budd
"""
import numpy as np
from numpy import dot
import numpy.linalg as npl
from orbital_mechanics.orbit import diff_elements
from .utilities import diff_elements_theta_into_p
from .utilities import GaussVariationalEqns
from .two_body import TwoBody


class Proportional5ElementControl():
    """
    Proportional control for orbital element orbit transfer.

    Phase angle element must be the last (6th) state listed.

    Members
    -------
    mu : float
        Standard gravitational parameter
    K : ndarray
        Diagonal nxn gain array, where n is the state dimension.
    a_t : float
        Thrust magnitude
    Xref : callable
        Can be called with input T (an mx1 ndarray) to produce a reference
        trajectory, X (mxn ndarray), where n is the state dimension defined by
        the Xref.model attribute.
    gve : callable
        Can be called with input X (mxn ndarray) to produce a list of ndarrays
        representing the Gauss's form of Lagrange's Planetary Equations for
        each passed state.
    u : ndarray
        Cartesian control history mx3 where m is the number of samples and 3 is
        the control dimension.
    Xdot : ndarray
        The most recently computed call output
    """

    def __init__(self, mu, K, a_t, element_set, X0):  # Xref, gve):
        """.

        Parameters
        ----------
        element_set : string
            See two_body.py for more details.
        X0 : ndarray
            See two_body.py for more details.
        """
        self.mu = mu
        self.K = K
        self.a_t = a_t
        self.Xref = TwoBody(mu, element_set, X0=X0)
        self.gve = GaussVariationalEqns(mu, element_set)
        self.u = np.zeros(())
        self.Xdot = np.array([[]])

    def __call__(self, T, X):
        """Evaluate the control at the given times.

        X = [e1 e2 e3 e4 e5 e_phase]

        See dynamics_abstract.py for more details.
        """
        G = self.gve(X)
        Xref = self.Xref(T)
        k = 1e-1
        Eta = diff_elements_theta_into_p(self.mu, k,
                                         X, Xref, angle_idx=[2, 3, 4, 5])

        U = np.zeros(X.shape)
        self.u = np.zeros((X.shape[0], 3))
        for i, eta in enumerate(Eta):
            u = (-1./self.a_t * npl.inv(G[i].T @ G[i]) @ G[i].T @ self.K @ eta)
            u_norm = npl.norm(u)
            if u_norm > 1.:
                self.u[i] = u / u_norm
            else:
                self.u[i] = u

            U[i] = (self.a_t * G[i] @ self.u[i]).T

        self.Xdot = U
        return U

        def __repr__(self):
            """Printable represenation of the object."""
            return 'ProportionalElementControl({}, {}, {}, {}, {})'.format(
                self.mu, self.K, self.a_t, self.Xref, self.gve)

        def __str__(self):
            """Human readable represenation of the object."""
            output = 'ProportionalElementControl'
            output += '(mu={}, K={}, a_t={}, Xref={}, gve={})'.format(
                self.mu, self.K, self.a_t, self.Xref, self.gve)
            return output
