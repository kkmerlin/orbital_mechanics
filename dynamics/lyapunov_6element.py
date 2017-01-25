"""Created on Tue Jan 24 2017 13:17.

@author: Nathan Budd
"""
import numpy as np
import numpy.linalg as npl
from ..orbit import diff_elements
from .two_body import TwoBody
from .utilities import GaussVariationalEqns


class Lyapunov6Element():
    """
    Lyapunov control for orbital elements, tracking a target with a phase
    that steps by one orbit every target period.

    Phase angle element must be the last (6th) state listed.

    Members
    -------
    mu : float
        Standard gravitational parameter
    W : ndarray
        Diagonal nxn weight array, where n is the state dimension.
    a_t : float
        Thrust magnitude
    Xref : callable
        Can be called with input T (an mx1 ndarray) to produce a reference
        trajectory, X (mxn ndarray), where n is the state dimension defined by
        the Xref.model attribute.
    model : callable
        Can be called with input T (an mx1 ndarray) and X (mxn ndarray)
        to produce state derivatives for this element set.
    gve : callable
        Can be called with input X (mxn ndarray) to produce a 3darray
        representing the Gauss's form of Lagrange's Planetary Equations for
        each passed state.
    u : ndarray
        Cartesian control history mx3 where m is the number of samples and 3 is
        the control dimension.
    V : ndarray
        Most recent Lyapunov function trajectory, mx1 where m is the number of
        samples.
    Vdot : ndarray
        Most recent Lyapunov function derivative history, mx1 where m is the
        number of samples.
    Xdot : ndarray
        The most recently computed call output
    X0 : ndarray
        See two_body.py for more details.
    Eta_last : ndarray
        The most recently computed element errors, including modulo 2pi phase
    Xref_last : ndarray
        The most recently computed reference trajectory
    """

    def __init__(self, mu, W, a_t, element_set, X0):
        """.

        Parameters
        ----------
        element_set : string
            See two_body.py for more details.
        """
        self.mu = mu
        self.W = W
        self.a_t = a_t
        self.X0 = X0
        self.model = TwoBody(mu, element_set)
        self.gve = GaussVariationalEqns(mu, element_set)
        self.u = np.zeros(())
        self.V = np.zeros(())
        self.Vdot = np.zeros(())
        self.Xdot = np.array([[]])
        self.Eta_last = np.array([[]])
        self.Xref_last = np.array([[]])

    def __call__(self, T, X):
        """Evaluate the control at the given times.

        X = [e1 e2 e3 e4 e5 e_phase]

        See dynamics_abstract.py for more details.
        """
        # construct step-altering target
        a_ref = self.X0[0, -1]
        nu_ref = self.X0[0, 5]
        period_ref = 2.*np.pi*(a_ref**3 / self.mu)**.5
        Xref_nu_free = np.ones(T.shape) @ self.X0[0:1, 0:5]
        phase_factors = [int(t/period_ref) for t in T]
        Xref_nu = np.array([[nu_ref + 2*np.pi*n for n in phase_factors]]).T
        Xref = np.concatenate((Xref_nu_free, Xref_nu), axis=1)
        self.Xref_last = Xref

        # or...
        # Xref = TwoBody(self.mu, 'coea', X0=self.X0)

        # compute errors and controls
        G = self.gve(X)
        Eta = diff_elements(X, Xref, angle_idx=[2, 3, 4])
        self.Eta_last = diff_elements(X, Xref, angle_idx=[2, 3, 4, 5])

        Xdot = self.model(T, X)
        Xrefdot = self.model(T, Xref)
        Etadot = Xdot - Xrefdot

        U = np.zeros(X.shape)
        self.u = np.zeros((X.shape[0], 3))
        self.V = np.zeros((X.shape[0], 1))
        self.Vdot = np.zeros((X.shape[0], 1))
        for i, eta in enumerate(Eta):
            # Vdot = c'*u, where u is a unit vector
            c = (eta @ self.W @ G[i]).reshape((1, 3))
            c_norm = npl.norm(c)
            if c_norm > 1:
                u = - c.T / c_norm
            else:
                u = -c.T
            # u = -c.T / c_norm

            self.u[i] = u.T
            self.V[i] = eta @ self.W @ eta.T
            self.Vdot[i] = eta @ self.W @ (Etadot[i:i+1, 0:].T + self.a_t *
                                           G[i] @ u)
            U[i:i+1] = (self.a_t * G[i] @ u).T

        self.Xdot = U
        return U

        def __repr__(self):
            """Printable represenation of the object."""
            return 'LyapunovElementSteering({}, {}, {}, {}, {})'.format(
                self.mu, self.W, self.a_t, self.Xref, self.gve)

        def __str__(self):
            """Human readable represenation of the object."""
            output = 'LyapunovElementSteering'
            output += '(mu={}, W={}, a_t={}, Xref={}, glpe={})'.format(
                self.mu, self.W, self.a_t, self.Xref, self.gve)
            return output
