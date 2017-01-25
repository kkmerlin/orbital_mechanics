"""Created on Sun Jan 22 2017 16:25.

@author: Nathan Budd
"""
import numpy as np
import numpy.linalg as npl
from math import sin, cos, atan2
from ..orbit import diff_elements
from .two_body import TwoBody
from .utilities import GaussVariationalEqns


class LyapunovPositionVelocity():
    """
    Lyapunov control for position-velocity, using a spline trajectory for
    tracking.

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
    spline_interval : float
        Interval of time allowed to pass before creating a new spline.
    """

    def __init__(self, mu, W, a_t, X0, t_spline):
        """.

        Parameters
        ----------
        element_set : string
            See two_body.py for more details.
        """
        element_set = 'rv'
        self.mu = mu
        self.W = W
        self.a_t = a_t
        self.Xref = Xref(mu, element_set, X0=X0)
        self.model = TwoBody(mu, element_set)
        self.gve = GaussVariationalEqns(mu, element_set)
        self.u = np.zeros(())
        self.V = np.zeros(())
        self.Vdot = np.zeros(())
        self.Xdot = np.array([[]])
        self.spline_interval = spline_interval

    def __call__(self, T, X):
        """Evaluate the control at the given times.

        X = [rx ry rz vx vy vz]

        See two_body.py for more details.
        """
        # generage splines
        steps = int(T[-1]/self.spline_interval)
        spline_starts = [j*spline_interval for j in range(steps)]


        G = self.gve(X)
        Eta = diff_elements(X, Xref, angle_idx=[2, 3, 4, 5])
        dr = Eta[:, 0:3]
        dv = Eta[:, 3:6]

        U = np.zeros(X.shape)
        self.u = np.zeros((X.shape[0], 3))
        self.V = np.zeros((X.shape[0], 1))
        self.Vdot = np.zeros((X.shape[0], 1))
        for i, eta in enumerate(Eta):
            # Vdot = c'*u, where u is a unit vector
            c = (eta @ self.W @ G[i]).reshape((1, 3))
            u = - c.T / npl.norm(c)

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
