"""Created on Sun Jan 22 2015 13:31.

@author: Nathan Budd
"""
import numpy as np
import numpy.linalg as npl
from ..orbit import diff_elements
from .two_body import TwoBody
from .utilities import GaussVariationalEqns


class LyapunovElementSlidingA():
    """
    Lyapunov control for orbital elements, incorporating phase angle into a
    sliding semi-major axis that increases/decreases the angular rate to slowly
    draw the current state not just to the correct orbit, but to the correct
    angular position in that orbit as well.

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
    Xref_last : ndarray
        The most recently computed reference trajectory
    Eta_aug : ndarray
        The most recently computed element errors
    k_a0 : float
        Gain for augmented semi-major axis
    """

    def __init__(self, mu, W, a_t, element_set, X0, k_a0):
        """.

        Parameters
        ----------
        element_set : string
            See two_body.py for more details.
        X0 : ndarray
            See two_body.py for more details.
        """
        self.mu = mu
        self.W = W
        self.a_t = a_t
        self.Xref = TwoBody(mu, element_set, X0=X0)
        self.model = TwoBody(mu, element_set)
        self.gve = GaussVariationalEqns(mu, element_set)
        self.u = np.zeros(())
        self.V = np.zeros(())
        self.Vdot = np.zeros(())
        self.Xdot = np.array([[]])
        self.Xref_last = np.array([[]])
        self.Eta_aug = np.array([[]])
        self.k_a0 = k_a0

    def __call__(self, T, X):
        """Evaluate the control at the given times.

        X = [e1 e2 e3 e4 e5 e_phase]

        See dynamics_abstract.py for more details.
        """
        G = self.gve(X)
        Xref = self.Xref(T)
        Eta = diff_elements(X, Xref, angle_idx=[2, 3, 4, 5])

        # augment target semi-major axis
        a_t = Xref[:, 0]
        d_nu = Eta[:, 5]
        # k_a = 1 + (self.k_a0-1)/np.pi*d_nu
        k_a_coeff = [1,
                     0,
                     3*(self.k_a0-1)/np.pi**2,
                     2*(1-self.k_a0)/np.pi**3]
        k_a = (1*k_a_coeff[0] + d_nu*k_a_coeff[1] + d_nu**2*k_a_coeff[2] +
               d_nu**3*k_a_coeff[3])
        a_T = a_t# * (1 + d_nu/(k_a*np.pi))
        Xref_aug = np.array(Xref)
        Xref_aug[:, 0] = a_T
        self.Eta_aug = diff_elements(X, Xref_aug, angle_idx=[2, 3, 4, 5])
        self.Xref_last = Xref_aug

        Xdot = self.model(T, X)
        Xrefdot = self.model(T, Xref)
        Etadot = Xdot - Xrefdot

        U = np.zeros(X.shape)
        self.u = np.zeros((X.shape[0], 3))
        self.V = np.zeros((X.shape[0], 1))
        self.Vdot = np.zeros((X.shape[0], 1))
        for i, eta in enumerate(Eta):
            # Vdot = c'*u, where u is a unit vector
            c = (self.Eta_aug[i] @ self.W @ G[i]).reshape((1, 3))
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
