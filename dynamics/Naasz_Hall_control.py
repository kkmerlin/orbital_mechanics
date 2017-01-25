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


class NaaszHallControl():
    """
    Proportional control based on the work of Naasz and Hall, 2002.

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
        Can be called with input X (mxn ndarray) to produce a 3darray
        representing the Gauss's form of Lagrange's Planetary Equations for
        each passed state.
    u : ndarray
        Cartesian control history mx3 where m is the number of samples and 3 is
        the control dimension.
    Xdot : ndarray
        The most recently computed call output
    """

    def __init__(self, mu, a_t, element_set, X0):  # Xref, gve):
        """.

        Parameters
        ----------
        element_set : string
            See two_body.py for more details.
        X0 : ndarray
            See two_body.py for more details.
        """
        self.mu = mu
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
        G = self.gve(X)[:, 0:5]
        Xref = self.Xref(T)
        Eta = diff_elements(X, Xref, angle_idx=[2, 3, 4, 5])

        # INCORPORATE f error INTO p error
        K_n = 1e-4
        a_r = Xref[:, 0]
        f_eta = Eta[:, 5]
        a_r_aug = (-K_n*f_eta + a_r**(-3/2))**(-2/3)
        Eta[:, 0] = X[:, 0] - a_r_aug
        Eta = Eta[:, 0:5]

        a = Xref[0, 0]
        e = Xref[0, 1]
        i = Xref[0, 2]
        w = Xref[0, 4]
        p = a * (1-e**2)
        h = (self.mu * p)**0.5
        b = a * (1 - e**2)**0.5

        T2 = T[1:]
        dT = T*1
        dT[0:-1] = T2 - T[0:-1]
        dT[-1] = dT[0]
        dT = dT.reshape(dT.shape[0],)

        K_a = h**2 / (4 * a**4 * (1+e**2) * dT)
        K_e = h**2 / (4 * p**2 * dT)
        K_i = ((h + e*h*np.cos(w + np.arcsin(e*np.sin(w)))) /
               (p * (e**2 * np.sin(w)**2 - 1)))**2 / dT
        K_W = ((h * np.sin(i) * (e * np.sin(w + np.arcsin(e*np.cos(w))) - 1)) /
               (p * (1 - e**2 * np.cos(w)**2)))**2 / dT
        K_w = e**2 * h**2 / 4 / p**2 * (1 - e**2/4) / dT
        K_f = (a * e * h / 2 / b / p)**2 * (1 - e**2/4) * dT

        n_T = dT.shape[0]
        n_X = Eta.shape[1]
        K = np.zeros((n_T, n_X, n_X))
        K[:, 0, 0] = K_a
        K[:, 1, 1] = K_e
        K[:, 2, 2] = K_i
        K[:, 3, 3] = K_W
        K[:, 4, 4] = K_w
        # K[:, 5, 5] = K_f

        U = np.zeros(X.shape)
        self.u = np.zeros((X.shape[0], 3))
        for i, eta in enumerate(Eta):
            u = (-1./self.a_t * npl.inv(G[i].T @ G[i]) @ G[i].T @ K[i] @ eta)
            u_norm = npl.norm(u)
            if u_norm > 1.:
                self.u[i] = u / u_norm
            else:
                self.u[i] = u

            U[i, 0:5] = (self.a_t * G[i] @ self.u[i]).T

        self.Xdot = U
        return U

        def __repr__(self):
            """Printable represenation of the object."""
            return 'ProportionalElementControl({}, {}, {}, {}, {})'.format(
                self.mu, K, self.a_t, self.Xref, self.gve)

        def __str__(self):
            """Human readable represenation of the object."""
            output = 'ProportionalElementControl'
            output += '(mu={}, K={}, a_t={}, Xref={}, gve={})'.format(
                self.mu, K, self.a_t, self.Xref, self.gve)
            return output
