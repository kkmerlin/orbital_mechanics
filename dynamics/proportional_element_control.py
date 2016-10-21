"""Created on Wed Sep 08 2015 15:10.

@author: Nathan Budd
"""
import numpy as np
from numpy import dot
import numpy.linalg as npl
from math import sin, cos, atan2
from .model_abstract import ModelAbstract
from orbit import diff_elements
from .diff_elements_theta_into_p import diff_elements_theta_into_p


class ProportionalElementControl(ModelAbstract):
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
    glpe : callable
        Can be called with input X (mxn ndarray) to produce a list of ndarrays
        representing the Gauss's form of Lagrange's Planetary Equations for
        each passed state.
    u : ndarray
        Cartesian control history mx3 where m is the number of samples and 3 is
        the control dimension.
    """

    def __init__(self, mu, K, a_t, Xref, glpe):
        """."""
        self.mu = mu
        self.K = K
        self.a_t = a_t
        self.Xref = Xref
        self.glpe = glpe
        self.u = np.zeros(())
        super().__init__()

    def __call__(self, T, X):
        """Evaluate the control at the given times.

        X = [e1 e2 e3 e4 e5 e_phase]

        See dynamics_abstract.py for more details.
        """
        G = self.glpe(X)
        Xref = self.Xref(T)
        Eta = diff_elements(X, Xref, angle_idx=[2, 3, 4, 5])
        # k = 100.
        # Eta = diff_elements_theta_into_p(self.mu, k,
        #                                  X, Xref, angle_idx=[2, 3, 4, 5])

        U = np.zeros(X.shape)
        self.u = np.zeros((X.shape[0], 3))
        for i, eta in enumerate(Eta):
            K = np.eye(6)
            eta_trimmed = np.trim_zeros(np.sort(np.absolute(eta[0:5])))
            max_exponent = np.max(np.log10(eta_trimmed))
            try:
                exponent_5 = max_exponent  # - np.log10(eta[5])
            except RuntimeWarning:  # if eta[5] is zero
                exponent_5 = max_exponent
            K[5, 5] = np.power(10., exponent_5+1)

            # u = (-1./self.a_t * npl.inv(G[i].T @ G[i]) @ G[i].T @ self.K @ eta)
            u = (-1./self.a_t * npl.inv(G[i].T @ G[i]) @ G[i].T @ K @ eta)
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
                self.mu, self.K, self.a_t, self.Xref, self.glpe)

        def __str__(self):
            """Human readable represenation of the object."""
            output = 'ProportionalElementControl'
            output += '(mu={}, K={}, a_t={}, Xref={}, glpe={})'.format(
                self.mu, self.K, self.a_t, self.Xref, self.glpe)
            return output
