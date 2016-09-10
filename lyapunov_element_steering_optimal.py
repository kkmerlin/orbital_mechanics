"""Created on Wed Sep 08 2015 15:10.

@author: Nathan Budd
"""
import numpy as np
import numpy.linalg as npl
import numpy.matlib as npm
from math import sin, cos, atan2
from .model_abstract import ModelAbstract
from .gauss_lagrange_planetary_eqns import GaussLagrangePlanetaryEqns as GLPE


class LyapunovElementSteeringOptimal(ModelAbstract):
    """Lyapunov control for orbital elements, neglecting phase angle.

    Phase angle element must be the last (6th) state listed.

    Instance Members
    ----------------
    mu : float
    Standard gravitational parameter

    a_t : float
    Thrust magnitude

    Xref : numpy.matrix
    Reference state history
    """

    def __init__(self, mu, a_t, Xref):
        """."""
        self.mu = mu
        self.a_t = a_t
        self.Xref = Xref
        super().__init__()

    def __call__(self, T, X):
        """Evaluate the control at the given times.

        X = [e1 e2 e3 e4 e5 e_phase]

        See dynamics_abstract.py for more details.
        """
        W = np.matrix(np.diag([1.]*5 + [0.]))
        M = GLPE(self.mu).coe(X)
        Eta = X - self.Xref

        U = npm.zeros(X.shape)
        for i, eta in enumerate(Eta):
            # Vdot = c'*u, where u is a unit vector
            # solve for the optimal unit vector direction
            c = self.a_t * eta * W * M[i]
            alpha1 = atan2(c[0, 1], c[0, 0])
            ca1 = cos(alpha1)
            sa1 = sin(alpha1)
            alpha2 = atan2((c[0, 0]*ca1 + c[0, 1]*sa1), c[0, 2])
            ca2 = cos(alpha2)
            sa2 = sin(alpha2)
            u = np.matrix([[ca1*sa2],
                           [sa1*sa2],
                           [ca2]])

            U[i, :] = (M[i] * u).T

        self.Xdot = U
        return U
