"""Created on Wed Sep 08 2015 15:10.

@author: Nathan Budd
"""
import numpy as np
import numpy.linalg as npl
import numpy.matlib as npm
from math import sin, cos, atan2
from .model_abstract import ModelAbstract
from .gauss_lagrange_planetary_eqns import GaussLagrangePlanetaryEqns as GLPE


class LyapunovElementSteering(ModelAbstract):
    """Lyapunov control for orbital elements, neglecting phase angle.

    Phase angle element must be the last (6th) state listed.

    Instance Members
    ----------------
    mu : float
    Standard gravitational parameter

    W : numpy.matrix
    Diagonal nxn weight matrix, where n is the state dimension.

    a_t : float
    Thrust magnitude

    Xref : ReferenceCOE or ReferenceMEE object
    Can be called with input T (an mx1 numpy.matrix) to produce a reference
    trajectory, X (mxn numpy.matrix) of reference states, where n is the state
    dimension defined by the Xref.model attribute.
    """

    def __init__(self, mu, W, a_t, Xref):
        """."""
        self.mu = mu
        self.W = W
        self.a_t = a_t
        self.Xref = Xref
        super().__init__()

    def __call__(self, T, X):
        """Evaluate the control at the given times.

        X = [e1 e2 e3 e4 e5 e_phase]

        See dynamics_abstract.py for more details.
        """
        G = GLPE(self.mu).coe(X)
        Eta = X - self.Xref(T)

        U = npm.zeros(X.shape)
        for i, eta in enumerate(Eta):
            # Vdot = c'*u, where u is a unit vector
            c = eta * self.W * G[i]
            try:
                u = - c.T / npl.norm(c)
            except RuntimeWarning:
                u = npm.zeros(c.T.shape)

            U[i, :] = (self.a_t * G[i] * u).T

        self.Xdot = U
        return U

        def __repr__(self):
            """Printable represenation of the object."""
            return 'LyapunovElementSteering({}, {}, {}, {})'.format(
                self.mu, self.W, self.a_t, self.Xref)

        def __str__(self):
            """Human readable represenation of the object."""
            output = 'LyapunovElementSteering'
            output += '(mu={}, W={}, a_t={}, Xref={})'.format(
                self.mu, self.W, self.a_t, self.Xref)
            return output
