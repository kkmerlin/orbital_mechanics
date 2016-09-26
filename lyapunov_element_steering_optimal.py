"""Created on Wed Sep 08 2015 15:10.

@author: Nathan Budd
"""
import numpy as np
import numpy.linalg as npl
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

    W : numpy.array
    Diagonal nxn weight array, where n is the state dimension.

    a_t : float
    Thrust magnitude

    Xref : ReferenceCOE or ReferenceMEE object
    Can be called with input T (an mx1 numpy.array) to produce a reference
    trajectory, X (mxn numpy.array) of reference states, where n is the state
    dimension defined by the Xref.model attribute.
    """

    def __init__(self, mu, W, a_t, Xref):
        """."""
        self.mu = mu
        self.W = W
        self.a_t = a_t
        self.Xref = Xref
        self.u = np.zeros(())
        super().__init__()

    def __call__(self, T, X):
        """Evaluate the control at the given times.

        X = [e1 e2 e3 e4 e5 e_phase]

        See dynamics_abstract.py for more details.
        """
        G = GLPE(self.mu).coe(X)
        Eta = X - self.Xref(T)

        U = np.zeros(X.shape)
        self.u = np.zeros((X.shape[0], 3))
        for i, eta in enumerate(Eta):
            # Vdot = c'*u, where u is a unit vector
            # solve for the optimal unit vector direction
            c = self.a_t * np.dot(eta, np.dot(self.W, G[i]))
            alpha1 = atan2(c[0, 1], c[0, 0])
            ca1 = cos(alpha1)
            sa1 = sin(alpha1)
            alpha2 = atan2((c[0, 0]*ca1 + c[0, 1]*sa1), c[0, 2])
            ca2 = cos(alpha2)
            sa2 = sin(alpha2)
            u = -np.array([[ca1*sa2], [sa1*sa2], [ca2]])

            self.u[i] = u.T
            U[i, :] = np.dot(self.a_t, np.dot(G[i], u).T)

        self.Xdot = U
        return U

        def __repr__(self):
            """Printable represenation of the object."""
            return 'LyapunovElementSteeringOptimal({}, {}, {}, {})'.format(
                self.mu, self.W, self.a_t, self.Xref)

        def __str__(self):
            """Human readable represenation of the object."""
            output = 'LyapunovElementSteeringOptimal'
            output += '(mu={}, W={}, a_t={}, Xref={})'.format(
                self.mu, self.W, self.a_t, self.Xref)
            return output
