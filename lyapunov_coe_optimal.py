"""Created on Wed Sep 08 2015 15:10.

@author: Nathan Budd
"""
import numpy as np
import numpy.linalg as npl
import numpy.matlib as npm
from math import sin, cos, atan2
from .dynamics_abstract import DynamicsAbstract
from .gauss_lagrange_planetary_eqns import GaussLagrangePlanetaryEqns as GLPE


class LyapunovMEEOptimal(DynamicsAbstract):
    """Lyapunov control for MEE, neglecting phase angle.

    Static Members
    -------
    _parameter_list : list
        a_t - thrust magnitude
        mu - standard gravitational parameter
        xref - reference MEEs, numpy.matrix
    """

    _class_string = 'LyapunovMEEOptimal'

    _parameter_list = ['a_t', 'mu', 'xref']

    def __init__(self, arg):
        """."""
        super().__init__(arg)

    def __call__(self, T, X):
        """Evaluate the control at the given times.

        X = [p e i W w nu]

        See dynamics_abstract.py for more details.
        """
        W = np.matrix(np.diag([1.]*5 + [0.]))
        M = GLPE(self.mu).coe(X)
        Xref = npm.ones((len(T), 1)) * self.xref
        Eta = X - Xref

        U = npm.zeros((len(T), 6))
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

        return U
