"""Created on Wed Sep 08 2015 15:10.

@author: Nathan Budd
"""
import numpy as np
import numpy.linalg as npl
import numpy.matlib as npm
from .dynamics_abstract import DynamicsAbstract
from .gauss_lagrange_planetary_eqns import GaussLagrangePlanetaryEqns as GLPE


class LyapunovMEEPhaseFree(DynamicsAbstract):
    """Lyapunov control for MEE, neglecting phase angle.

    Static Members
    -------
    _parameter_list : list
        mu - standard gravitational parameter
        xref - reference MEEs, without true longitude (L), row numpy.matrix
    """

    _class_string = 'LyapunovMEEPhaseFree'

    _parameter_list = ['mu', 'xref']

    def __init__(self, arg):
        """."""
        super().__init__(arg)

    def __call__(self, T, X):
        """Evaluate the control at the given times.

        X = [p e i W w M]

        See dynamics_abstract.py for more details.
        """
        Xnophase = X[:, :-1]
        Xref = npm.ones((len(T), 1)) * self.xref
        Eta = Xnophase - Xref
        glpe = GLPE(self.mu)
        M = glpe.mee(X)

        U = npm.zeros((len(T), 6))
        for i, eta in enumerate(Eta):
            K_weights = [1, 1, 1, 1, 1]
            K = np.matrix(np.diag(K_weights))
            U[i, :] = (-M[i] * M[i][:-1, :].T * K * eta.T).T

        return U
