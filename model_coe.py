"""Created on Wed Sep 07 2015 16:36.

@author: Nathan Budd
"""
import numpy as np
import numpy.linalg as npl
from .model_abstract import ModelAbstract


class ModelCOE(ModelAbstract):
    """COE two-body model.

    Instance Members
    ----------------
    mu : float
    Standard gravitational parameter
    """

    def __init__(self, mu):
        """."""
        self.mu = mu
        super().__init__()

    def __call__(self, T, X):
        """Evaluate the dynamics at the given times.

        X = [p e i W w f]

        See dynamics_abstract.py for more details.
        """
        p = X[:, 0]
        e = X[:, 1]
        f = X[:, 5]
        r = p / (1. + np.multiply(e, np.cos(f)))
        h = np.power(self.mu * p, .5)
        f_dot = h / np.power(r, 2)

        shape = X.shape
        self.Xdot = np.zeros(shape)
        self.Xdot[:, -1] = f_dot

        return self.Xdot

        def __repr__(self):
            """Printable represenation of the object."""
            return 'ModelCOE({})'.format(self.mu)

        def __str__(self):
            """Human readable represenation of the object."""
            return 'ModelCOE(mu={})'.format(self.mu)
