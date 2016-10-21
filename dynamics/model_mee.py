"""Created on Wed Sep 07 2015 16:36.

@author: Nathan Budd
"""
import numpy as np
import numpy.linalg as npl
from .model_abstract import ModelAbstract


class ModelMEE(ModelAbstract):
    """MEE two-body model.

    Instance Members
    ----------------
    mu : float
    Standard gravitational parameter
    """

    def __init__(self, mu):
        """."""
        print('THIS NEEDS TO BE UPDATED TO USE MEAN ANGULAR RATE PROPERLY')
        print('SEE MODEL_COE.PY AS AN EXAMPLE')
        self.mu = mu
        super().__init__()

    def __call__(self, T, X):
        """Evaluate the dynamics at the given times.

        X = [p f g h k L]

        See dynamics_abstract.py for more details.
        """
        # Ldot = sqrt(mu p) / r^2
        # r = p / (1. + f*cL + g*sL)
        cL = np.cos(X[:, -1])
        sL = np.sin(X[:, -1])
        p = X[:, 0]
        f = X[:, 1]
        g = X[:, 2]
        r = p / (1. + np.multiply(f, cL) + np.multiply(g, sL))
        Ldot = (self.mu * p)**.5 / r**2

        shape = X.shape
        self.Xdot = np.zeros(shape)
        self.Xdot[:, -1] = Ldot

        return self.Xdot

        def __repr__(self):
            """Printable represenation of the object."""
            return 'ModelMEE({})'.format(self.mu)

        def __str__(self):
            """Human readable represenation of the object."""
            return 'ModelMEE(mu={})'.format(self.mu)
