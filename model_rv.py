"""Created on Wed Sep 07 2015 12:18.

@author: Nathan Budd
"""
import numpy as np
import numpy.linalg as npl
import numpy.matlib as npm
from .model_abstract import ModelAbstract


class ModelRV(ModelAbstract):
    """RV two-body model.

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

        X = [rx ry rz vx vy vz]

        See dynamics_abstract.py for more details.
        """
        # take the 2 norm at each instance in time (across the rows)
        R = X[:, 0:3]
        V = X[:, 3:6]
        Rnorm = npl.norm(R, 2, 1, True)
        neg_mu_by_r3 = -self.mu / np.power(Rnorm, 3)  # element-wise division
        Neg_mu_by_r3 = (neg_mu_by_r3 * npm.ones((1, 1)))
        Vdot = np.multiply(Neg_mu_by_r3, R)
        self.Xdot = np.concatenate((V, Vdot), 1)

        return self.Xdot
