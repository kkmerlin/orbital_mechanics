"""Created on Wed Sep 07 2015 12:18.

@author: Nathan Budd
"""
import numpy as np
import numpy.linalg as npl
import numpy.matlib as npm
from .dynamics_abstract import DynamicsAbstract


class DynamicsRV(DynamicsAbstract):
    """RV dynamics.

    Static Members
    -------
    _parameter_list : list
        mu - standard gravitational parameter
        a_d - DynamicsAbstract subclass
    """

    _class_string = 'DynamicsRV'

    _parameter_list = ['mu', 'a_d']

    def __init__(self, arg):
        """."""
        super().__init__(arg)

    def __call__(self, T, X):
        """Evaluate the dynamics at the given times.

        X = [rx ry rz]

        See dynamics_abstract.py for more details.
        """
        # take the 2 norm at each instance in time (across the rows)
        Xnorm = npl.norm(X, 2, 1, True)
        neg_mu_by_r3 = -self.mu / np.power(Xnorm, 3)  # element-wise division
        Neg_mu_by_r3 = (neg_mu_by_r3 * npm.ones((1, 1)))
        two_body = np.multiply(Neg_mu_by_r3, X)

        perturbations = self.a_d(T, X)
        Xdot = two_body + perturbations
        return Xdot
