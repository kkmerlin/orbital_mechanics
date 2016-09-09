"""Created on Wed Sep 07 2015 16:36.

@author: Nathan Budd
"""
import numpy as np
import numpy.linalg as npl
import numpy.matlib as npm
from .dynamics_abstract import DynamicsAbstract


class DynamicsCOE(DynamicsAbstract):
    """COE dynamics.

    Static Members
    -------
    _parameter_list : list
        mu - standard gravitational parameter
        a_d - DynamicsAbstract subclass
    """

    _class_string = 'DynamicsCOE'

    _parameter_list = ['mu', 'a_d']

    def __init__(self, arg):
        """."""
        super().__init__(arg)

    def __call__(self, T, X):
        """Evaluate the dynamics at the given times.

        X = [p e i W w nu]

        See dynamics_abstract.py for more details.
        """
        p = X[:, 0]
        e = X[:, 1]
        nu = X[:, 5]
        r = p / (1. + np.multiply(e, np.cos(nu)))
        h = np.power(self.mu * p, .5)
        nu_dot = h / np.power(r, 2)

        shape = X.shape
        zeros = npm.zeros((shape[0], shape[1]-1))
        two_body = np.concatenate((zeros, nu_dot), 1)

        perturbations = self.a_d(T, X)
        Xdot = two_body + perturbations
        return Xdot
