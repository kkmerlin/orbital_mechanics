"""Created on Wed Sep 07 2015 16:36.

@author: Nathan Budd
"""
import numpy as np
import numpy.linalg as npl
import numpy.matlib as npm
from .dynamics_abstract import DynamicsAbstract


class DynamicsMEE(DynamicsAbstract):
    """MEE dynamics.

    Static Members
    -------
    _parameter_list : list
        mu - standard gravitational parameter
        a_d - DynamicsAbstract subclass
    """

    _class_string = 'DynamicsMEE'

    _parameter_list = ['mu', 'a_d']

    def __init__(self, arg):
        """."""
        super().__init__(arg)

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
        Ldot = np.power(self.mu * p, .5) / np.power(r, 2)

        shape = X.shape
        zeros = npm.zeros((shape[0], shape[1]-1))
        two_body = np.concatenate((zeros, Ldot), 1)

        perturbations = self.a_d(T, X)
        Xdot = two_body + perturbations
        return Xdot
