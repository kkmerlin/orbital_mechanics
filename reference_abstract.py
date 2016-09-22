"""Created on Sun Sep 11 2015 17:31.

@author: Nathan Budd
"""
from abc import ABCMeta, abstractmethod
import numpy as np


class ReferenceAbstract(metaclass=ABCMeta):
    """Abstract class for generating reference trajectories with COEs or MEEs.

    Creates an osculating two-body reference trajectory out of the initial
    state.

    Instance Members
    ----------------
    X0 : np.array
    A 1xn array of initial states.

    mu : float
    Standard gravitational parameter.
    """

    def __init__(self, X0, mu):
        """."""
        self.X0 = X0
        self.mu = mu

    def __call__(self, T):
        """Generate reference trajectory in terms of mean anomaly.

        Input
        -----
        T : np.array
        An mx1 column array of times.

        Output
        ------
        X : ndarray
        An mxn array of reference states.
        """
        T0 = T[0, 0]
        p = segit lf.X0[0, 0]
        e = self.X0[0, 1]
        a = p / (1. - e**2)
        Mdot = (self.mu / a**3)**.5

        dT = T - T0
        dM = dT * Mdot
        X_ref = np.tile(self.X0, T.shape)
        X_ref[0:, -1:] = X_ref[0:, -1:] + dM

        return X_ref

        def __repr__(self, class_name):
            """Printable represenation of the object."""
            return class_name+'({}, {})'.format(self.X0, self.model)

        def __str__(self):
            """Human readable represenation of the object."""
            output = class_name
            output += '(X0={}, model={})'.format(self.X0, self.model)
            return output
