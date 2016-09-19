"""Created on Sun Sep 11 2015 17:31.

@author: Nathan Budd
"""
from abc import ABCMeta, abstractmethod
import numpy as np


class ReferenceAbstract(metaclass=ABCMeta):
    """Abstract class for generating reference trajectories with COEs or MEEs.

    Instance Members
    ----------------
    X0 : np.array
    A 1xn array of initial states.

    model : ModelCOE or ModelMEE object
    Holds the dynamics that will be used to generate the reference trajectory's
    constant derivative.
    """

    def __init__(self, X0, model):
        """."""
        self.X0 = X0
        self.model = model

    def __call__(self, T):
        """Generate reference trajectory.

        Input
        -----
        T : np.array
        An mx1 column array of times.

        Output
        ------
        X : np.array
        An mxn array of reference states.
        """
        T0 = T[0]
        Xdot = self.model.__call__(T0, self.X0)

        dT = T - T0
        dX = np.dot(dT, Xdot)
        X = np.ones(T0.shape) * self.X0 + dX

        return X

        def repr(self, class_name):
            """Printable represenation of the object."""
            return class_name+'({}, {})'.format(self.X0, self.model)

        def str(self):
            """Human readable represenation of the object."""
            output = class_name
            output += '(X0={}, model={})'.format(self.X0, self.model)
            return output
