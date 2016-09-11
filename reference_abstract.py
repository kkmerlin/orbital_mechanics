"""Created on Sun Sep 11 2015 17:31.

@author: Nathan Budd
"""
from abc import ABCMeta, abstractmethod
import numpy as np
import numpy.linalg as npl
import numpy.matlib as npm


class ReferenceAbstract(metaclass=ABCMeta):
    """Abstract class for generating reference trajectories with COEs or MEEs.

    Instance Members
    ----------------
    X0 : np.matrix
    A 1xn matrix of initial states.

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
        T : np.matrix
        An mx1 column matrix of times.

        Output
        ------
        X : np.matrix
        An mxn matrix of reference states.
        """
        T0 = T[0, 0]
        Xdot = self.model.__call__(T0, self.X0)

        dT = T - T0
        dX = dT * Xdot
        X = npm.ones(T0.shape) * self.X0 + dX

        return X
