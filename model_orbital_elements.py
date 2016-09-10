"""Created on Fri Sep 10 2016 17:55.

@author: Nathan Budd
"""
from abc import ABCMeta, abstractmethod
import numpy as np
import numpy.matlib as npm
from .model_abstract import ModelAbstract


class ModelOrbitalElements(ModelAbstract):
    """An abstract parent class for orbital element model classes.

    Adds a method to the ModelAbstract framework that allows orbital element
    dynamics classes (e.g. ModelCOE, ModelMEE) to use their two-body dynamics
    to produce reference trajectories.
    """
    def __init__(self):
        """."""
        super().__init__()

    def reference(self, T, X0):
        """Create reference trajectories for orbital element sets.

        Input
        -----
        T : np.matrix
        An mx1 column matrix of times.

        X0 : np.matrix
        A 1xn matrix of states.

        Output
        ------
        X : np.matrix
        An mxn matrix of state derivatives
        """
        T0 = T[0, 0]
        Xdot = self.__call__(T0, X0)

        dT = T - T0
        dX = dT * Xdot
        X = npm.ones(T0.shape) * X0 + dX

        return X
