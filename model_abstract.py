"""Created on Wed Sep 07 2016 11:41.

@author: Nathan Budd
"""
from abc import ABCMeta, abstractmethod
import numpy as np


class ModelAbstract(metaclass=ABCMeta):
    """An abstract parent class for orbital dynamic model classes.

    Sublcasses allow a variable number of parameter inputs at instantiation to
    accomodate all types of dynamic systems. Each of these parameters can be
    accessed as if they were standalone parameters.

    Subclasses can be called with inputs (t, X) where t is an mx1 np.matrix and
    X is an mxn np.matrix, with m being the number of time steps and n being
    the number of states. The output is the state derivative history.

    Instance Members
    --------------
    Xdot : numpy.matrix
    The most recently computed value
    """
    @abstractmethod
    def __init__(self):
        """."""
        self.Xdot = np.matrix([])

    @abstractmethod
    def __call__(self, T, X):
        """Evaluate the dynamics, disturbance, or control.

        Input
        -----
        T : np.matrix
        An mx1 column matrix of times.

        X : np.matrix
        An mxn matrix of states.

        Output
        ------
        Xdot : np.matrix
        An mxn matrix of state derivatives
        """
        pass
