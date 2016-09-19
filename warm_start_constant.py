"""Created on Fri Sep 09 2015 16:25.

@author: Nathan Budd
"""
import numpy as np
import numpy.linalg as npl
from .model_abstract import ModelAbstract


class WarmStartConstant(ModelAbstract):
    """Warm start that returns the initial condition at all times t."""
    def __init__(self):
        """."""
        super().__init__()

    def __call__(self, T, X):
        """Constant output warm start.

        See dynamics_abstract.py for more details.
        """
        X0 = X[0]
        self.Xdot = np.ones((len(T), 1)) * X0

        return self.Xdot

    def __repr__(self):
        """Printable represenation of the object."""
        return 'WarmStartConstant()'

    def __str__(self):
        """Human readable represenation of the object."""
        return 'WarmStartConstant()'
