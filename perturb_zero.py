"""Created on Wed Sep 07 2015 17:51.

@author: Nathan Budd
"""
import numpy as np
import numpy.linalg as npl
import numpy.matlib as npm
from .dynamics_abstract import DynamicsAbstract


class PerturbZero(DynamicsAbstract):
    """Perturbations with magnitude zero.
    """

    def __init__(self):
        """."""
        super().__init__()

    def __call__(self, T, X):
        """Output an appropriately sized matrix of zeros.

        See dynamics_abstract.py for more details.
        """
        self.Xdot = npm.zeros(X.shape)
        return self.Xdot
