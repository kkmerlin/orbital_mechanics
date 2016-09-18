"""Created on Wed Sep 07 2015 17:51.

@author: Nathan Budd
"""
import numpy as np
import numpy.linalg as npl
import numpy.matlib as npm
from .model_abstract import ModelAbstract


class PerturbZero(ModelAbstract):
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

        def __repr__(self):
            """Printable represenation of the object."""
            return 'PerturbZero()'

        def __str__(self):
            """Human readable represenation of the object."""
            return 'PerturbZero()'
