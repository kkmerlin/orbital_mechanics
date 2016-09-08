"""Created on Wed Sep 07 2015 17:51.

@author: Nathan Budd
"""
import numpy as np
import numpy.linalg as npl
import numpy.matlib as npm
from .dynamics_abstract import DynamicsAbstract


class PerturbZero(DynamicsAbstract):
    """Zero perturbations.

    Creates a zero disturbance output according to the dimension of X
    """

    _class_string = 'PerturbZero'

    _parameter_list = []

    def __init__(self):
        """."""
        super().__init__()

    def __call__(self, T, X):
        """Output an appropriately sized matrix of zeros.

        See dynamics_abstract.py for more details.
        """
        a_d = npm.zeros(X.shape)
        return a_d
