"""Created on Wed Sep 07 2015 17:51.

@author: Nathan Budd
"""
import numpy as np
import numpy.linalg as npl
import numpy.matlib as npm
from .dynamics_abstract import DynamicsAbstract


class PerturbZero(DynamicsAbstract):
    """Zero perturbations.

    Static Members
    -------
    _parameter_list : list
        n - dimension of the state vector
    """

    _class_string = 'PerturbZero'

    _parameter_list = ['n']

    def __init__(self, arg):
        """.
        Input
        -----
        arg : list or dict
        Corresponding to the elements in _parameter_list
        """
        super().__init__(arg)

    def __call__(self, T, X):
        """Output an appropriately sized matrix of zeros.

        See dynamics_abstract.py for more details.
        """
        m = len(T)
        a_d = npm.zeros((m, self.n))
        return a_d
