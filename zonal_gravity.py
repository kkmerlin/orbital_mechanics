"""Created on Sat Oct 01 2015 11:34.

@author: Nathan Budd
"""
import numpy as np
import numpy.linalg as npl
from .model_abstract import ModelAbstract


class ZonalGravity(ModelAbstract):
    """
    Zonal gravity perturbations, J2 up to J6, in canonical units.

    Members
    -------
    J_list : list of floats
        List of all zonal gravity coefficients from J2 up to no more than J6.
    R : float
        Radius of the earth
    """

    def __init__(self):
        """."""
        super().__init__()

    def __call__(self, T, X):
        """Output an appropriately sized array of zeros.

        See dynamics_abstract.py for more details.
        """
        self.Xdot = np.zeros(X.shape)
        return self.Xdot

        def __repr__(self):
            """Printable represenation of the object."""
            return 'PerturbZero()'

        def __str__(self):
            """Human readable represenation of the object."""
            return 'PerturbZero()'
