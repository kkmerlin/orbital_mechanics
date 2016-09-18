"""Created on Sun Sep 11 2015 17:31.

@author: Nathan Budd
"""
import numpy as np
import numpy.linalg as npl
import numpy.matlib as npm
from .reference_abstract import ReferenceAbstract
from .model_coe import ModelCOE


class ReferenceCOE(ReferenceAbstract):
    """Class for generating reference trajectories with COEs."""

    def __init__(self, X0, mu):
        """."""
        super().__init__(X0, ModelCOE(mu))

    def __repr__(self):
        """Printable represenation of the object."""
        return super().repr('ReferenceCOE')

    def __str__(self):
        """Human readable represenation of the object."""
        return super().str('ReferenceCOE')
