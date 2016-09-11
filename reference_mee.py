"""Created on Sun Sep 11 2015 17:31.

@author: Nathan Budd
"""
import numpy as np
import numpy.linalg as npl
import numpy.matlib as npm
from .reference_abstract import ReferenceAbstract
from .model_mee import ModelMEE


class ReferenceMEE(ReferenceAbstract):
    """Class for generating reference trajectories with MEEs."""

    def __init__(self, X0, model):
        """."""
        super().__init__(X0, model)
