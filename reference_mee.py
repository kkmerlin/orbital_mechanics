"""Created on Sun Sep 11 2015 17:31.

@author: Nathan Budd
"""
from .reference_abstract import ReferenceAbstract
from .model_mee import ModelMEE


class ReferenceMEE(ReferenceAbstract):
    """Class for generating reference trajectories with MEEs."""

    def __init__(self, X0, mu):
        """."""
        super().__init__(X0, ModelMEE(mu))

    def __repr__(self):
        """Printable represenation of the object."""
        return super().repr('ReferenceMEE')

    def __str__(self):
        """Human readable represenation of the object."""
        return super().str('ReferenceMEE')
