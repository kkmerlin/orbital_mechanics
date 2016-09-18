"""Created on Fri Sep 09 2015 16:25.

@author: Nathan Budd
"""
import numpy as np
import numpy.linalg as npl
import numpy.matlib as npm
from .model_abstract import ModelAbstract


class ThrustConstant(ModelAbstract):
    """Output the result of a constant acceleration vector defined in LVLH.

    Instance Members
    ----------------
    vector : numpy.matrix
    3x1 column vector representing the LVLH-constant acceleration applied.

    glpe : reference to GaussLagrangePlanetaryEqns.*element(X)
    Takes the constant LVLH acceleration vector into state space time
    derivatives.
    """
    def __init__(self, vector, glpe):
        """."""
        self.vector = vector
        self.glpe = glpe
        super().__init__()

    def __call__(self, T, X):
        """Output the result of an LVLH-constant thrust vector.

        See dynamics_abstract.py for more details.
        """
        Gs = self.glpe(X)

        Xdot = npm.zeros(X.shape)
        for i, G in enumerate(Gs):
            Xdot[i, :] = (G * self.vector).T

        self.Xdot = Xdot
        return Xdot

    def __repr__(self):
        """Printable represenation of the object."""
        return 'ThrustConstant({}, {})'.format(
            self.vector, self.glpe)

    def __str__(self):
        """Human readable represenation of the object."""
        output = 'ThrustConstant'
        output += '(vector={}, glpe={}, perturbations={})'.format(
            self.vector, self.glpe)
        return output
