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

    stm : reference to GaussLagrangePlanetaryEqns.*element(X)
    For generating the state transition matrix to take the constant LVLH
    acceleration vector into state space time derivatives.
    """
    def __init__(self, vector, stm):
        """."""
        self.vector = vector
        self.stm = stm
        super().__init__()

    def __call__(self, T, X):
        """Output the result of an LVLH-constant thrust vector.

        See dynamics_abstract.py for more details.
        """
        Gs = self.stm(X)

        Xdot = npm.zeros(X.shape)
        for i, G in enumerate(Gs):
            Xdot[i, :] = (G * self.vector).T

        self.Xdot = Xdot
        return Xdot
