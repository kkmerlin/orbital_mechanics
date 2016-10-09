"""Created on Fri Sep 09 2015 16:25.

@author: Nathan Budd
"""
import numpy as np
from .model_abstract import ModelAbstract
from .gauss_variational_eqns import GaussVariationalEqns


class ThrustConstant(ModelAbstract):
    """Output the result of a constant acceleration vector defined in LVLH.

    Instance Members
    ----------------
    vector : numpy.array
    3x1 column vector representing the LVLH-constant acceleration applied.

    gve : reference to GaussVariationalEqns.*element(X)
    Takes the constant LVLH acceleration vector into state space time
    derivatives.
    """
    def __init__(self, mu, vector, elements):
        """.
        Parameters
        ----------
        mu : float
            Standard gravitational parameter.
        elements : string
            Indicates the set of element time derivatives that will be output.
            Allowable values: coe, mee, rv
        """
        gve = dict(coe=GaussVariationalEqns(mu).coe,
                   mee=GaussVariationalEqns(mu).mee,
                   rv=GaussVariationalEqns(mu).rv)

        self.vector = vector
        self.gve = gve[elements]
        super().__init__()

    def __call__(self, T, X):
        """Output the result of an LVLH-constant thrust vector.

        See dynamics_abstract.py for more details.
        """
        Gs = self.gve(X)

        Xdot = np.zeros(X.shape)
        for i, G in enumerate(Gs):
            Xdot[i, :] = np.dot(G, self.vector).T

        self.Xdot = Xdot
        return Xdot

    def __repr__(self):
        """Printable represenation of the object."""
        return 'ThrustConstant({}, {})'.format(
            self.vector, self.gve)

    def __str__(self):
        """Human readable represenation of the object."""
        output = 'ThrustConstant'
        output += '(vector={}, gve={}, perturbations={})'.format(
            self.vector, self.gve)
        return output
