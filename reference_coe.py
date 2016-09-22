"""Created on Sun Sep 11 2015 17:31.

@author: Nathan Budd
"""
from .reference_abstract import ReferenceAbstract
from orbit.orbit import Orbit


class ReferenceCOE(ReferenceAbstract):
    """Class for generating reference trajectories with COEs."""

    def __init__(self, X0, mu):
        """."""
        super().__init__(X0, mu)

    def __call__(self, T):
        """Generate reference trajectory in terms of true anomaly.

        Input
        -----
        T : np.array
        An mx1 column array of times.

        Output
        ------
        X : ndarray
        An mxn array of reference states.
        """
        X_ref_M = super().__call__(T)
        X = Orbit(self.mu).M2f_ellipse(X_ref_M)
        return X


    def __repr__(self):
        """Printable represenation of the object."""
        return super().repr('ReferenceCOE')

    def __str__(self):
        """Human readable represenation of the object."""
        return super().str('ReferenceCOE')
