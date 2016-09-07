"""Created on Wed Sep 07 2015 12:18.

@author: Nathan Budd
"""
import numpy as np
import numpy.linalg as npl
import numpy.matlib as npm
from .dynamics_abstract import DynamicsAbstract


class DynamicsRV2body(DynamicsAbstract):
    """RV-based 2-body dynamics.

    Static Members
    -------
    _parameter_list : list
        mu - standard gravitational parameter
    """

    _class_string = 'RV2body'

    _parameter_list = ['mu']

    def __init__(self, arg):
        """.

        Input
        -----
        arg can be:
            - a 1 element list: [mu]
            - a 1 element dict, with keys: mu

        Instance Members
        ----------------
        _parameters : dict
        Key-value pairs of the standard elements

        More information in dynamics_abstract.py
        """
        size = len(arg)

        try:
            parameters = {'mu': arg[0]}
        except KeyError:
            parameters = arg

        super().__init__(parameters)

    def __call__(self, T, X):
        """Evaluate the dynamics at the given times.

        See dynamics_abstract.py for more details.
        """
        # take the 2 norm at each instance in time (across the rows)
        Xnorm = npl.norm(X, 2, 1, True)
        neg_mu_by_r3 = -self.mu / np.power(Xnorm, 3)  # element-wise division
        Neg_mu_by_r3 = (neg_mu_by_r3 * npm.ones((1, 1)))
        Xdot = np.multiply(Neg_mu_by_r3, X)
        return Xdot
