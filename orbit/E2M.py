"""Created on Sun Oct 23 2015 16:23.

@author: Nathan Budd
"""
import numpy as np


def E2M(coe_E):
    """Convert eccentric anomaly, E, to elliptic mean anomaly, M.

    Input
    -----
    coe_E : ndarray
    mx6 array of classical orbital elements [p e i W w E]. m is the number
    of samples and 6 is the dimension of the element set.

    Output
    ------
    coe_M : ndarray
    mx6 array of classical orbital elements [p e i W w M]. m is the number
    of samples and 6 is the dimension of the element set.
    """
    e = coe_E[0:, 1:2]
    E = coe_E[0:, -1:]

    M = E - e*np.sin(E)

    return np.concatenate((coe_E[0:, 0:-1], M), 1)
