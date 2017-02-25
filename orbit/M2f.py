"""Created on Thu Sep 22 2015 14:54.

@author: Nathan Budd
"""
import numpy as np
from .M2E import M2E
from .E2f import E2f


def M2f(coe_M):
    """Convert mean anomaly, M,  to true anomaly, f.

    Input
    -----
    coe_M : ndarray
    mx6 array of classical orbital elements [a e i W w M]. m is the number
    of samples and 6 is the dimension of the element set.

    Output
    ------
    coe_f : ndarray
    mx6 array of classical orbital elements [a e i W w f]. m is the number
    of samples and 6 is the dimension of the element set.
    """
    coe_E = M2E(coe_M)
    coe_f = E2f(coe_E)
    return coe_f
