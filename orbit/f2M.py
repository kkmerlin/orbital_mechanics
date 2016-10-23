"""Created on Sun Oct 23 2015 16:51.

@author: Nathan Budd
"""
import numpy as np
from .M2E import M2E
from .E2f import E2f


def f2M(coe_f):
    """Convert true anomaly, f, to mean anomaly, M.

    Input
    -----
    coe_f : ndarray
    mx6 array of classical orbital elements [p e i W w f]. m is the number
    of samples and 6 is the dimension of the element set.

    Output
    ------
    coe_M : ndarray
    mx6 array of classical orbital elements [p e i W w M]. m is the number
    of samples and 6 is the dimension of the element set.
    """
    coe_E = f2E(coe_f)
    return E2m(coe_E)
