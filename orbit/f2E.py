"""Created on Sun Oct 23 2015 16:08.

@author: Nathan Budd
"""
import numpy as np


def f2E(coe_f):
    """Convert true anomaly, f, to eccentric anomaly, E.

    Input
    -----
    coe_f : ndarray
    mx6 array of classical orbital elements [p e i W w f]. m is the number
    of samples and 6 is the dimension of the element set.

    Output
    ------
    coe_E : ndarray
    mx6 array of classical orbital elements [p e i W w E ]. m is the number
    of samples and 6 is the dimension of the element set.
    """
    e = coe_f[0:, 1:2]
    f = coe_f[0:, -1:]
    tan_E_by_2 = ((1.-e)/(1.+e))**.5 * np.tan(f/2)
    E = np.mod(2 * np.arctan(tan_E_by_2), 2*np.pi)
    return np.concatenate((coe_f[0:, 0:-1], E), 1)
