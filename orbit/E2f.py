"""Created on Thu Sep 22 2015 14:54.

@author: Nathan Budd
"""
import numpy as np


def E2f(coe_E):
    """Convert eccentric anomaly, E, to true anomaly, f.

    Input
    -----
    coe_E : ndarray
    mx6 array of classical orbital elements [a e i W w E]. m is the number
    of samples and 6 is the dimension of the element set.

    Output
    ------
    coe_f : ndarray
    mx6 array of classical orbital elements [a e i W w f]. m is the number
    of samples and 6 is the dimension of the element set.
    """
    e = coe_E[0:, 1:2]
    E = coe_E[0:, -1:]
    tan_f_by_2 = ((1.+e)/(1.-e))**.5 * np.tan(E/2)
    f = np.mod(2 * np.arctan(tan_f_by_2), 2*np.pi)
    return np.concatenate((coe_E[0:, 0:-1], f), 1)
