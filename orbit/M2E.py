"""Created on Thu Sep 22 2015 14:54.

@author: Nathan Budd
"""
import numpy as np


def M2E(coe_M):
    """Convert elliptic mean anomaly, M,  to eccentric anomaly, E.

    Input
    -----
    coe_M : ndarray
    mx6 array of classical orbital elements [a e i W w M]. m is the number
    of samples and 6 is the dimension of the element set.

    Output
    ------
    coe_E : ndarray
    mx6 array of classical orbital elements [a e i W w E]. m is the number
    of samples and 6 is the dimension of the element set.
    """
    tol = 1e-14
    e = coe_M[0:, 1:2]
    M = coe_M[0:, -1:]
    E0 = M + np.sign(np.sin(M))*e
    E1 = E0 + 1.
    max_iterations = 100
    iterations = 0

    while np.max(np.absolute(E1 - E0)) > tol:
        E0 = E1
        E1 = E0 + (M - E0 + e*np.sin(E0)) / (1. - e*np.cos(E0))
        iterations += 1
        if iterations > max_iterations:
            print(
                'max ({}) iterations reached in M2E. Error: {}'.format(
                    max_iterations, np.max(np.absolute(E1 - E0))))
            break

    coe_E = np.concatenate((coe_M[0:, 0:-1], np.mod(E1, 2*np.pi)), 1)
    return coe_E
