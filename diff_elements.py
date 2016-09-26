"""Created on Thu Sep 22 2015 15:05.

@author: Nathan Budd
"""
import numpy as np


def diff_elements(Xa, Xb, angle_idx=[5]):
    """Calculate element differences Xa-Xb, and acute differences for angles.

    For two sets of element trajectories, the algebraic differences are
    calculated for non-angle elements. For angle elements, the acute angle
    between them is given.

    Input
    -----
    Xa : ndarray
    mxn array of orbital elements, where m is the number of samples and n is
    the state dimension.

    Xb : ndarray
    mxn array of orbital elements, where m is the number of samples and n is
    the state dimension.

    angle_idx : list or tuple
    Indices of angle elements in the state vector. By default this is taken to
    be the last element (5). For COEs [p, e, i, W, w, f] use [2, 3, 4 ,5].

    Output
    ------
    dX : ndarray
    mxn array of orbital element differences, where m is the number of samples
    and n is the state dimension.
    """
    mn = Xa.shape
    dX = np.zeros(mn)

    for j in range(mn[1]):
        if j in angle_idx:
            dX[0:, j] = np.arccos(np.cos(Xa[0:, j] - Xb[0:, j]))
        else:
            dX[0:, j] = Xa[0:, j] - Xb[0:, j]

    return dX
