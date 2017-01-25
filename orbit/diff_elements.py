"""Created on Thu Sep 22 2015 15:05.

@author: Nathan Budd
"""
import numpy as np


def diff_elements(X, X_r, angle_idx=[5]):
    """Calculate element differences X-X_r, and acute differences for angles.

    For two sets of element trajectories, the algebraic differences are
    calculated for non-angle elements. For angle elements, an angular
    difference on [-pi, pi] is given.

    Input
    -----
    X : ndarray
    mxn array of true orbital elements, where m is the number of samples and n
    is the state dimension.

    X_r : ndarray
    mxn array of reference orbital elements, where m is the number of samples
    and n is the state dimension.

    angle_idx : list or tuple
    Indices of angle elements in the state vector. By default this is taken to
    be the last element (5). For COEs [p, e, i, W, w, f] use [2, 3, 4 ,5].

    Output
    ------
    dX : ndarray
    mxn array of orbital element differences, where m is the number of samples
    and n is the state dimension.
    """
    mn = X.shape

    dX = X - X_r

    for j in angle_idx:
        dX[0:, j] = np.fmod(dX[0:, j], 2*np.pi)
        dX[0:, j] = np.array(
            [a-2*np.pi if a > np.pi else a for a in dX[0:, j]])
        dX[0:, j] = np.array(
            [a+2*np.pi if a < -np.pi else a for a in dX[0:, j]])

    return dX
