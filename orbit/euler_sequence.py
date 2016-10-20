"""Created on Sat Oct 01 2016 21:57.

@author: Nathan Budd
"""
import numpy as np
from math import cos
from math import sin


def euler_sequence(axes, *args):
    """
    Create direction cosine matrices from any number of principal roations.

    Generates a DCM by left-multiplying an arbitrary number of principal axis
    DCMs. The first listed axis and first given array of angles is the first
    DCM, and so is at the right-most of the multiplication.

    e.g. given axes = [1, 2] and args = [a, b], the final DCM will be the
    result of a 1-axis rotation through a followed by a 2-axis rotation through
    b: C = C_2(b) @ C_1(a)

    Parameters
    ----------
    axes : list of ints
        Ordered list of the axis of each principal rotation.
        Allowable values: (1, 2, 3)
    args : list of ndarrays
        Each ndarray is a 2D column of angles.

    Returns
    -------
    DCM : ndarray
        A 3D ndarray. Each entry along the first index is a 3x3 DCM
        corresponding to the input angles.
    """

    def C(i, x):
        """
        Create an i-axis DCM through x.

        Parameters
        ----------
        i : int
            Axis number.
        x : float
            Angle of rotation.

        Returns
        -------
        DCM : ndarray
            A 3x3 direction principal axis DCM.
        """
        if i == 1:
            return np.array([[1., 0., 0.],
                             [0., cos(x), sin(x)],
                             [0., -sin(x), cos(x)]])
        elif i == 2:
            return np.array([[cos(x), 0., -sin(x)],
                             [0., 1., 0.],
                             [sin(x), 0., cos(x)]])
        elif i == 3:
            return np.array([[cos(x), sin(x), 0.],
                             [-sin(x), cos(x), 0.],
                             [0., 0., 1.]])

    m, n = args[0].shape
    DCM = np.tile(np.eye(3), (m, 1, 1))

    for i in range(m):
        for j, axis in enumerate(axes):
            DCM[i] = C(axis, args[j][i, 0]) @ DCM[i]

    return DCM
