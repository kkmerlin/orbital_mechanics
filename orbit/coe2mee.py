"""Created on Sat Oct 01 2015 16:24.

@author: Nathan Budd
"""
import numpy as np


def coe2mee(COE, mu=1.):
    """
    Convert classical orbital elements to modified equinoctial elements.

    Parameters
    ----------
    COE : ndarray
        mx6 array of elements ordered as [p e i W w nu].
    mu : float
        Standard gravitational parameter. Defaults to canonical units.

    Returns
    -------
    MEE : ndarray
        mx6 array of elements ordered as [p f g h k L].
    """

    p = COE[0:, 0:1]
    e = COE[0:, 1:2]
    i = COE[0:, 2:3]
    W = COE[0:, 3:4]
    w = COE[0:, 4:5]
    nu = COE[0:, 5:6]

    # x,y components of eccentricity vector
    f = e * np.cos(w + W)
    g = e * np.sin(w + W)

    # x,y components of ascending node vector
    h = np.tan(i/2.) * np.cos(W)
    k = np.tan(i/2.) * np.sin(W)

    # true longitude
    L = np.mod(W+w+nu, 2*np.pi)

    return np.concatenate((p, f, g, h, k, L), 1)
