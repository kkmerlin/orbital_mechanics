"""Created on Sat Oct 01 2015 16:14.

@author: Nathan Budd
"""
import numpy as np


def mee2coe(MEE, mu=1.):
    """
    Convert modified equinoctial elements to classical orbital elements.

    Parameters
    ----------
    MEE : ndarray
        mx6 array of elements ordered as [p f g h k L].
    mu : float
        Standard gravitational parameter. Defaults to canonical units.

    Returns
    -------
    COE : ndarray
        mx6 array of elements ordered as [p e i W w f].
    """

    p = MEE[0:, 0:1]
    f = MEE[0:, 1:2]
    g = MEE[0:, 2:3]
    h = MEE[0:, 3:4]
    k = MEE[0:, 4:5]
    L = MEE[0:, 5:6]

    # inclination
    i = 2. * np.arctan((h**2 + k**2)**.5)

    # right ascension of the ascending node
    W = np.mod(np.arctan2(k, h), 2*np.pi)

    # eccentricity
    e = (f**2 + g**2)**.5

    # argument of periapsis
    w_bar = np.mod(np.arctan2(g, f), 2*np.pi)

    w = np.mod(w_bar - W, 2*np.pi)

    # true anomaly
    f = np.mod(L - w_bar, 2*np.pi)

    return np.concatenate((p, e, i, W, w, f), 1)
