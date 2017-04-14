"""Created by Nathan I. Budd on 26 Feb 2017 @ 14:18."""
import numpy as np
from .f2M import f2M


def coe2coeM0(COE, T, mu=1.):
    """
    Convert classical orbital elements and true anomaly to epoch mean anomaly.

    Parameters
    ----------
    COE : ndarray
        mx6 array of elements ordered as [a e i W w f].
    T : ndarray
        mx1 array of times.
    mu : float
        Standard gravitational parameter. Defaults to canonical units.

    Returns
    -------
    COEM0 : ndarray
        mx6 array of elements ordered as [a e i W w M0].
    """

    COEM = f2M(COE)
    a = COEM[:, 0:1]
    M = COEM[:, 5:6]

    n = (mu / a**3)**0.5
    M0 = np.mod(M - n*T, 2*np.pi)

    return np.concatenate((COE[:, 0:-1], M0), axis=1)
