"""Created by Nathan I. Budd on 26 Feb 2017 @ 14:18."""
import numpy as np
from .M2f import M2f


def coeM02coe(COEM0, T, mu=1.):
    """
    Convert classical orbital elements to inertial position and velocity.

    Parameters
    ----------
    COEM0 : ndarray
        mx6 array of elements ordered as [a e i W w M0].
    T : ndarray
        mx1 array of times.
    mu : float
        Standard gravitational parameter. Defaults to canonical units.

    Returns
    -------
    COE : ndarray
        mx6 array of elements ordered as [a e i W w f].
    """

    a = COEM0[:, 0:1]
    M0 = COEM0[:, 5:6]

    n = (mu / a**3)**0.5
    M = np.mod(M0 + n*T, 2*np.pi)

    return M2f(
        np.concatenate((COEM0[:, 0:-1], M), axis=1)
    )
