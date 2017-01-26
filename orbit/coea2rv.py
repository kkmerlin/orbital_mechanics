"""Created on Wed Jan 25 2017 17:58.

@author: Nathan Budd
"""
import numpy as np
from .coe2mee import coe2mee
from .mee2rv import mee2rv


def coea2rv(COEa, mu=1.):
    """
    Convert classical orbital elements to inertial position and velocity.

    Parameters
    ----------
    COEa : ndarray
        mx6 array of elements ordered as [a e i W w f].
    mu : float
        Standard gravitational parameter. Defaults to canonical units.

    Returns
    -------
    RV : ndarray
        mx6 array of elements ordered as [r_x r_y r_z v_x v_y v_z].
    """
    a = COEa[:, 0:1]
    e = COEa[:, 1:2]
    p = a * (1 - e**2)
    COE = np.concatenate((p, COEa[:, 1:]), 1)

    return mee2rv(coe2mee(COE))
