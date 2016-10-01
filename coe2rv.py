"""Created on Sat Oct 01 2015 16:44.

@author: Nathan Budd
"""
import numpy as np
from .coe2mee import coe2mee
from .mee2rv import mee2rv


def coe2rv(COE, mu=1.):
    """
    Convert classical orbital elements to inertial position and velocity.

    Parameters
    ----------
    COE : ndarray
        mx6 array of elements ordered as [p e i W w f].
    mu : float
        Standard gravitational parameter. Defaults to canonical units.

    Returns
    -------
    RV : ndarray
        mx6 array of elements ordered as [r_x r_y r_z v_x v_y v_z].
    """

    return mee2rv(coe2mee(COE))
