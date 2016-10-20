"""Created on Sat Oct 01 2015 16:44.

@author: Nathan Budd
"""
import numpy as np
from .rv2mee import rv2mee
from .mee2coe import mee2coe


def rv2coe(RV, mu=1.):
    """
    Convert inertial position and velocity to classical orbital elements.

    Parameters
    ----------
    RV : ndarray
        mx6 array of elements ordered as [r_x r_y r_z v_x v_y v_z].
    mu : float
        Standard gravitational parameter. Defaults to canonical units.

    Returns
    -------
    COE : ndarray
        mx6 array of elements ordered as [p e i W w f].
    """

    return mee2coe(rv2mee(RV))
