"""Created on Sat Oct 01 2015 16:44.

@author: Nathan Budd
"""
import numpy as np
from .coe2mee import coe2mee
from .mee2rv import mee2rv
from .coeM02coe import coeM02coe


def coeM02rv(COEM0, T, mu=1.):
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
    RV : ndarray
        mx6 array of elements ordered as [r_x r_y r_z v_x v_y v_z].
    """

    return mee2rv(coe2mee(coeM02coe(COEM0, T)))
