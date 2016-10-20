"""Created on Sat Oct 01 2015 11:34.

@author: Nathan Budd
"""
import numpy as np
import numpy.linalg as npl
import scipy.linalg as spl


def mee2rv(MEE, mu=1.):
    """
    Convert modified equinoctial elements to inertial position and velocity.

    Parameters
    ----------
    MEE : ndarray
        mx6 array of elements ordered as [p f g h k L].
    mu : float
        Standard gravitational parameter. Defaults to canonical units.

    Returns
    -------
    RV : ndarray
        mx6 array of elements ordered as [r_x r_y r_z v_x v_y v_z].
    """

    p = MEE[0:, 0:1]
    f = MEE[0:, 1:2]
    g = MEE[0:, 2:3]
    h = MEE[0:, 3:4]
    k = MEE[0:, 4:5]
    L = MEE[0:, 5:6]

    cL = np.cos(L)
    sL = np.sin(L)

    r = p / (1. + f*cL + g*sL)

    # r in equinoctial frame
    zero = np.zeros(p.shape)
    r_equ = r * np.concatenate((cL, sL, zero), 1)

    # v in equinoctial frame
    r_dot = (mu/p)**(.5) * (f*sL - g*cL)
    rL_dot = (mu/p)**(.5) * (1. + f*cL + g*sL)
    v_equ = np.concatenate((r_dot*cL - rL_dot*sL,
                            r_dot*sL + rL_dot*cL,
                            zero), 1)

    RV_equ = np.concatenate((r_equ, v_equ), 1)
    RV = np.zeros(RV_equ.shape)

    for i, rv_equ in enumerate(RV_equ):
        # rotation matrix from equinoctial to earth-centered inertial frame
        h1 = h[i, 0]
        k1 = k[i, 0]
        h2 = h1**2
        k2 = k1**2
        den = 1. / (1. + h2 + k2)
        eci_C_equ = den * np.array([[1.+h2-k2, 2.*h1*k1, 2.*k1],
                                    [2.*h1*k1, 1.-h2+k2, -2.*h1],
                                    [-2.*k1, 2.*h1, 1.-h2-k2]])
        C_rv = spl.block_diag(eci_C_equ, eci_C_equ)

        # rotate r,v into ECI frame
        RV[i:i+1] = (C_rv @ rv_equ.reshape((6, 1))).T

    return RV
