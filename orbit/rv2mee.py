"""Created on Sat Oct 01 2015 12:12.

@author: Nathan Budd
"""
import numpy as np
import numpy.linalg as npl


def rv2mee(RV, mu=1.):
    """
    Convert inertial position and velocity to modified equinoctial elements.

    Parameters
    ----------
    RV : ndarray
        mx6 array of elements ordered as [r_x r_y r_z v_x v_y v_z].
    mu : float
        Standard gravitational parameter. Defaults to canonical units.

    Returns
    -------
    MEE : ndarray
        mx6 array of elements ordered as [p f g h k L].
    """
    tol = 1e-14

    R = RV[0:, 0:3]
    V = RV[0:, 3:6]
    m, n = RV.shape

    r = npl.norm(R, ord=2, axis=1).reshape((m, 1))
    v = npl.norm(V, ord=2, axis=1).reshape((m, 1))

    # angular momentum
    H = np.cross(R, V)
    H_norm = npl.norm(H, ord=2, axis=1).reshape((m, 1))
    H_hat = H / H_norm

    energy = (v**2)/2 - mu/r

    # semilatus rectum
    p = np.zeros((m, 1))
    for i, en in enumerate(energy):
        is_parabola = -tol <= en <= tol
        p[i, 0] = (-H_norm[i, 0]**2 / mu if is_parabola
                   else H_norm[i, 0]**2 / mu)

    # equinocital x,y components of ascending node vector
    h = -H_hat[0:, 1:2] / (1. + H_hat[0:, 2:3])
    k = H_hat[0:, 0:1] / (1. + H_hat[0:, 2:3])

    # equinoctial x,y directions in equinoctial frame
    f_equ = np.array([[1., 0., 0.]]).T
    g_equ = np.array([[0., 1., 0.]]).T

    # equinoctial x,y directions in ECI frame
    f_eci = np.zeros((m, 3))
    g_eci = np.zeros((m, 3))

    for i in range(h.shape[0]):
        # rotation matrix from equinoctial to earth-centered inertial frame
        h1 = h[i, 0]
        k1 = k[i, 0]
        h2 = h1**2
        k2 = k1**2
        den = 1. / (1. + h2 + k2)
        eci_C_equ = den * np.array([[1.+h2-k2, 2.*h1*k1, 2.*k1],
                                    [2.*h1*k1, 1.-h2+k2, -2.*h1],
                                    [-2.*k1, 2.*h1, 1.-h2-k2]])

        f_eci[i:i+1] = (eci_C_equ @ f_equ).T
        g_eci[i:i+1] = (eci_C_equ @ g_equ).T

    # eccentricity vectors
    e = np.cross(V, H)/mu - R/r

    # equinoctial x,y components of eccentricity vector
    f = np.diagonal(e @ f_eci.T).reshape((m, 1))
    g = np.diagonal(e @ g_eci.T).reshape((m, 1))

    # true longitude
    cL = np.diagonal(f_eci @ R.T)
    sL = np.diagonal(g_eci @ R.T)
    L = np.mod(np.arctan2(sL, cL), 2*np.pi).reshape((m, 1))

    return np.concatenate((p, f, g, h, k, L), 1)
