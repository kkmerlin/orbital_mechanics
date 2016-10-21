"""Created on Thu Sep 29 2015 13:46.

@author: Nathan Budd
"""
import numpy as np
from orbit import diff_elements


def diff_elements_theta_into_p(mu, k, X, X_r, angle_idx=[5]):
    """
    Augment diff_elements.py by incorporating f error into p Errors.

    See diff_elements.py for all other detials.

    Parameters
    ----------
    mu : float
        Standard gravitational pramater.
    k : float
        Gain relating phase error to semi-major axis

    Returns
    -------
    ndarray
        Results from diff_elements with the updated p error
    """
    X_theta = np.concatenate((X, X[0:, 4:5]+X[0:, 5:6]), 1)
    X_r_theta = np.concatenate((X_r, X_r[0:, 4:5]+X_r[0:, 5:6]), 1)
    angle_idx_aug = angle_idx + [6]

    X_d_aug = diff_elements(X_theta, X_r_theta, angle_idx_aug)
    w_d = X_d_aug[0:, 4:5]
    f_d = X_d_aug[0:, 5:6]
    theta_d = np.arccos(np.cos(w_d + f_d))

    p_r = X_r[0:, 0:1]
    e_r = X_r[0:, 1:2]
    a_r = p_r / (1. - e_r**2)

    # NEED SOME KIND OF CHECK FOR NEGATIVE NUMBERS IN ROOT
    a_r_new_cubed = (mu / ((mu/a_r**3)**.5 - k*theta_d))
    a_r_new = np.sign(a_r_new_cubed) * np.fabs(a_r_new_cubed)**(1./3.)
    p_r_new = a_r_new * (1. - e_r**2)

    return np.concatenate((p_r_new, X_d_aug[0:, 1:6]), 1)
