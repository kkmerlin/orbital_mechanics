"""Created on Sun 23 Oct 2015 14:02.

@author: Nathan Budd
"""
import numpy as np
import numpy.linalg as npl
from .. import orbit as orb


class TwoBody():
    """Two-body dynamics and reference trajectories.

    The initial state array is an optional argument. If it is provided, calls
    to TwoBody output a reference trajectory. If it is not provided, calls to
    TwoBody output two body dynamics.

    Members
    -------
    mu : float
        Standard gravitational parameter
    element_set : string
        Indicates the element set to be used.
        Allowed values: coe, coea mee, rv
    X0 : ndarray
        A 1x6 array of initial states.
        X0_coe = [p e i W w f]
        X0_coea = [a e i W w f]
        X0_mee = [p f g h k L]
        X0_rv = [rx ry rz vx vy vz]
    Y : ndarray
        The most recent call output.
    """

    def __init__(self, mu, element_set, X0=None):
        """."""
        self.mu = mu
        self.element_set = element_set
        self.X0 = X0
        self.Y = np.array([[]])

    def __call__(self, T, X=None):
        """Evaluate the dynamics or reference trajectory at the sample times.

        Parameters
        ----------
        T : ndarray
            An mx1 column array of sample times.
        X : ndarray
            An mxn array of states.
            Not necessary when calling for a reference trajectory.

        Outputs
        -------
        Y : ndarray
            An mxn array of state derivatives or reference states
        """
        dyn_funcs = dict(coe=self._coe_dynamics,
                         coea=self._coea_dynamics,
                         mee=self._mee_dynamics,
                         rv=self._rv_dynamics)

        ref_funcs = dict(coe=self._coe_reference,
                         coea=self._coea_reference,
                         mee=self._mee_reference,
                         rv=self._rv_reference)

        if self.X0 is None:
            return dyn_funcs[self.element_set](T, X)
        else:
            return ref_funcs[self.element_set](T)

    def _coe_dynamics(self, T, X):
        """COE dynamics function.

        Parameters
        ----------
        T : ndarray
            An mx1 column array of sample times.
        X : ndarray
            An mxn array of states.

        Outputs
        -------
        Y : ndarray
            An mxn array of state derivatives
        """
        p = X[:, 0]
        e = X[:, 1]
        f = X[:, 5]
        r = p / (1. + np.multiply(e, np.cos(f)))
        h = np.power(self.mu * p, .5)
        f_dot = h / np.power(r, 2)

        shape = X.shape
        self.Y = np.zeros(shape)
        self.Y[:, -1] = f_dot

        return self.Y

    def _coea_dynamics(self, T, X):
        """COEA dynamics function.

        Parameters
        ----------
        T : ndarray
            An mx1 column array of sample times.
        X : ndarray
            An mxn array of states.

        Outputs
        -------
        Y : ndarray
            An mxn array of state derivatives
        """
        a = X[:, 0]
        e = X[:, 1]
        f = X[:, 5]
        p = a * (1-e**2)
        r = p / (1. + np.multiply(e, np.cos(f)))
        h = np.power(self.mu * p, .5)
        f_dot = h / np.power(r, 2)

        shape = X.shape
        self.Y = np.zeros(shape)
        self.Y[:, -1] = f_dot

        return self.Y

    def _mee_dynamics(self, T, X):
        """MEE dynamics function.

        Parameters
        ----------
        T : ndarray
            An mx1 column array of sample times.
        X : ndarray
            An mxn array of states.

        Outputs
        -------
        Y : ndarray
            An mxn array of state derivatives
        """
        print('_mee_dynamics IS INCOMPLETE!!')

        return None

    def _rv_dynamics(self, T, X):
        """RV dynamics function.

        Parameters
        ----------
        T : ndarray
            An mx1 column array of sample times.
        X : ndarray
            An mxn array of states.

        Outputs
        -------
        Y : ndarray
            An mxn array of state derivatives
        """
        # take the 2 norm at each instance in time (across the rows)
        R = X[:, 0:3]
        V = X[:, 3:6]

        Rnorm = npl.norm(R, axis=1, keepdims=True)
        neg_mu_by_r3 = -self.mu / np.power(Rnorm, 3)
        Neg_mu_by_r3 = (neg_mu_by_r3 * np.ones((1, 1)))
        Vdot = np.multiply(Neg_mu_by_r3, R)
        self.Y = np.concatenate((V, Vdot), 1)

        return self.Y

    def _coe_reference(self, T):
        """COE reference function.

        Parameters
        ----------
        T : ndarray
            An mx1 column array of sample times.

        Outputs
        -------
        Y : ndarray
            An mxn array of reference states.
        """
        T0 = T[0, 0]
        p = self.X0[0, 0]
        e = self.X0[0, 1]
        a = p / (1. - e**2)
        Mdot = (self.mu / a**3)**.5

        dT = T - T0
        dM = dT * Mdot
        Y_ref_M = np.tile(self.X0, T.shape)
        Y_ref_M[0:, -1:] = Y_ref_M[0:, -1:] + dM

        self.Y = orb.M2f(Y_ref_M)
        return self.Y

    def _coea_reference(self, T):
        """COEA reference function.

        Parameters
        ----------
        T : ndarray
            An mx1 column array of sample times.

        Outputs
        -------
        Y : ndarray
            An mxn array of reference states.
        """
        T0 = T[0, 0]
        a = self.X0[0, 0]
        e = self.X0[0, 1]
        Mdot = (self.mu / a**3)**.5

        dT = T - T0
        dM = dT * Mdot
        Y_ref_M = np.tile(self.X0, T.shape)
        Y_ref_M[0:, -1:] = Y_ref_M[0:, -1:] + dM

        self.Y = orb.M2f(Y_ref_M)
        return self.Y

    def _mee_reference(self, T):
        """MEE reference function.

        Parameters
        ----------
        T : ndarray
            An mx1 column array of sample times.

        Outputs
        -------
        Y : ndarray
            An mxn array of reference states.
        """
        Y_COE = self._coe_reference(T)
        self.Y = orb.coe2mee(Y_COE)

        return self.Y

    def _rv_reference(self, T):
        """RV reference function.

        Parameters
        ----------
        T : ndarray
            An mx1 column array of sample times.

        Outputs
        -------
        Y : ndarray
            An mxn array of reference states.
        """
        Y_COE = self._coe_reference(T)
        self.Y = orb.coe2rv(Y_COE)

        return self.Y

    def __repr__(self):
        """Printable represenation of the object."""
        return 'ModelCOE({}, {}, {})'.format(self.mu, self.element_set,
                                             self.X0)

    def __str__(self):
        """Human readable represenation of the object."""
        return ('ModelCOE(mu={}, element_set={}, X0={})'
                .format(self.mu, self.element_set, self.X0))
