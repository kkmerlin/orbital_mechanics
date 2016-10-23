"""Created on Sat Oct 01 2015 11:34.

@author: Nathan Budd
"""
import numpy as np
import numpy.linalg as npl
from .model_abstract import ModelAbstract
from ..orbit import coe2rv
from ..orbit import mee2rv
from ..orbit import mee2coe
from ..orbit import euler_sequence
from .gauss_variational_eqns import GaussVariationalEqns
from multiplot2d import MultiPlotter


class ZonalGravity(ModelAbstract):
    """
    Zonal gravity perturbations, J2 up to J6, in canonical units.

    Members
    -------
    J : list of floats
        List of all zonal gravity coefficients from J2 up to no more than J6.
        Defaults to only J2.
    Re : float
        Radius of the earth. Defaults to canonical units.
    mu : float
        Standard gravitational parameter. Defaults to canonical units.
    elements : string
        Indicates the element set being used as input and output. Allowable
        values include: 'coe', 'mee'. Defaults to 'coe'.
    """

    toRV = {'coe': coe2rv, 'mee': mee2rv}

    def __init__(self, ord=2, Re=1., mu=1., elements='coe'):
        """."""
        J2_to_6 = [1082.63e-6, -2.52e-6, -1.61e-6, -.15e-6, .57e-6]
        self.J = J2_to_6[0:ord-1]
        self.Re = Re
        self.mu = mu
        self.elements = elements
        super().__init__()

    def __call__(self, T, X):
        """Output indicated element derivatives resulting from zonal gravity.

        See dynamics_abstract.py for more details.
        """
        RV = self.toRV[self.elements](X)

        x = RV[0:, 0:1]
        y = RV[0:, 1:2]
        z = RV[0:, 2:3]
        r = npl.norm(RV[0:, 0:3], ord=2, axis=1).reshape(z.shape)

        # calculate and accumulate J_i ECI accelerations
        a_eci = np.zeros(RV[0:, 0:3].shape)
        for i, J in enumerate(self.J):
            if i == 0:  # J2
                factor = -3./2. * J * (self.mu/r**2) * (self.Re/r)**2
                a_x = (1. - 5.*(z/r)**2) * x/r
                a_y = (1. - 5.*(z/r)**2) * y/r
                a_z = (3. - 5.*(z/r)**2) * z/r

            elif i == 1:  # J3
                factor = 1./2. * J * (self.mu/r**2) * (self.Re/r)**3
                a_x = 5.*(7.*(z/r)**3 - 3.*(z/r)) * x/r
                a_y = 5.*(7.*(z/r)**3 - 3.*(z/r)) * y/r
                a_z = 3.*(1. - 10.*(z/r)**2 + 35./3.*(z/r)**4)

            elif i == 2:  # J4
                factor = 5./8. * J * (self.mu/r**2) * (self.Re/r)**4
                a_x = (3. - 42.*(z/r)**2 + 63.*(z/r)**4) * x/r
                a_y = (3. - 42.*(z/r)**2 + 63.*(z/r)**4) * y/r
                a_z = (15. - 70.*(z/r)**2 + 63.*(z/r)**4) * z/r

            elif i == 3:  # J5
                factor = 1./8. * J * (self.mu/r**2) * (self.Re/r)**5
                a_x = 3.*(35.*(z/r) - 210.*(z/r)**3 + 231.*(z/r)**5) * x/r
                a_y = 3.*(35.*(z/r) - 210.*(z/r)**3 + 231.*(z/r)**5) * y/r
                a_z = (-15. + 315.*(z/r)**2 - 945.*(z/r)**4 + 693.*(z/r)**6)

            elif i == 4:  # J6
                factor = -1./16. * J * (self.mu/r**2) * (self.Re/r)**6
                a_x = (35. - 945.*(z/r)**2 + 3465.*(z/r)**4 -
                       3003.*(z/r)**6) * x/r
                a_y = (35. - 945.*(z/r)**2 + 3465.*(z/r)**4 -
                       3003.*(z/r)**6) * y/r
                a_z = (245. - 2205.*(z/r)**2 + 4851.*(z/r)**4 -
                       3003.*(z/r)**6) * z/r

            a_eci = a_eci + np.concatenate((a_x, a_y, a_z), 1) * factor

        # create rotation matrices and Gauss-Lagrange matrices
        if self.elements is 'coe':
            i = X[0:, 2:3]
            W = X[0:, 3:4]
            w = X[0:, 4:5]
            f = X[0:, 5:6]

            G = GaussVariationalEqns(self.mu).coe(X)

        elif self.elements is 'mee':
            COE = mee2coe(X)
            i = COE[0:, 2:3]
            W = COE[0:, 3:4]
            w = COE[0:, 4:5]
            f = COE[0:, 5:6]

            G = GaussLagrangePlanetaryEqns(self.mu).mee(X)

        C = euler_sequence([3, 1, 3], W, i, w+f)

        # rotate ECI accelerations into the LVLH frame
        a_lvlh = np.zeros(a_eci.shape)
        for j, dcm in enumerate(C):
            a_lvlh[j] = (dcm @ a_eci[j].T).T

        # multiply accelerations into Gauss-Lagrange matrix
        self.Xdot = np.zeros(X.shape)
        for j, g in enumerate(G):
            self.Xdot[j] = (g @ a_lvlh[j].T).T

        return self.Xdot

    def __repr__(self):
        """Printable represenation of the object."""
        return 'ZonalGravity({}, {}, {}, {})'.format(
            self.J, self.Re, self.mu, self.elements)

    def __str__(self):
        """Human readable represenation of the object."""
        return 'ZonalGravity(J={}, Re={}, mu={}, elements={})'.format(
            self.J, self.Re, self.mu, self.elements)

    def hamiltonian(self, T, X):
        """
        Calculate the jacobi integral to check conservation of energy.

        Parameters
        ----------
        T : ndarray
            mx1 array of times
        X : ndarray
            mx6 array of states.

        Returns
        -------
        energy : ndarray
            mx1 array of energies at each sample point.
        """
        RV = self.toRV[self.elements](X)

        z = RV[0:, 2:3]
        r = npl.norm(RV[0:, 0:3], ord=2, axis=1).reshape(z.shape)
        v = npl.norm(RV[0:, 3:6], ord=2, axis=1).reshape(z.shape)
        sin_phi = z/r

        KE = .5 * v**2

        # calculate and accumulate potential function terms for each J_i
        V = -self.mu/r
        for i, J in enumerate(self.J):
            if i == 0:  # J2
                factor = -J/2. * self.mu/r * (self.Re/r)**2
                terms = 3.*sin_phi**2 - 1.

            elif i == 1:  # J3
                factor = -J/2. * self.mu/r * (self.Re/r)**3
                terms = 5.*sin_phi**3 - 3.*sin_phi

            elif i == 2:  # J4
                factor = -J/8. * self.mu/r * (self.Re/r)**4
                terms = 35.*sin_phi**4 - 30.*sin_phi**2 + 3.

            elif i == 3:  # J5
                factor = -J/8. * self.mu/r * (self.Re/r)**5
                terms = 63.*sin_phi**5 - 70.*sin_phi**3 + 15.*sin_phi

            elif i == 4:  # J6
                factor = -J/16. * self.mu/r * (self.Re/r)**6
                terms = (231.*sin_phi**6 - 315.*sin_phi**4 +
                         105.*sin_phi**2 - 5.)

            V = V - factor*terms

        H = KE + V
        H_rel = (H - H[0, 0]) / H[0, 0]

        # plot
        plot = MultiPlotter(1, name="Zonal Gravity Hamiltonian",
                            size_inches=(10, 10))
        plot.add_data(0, T, H_rel)
        plot.set_axis_titles(0, 'time', 'total energy')
        plot.display()

        return H
