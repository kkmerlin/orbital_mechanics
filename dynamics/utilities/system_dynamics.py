"""Created on Wed Sep 10 2015 11:06.

@author: Nathan Budd
"""


import numpy as np


class SystemDynamics():
    """Combines a plant model, control, and perturbations.

    A callable object that combines other callable objects representing a
    system's model, control, and perturbations. Each of these objects must take
    inputs (T, X) where T is an mx1 ndarray of sample times and X is an mxn
    ndarray of sample states. SystemDynamics is callable with the same inputs,
    and returns the sum of its constituent parts (model, control,
    perturbations) when called.

    Members
    -------
    plant : callable
        Represents the idealized unperturbed model.
    control : callable
        Represents the system control.
    preturbations : callable or list of callables
        Represent perturbations acting on the system.
    Xdot : ndarray
        An mxn array of state derivatives
    """

    def __init__(self, plant, control=None, perturbations=None):
        """."""
        self.plant = plant
        self.control = control
        self.perturbations = perturbations
        self.Xdot = np.array([[]])
        super().__init__()

    def __call__(self, T, X):
        """Evaluate the full system dynamics.

        Parameters
        ----------
        T : ndarray
            An mx1 column array of times.
        X : ndarray
            An mxn array of states.

        Returns
        -------
        self.Xdot
        """
        self.Xdot = self.plant(T, X)

        if self.control is not None:
            self.Xdot += self.control(T, X)

        if self.perturbations is not None:
            if isinstance(self.perturbations, list):
                for perturb in self.perturbations:
                    self.Xdot += perturb(T, X)
            else:
                self.Xdot += self.perturbations(T, X)

        return self.Xdot
