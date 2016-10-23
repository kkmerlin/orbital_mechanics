"""Created on Wed Sep 10 2015 11:06.

@author: Nathan Budd
"""


class SystemDynamics():
    """Combines a plant model, control, and perturbations.

    Plant, control, and perturbations are all callables which take inputs
    (T, X) where T is an mx1 ndarray of sample times and X is an mxn ndarray
    of sample states.

    Members
    -------
    plant : callable
        Represents the idealized unperturbed model.
    control : callable
        Represents the system control.
    preturbations : callable or list of callables
        Represent perturbations acting on the system.
    """

    def __init__(self, plant, control=None, perturbations=None):
        """."""
        self.plant = plant
        self.control = control
        self.perturbations = perturbations
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
        Xdot : ndarray
            An mxn array of state derivatives
        """
        Xdot = self.plant(T, X)

        if self.control is not None:
            Xdot += self.control(T, X)

        if self.perturbations is not None:
            if isinstance(self.perturbations, list):
                for perturb in self.perturbations:
                    Xdot = Xdot + perturb(T, X)
            else:
                Xdot += self.perturbations(T, X)

        self.Xdot = Xdot
        return Xdot

    def __repr__(self):
        """Printable represenation of the object."""
        return 'SystemDynamics({}, {}, {})'.format(
            self.plant, self.control, self.perturbations)

    def __str__(self):
        """Human readable represenation of the object."""
        output = 'SystemDynamics'
        output += '(plant={}, control={}, perturbations={})'.format(
            self.plant, self.control, self.perturbations)
        return output
