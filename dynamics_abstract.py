"""Created on Wed Sep 07 2016 11:41.

@author: Nathan Budd
"""
from abc import ABCMeta, abstractmethod


class DynamicsAbstract(metaclass=ABCMeta):
    """An abstract parent class for orbital dynamics classes.

    Sublcasses allow a variable number of parameter inputs at instantiation to
    accomodate all types of dynamic systems. Each of these parameters can be
    accessed as if they were standalone parameters.

    Subclasses can be called with inputs (t, X) where t is an mx1 np.matrix and
    X is an mxn np.matrix, with m being the number of time steps and n being
    the number of states. The output is the state derivative history.

    Static Members
    --------------
    _class_string : string
    An abbreviated name for the dynamical system.

    _parameter_list : list
    Parameters specific to the dynamical system.
    """
    @property
    def _class_string(self):
        ...

    @property
    def _parameter_list(self):
        ...

    @abstractmethod
    def __init__(self, arg=[]):
        """.
        Input
        -----
        arg : list or dict
        Corresponding to the elements in _parameter_list

        Instance Members
        ----------------
        _parameters : dictionary
        Parameter names and values.
        """
        size = len(arg)

        try:
            self._parameters = {self._parameter_list[k]: arg[k]
                                for k in range(size)}
        except KeyError:
            self._parameters = arg

    @abstractmethod
    def __call__(self, T, X):
        """Evaluate the dynamics at the given times.

        Input
        -----
        T : np.matrix
        An mx1 column matrix of times.

        X : np.matrix
        An mxn matrix of states.

        Output
        ------
        Xdot : np.matrix
        An mxn matrix of state derivatives
        """
        pass

    def __getattr__(self, name):
        """Allow easy access to parameters.

        Called when dot notation is used to access an element of the set.

        Input
        -----
        name : string
        The argument provided in the dot notation (e.g. name = arg when this
        expression is used: obj.arg)
        """
        return self._parameters[name]

    def __setattr__(self, name, value):
        """Allow changes to parameters.

        Called when dot notation is used to change an element of the standard
        element set.

        Input
        -----
        name : string
        The argument provided in the dot notation (e.g. name = arg when this
        expression is used: obj.arg)

        value : float or list of floats
        Value to be saved to self.name
        """
        if name == '_parameters':
            super().__setattr__(name, value)
        else:
            self._parameters[name] = value
