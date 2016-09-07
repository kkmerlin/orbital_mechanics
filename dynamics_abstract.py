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
    the number of states.

    Static Members
    --------------
    _class_string : string
    An abbreviated name for the dynamical system.

    _parameters : dictionary
    Parameters specific to the dynamical system.
    """
    @property
    def _class_
