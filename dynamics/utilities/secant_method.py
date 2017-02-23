# Created by Nathan Budd on Mon, 20 Feb, 2017 @ 12:53

# Provides a generalized secant method for n-dimensional functions and states

import numpy as np
import numpy.linalg as npl


def SecantMethod(f, x0, x1, tol=1e-14, Nmax=50):
    """Uses secant method to find approximate root of f.

    Inputs
    f : callable
        Takes an array with shape x0.shape as input
    x0 : ndarray
        First guess for f(x) = 0
    X1 : ndarray
        Second guess for f(x) = 0
    tol : float
        Error allowed between successive values of f(x)
    Nmax : int
        Total number of allowed iterations

    Outputs
    x : ndarray
        Approximate root of f
    """

    above_tol = True
    below_Nmax = True
    N = 0

    while above_tol and below_Nmax:
        f0 = f(x0)
        f1 = f(x1)

        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)

        x0 = x1
        x1 = x2

        N += 1
        above_tol = True if npl.norm(f0-f1) > tol else False
        below_Nmax = True if N < Nmax else False

    if below_Nmax == False:
        print('Reached maximum iterations (50)')
    return x2
