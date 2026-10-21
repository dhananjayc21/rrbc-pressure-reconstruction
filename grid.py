# -*- coding: utf-8 -*-
"""
grid.py
=======

The grid of one direction (x or y or z) is stored in a small dictionary:

    grid["coords"]  positions of the grid points (non-dimensional, x/H)
    grid["d_xi"]    spacing of the computational coordinate xi
    grid["x_xi"]    dx/dxi at every grid point
    grid["x_xixi"]  d2x/dxi2 at every grid point

Why "xi"?  The DNS grid is stretched: the points are closer together near the
walls. Derivatives are easiest on an equally spaced grid, so every grid point
also gets a coordinate xi that IS equally spaced, and the derivatives are
taken with respect to xi and then converted back with the chain rule:

    df/dx   = (df/dxi) / x_xi
    d2f/dx2 = (d2f/dxi2) / x_xi^2  -  (x_xixi / x_xi^3) * (df/dxi)

For the PIV data the grid is already equally spaced, so we simply take xi = x:
then x_xi = 1 and x_xixi = 0, and the formulas above reduce to the ordinary
ones. That is how the same code works for both data sets.
"""

import numpy as np


def uniform_grid(coords):
    """Grid dictionary for equally spaced points (the PIV grid)."""
    coords = np.asarray(coords, dtype=np.float64)
    spacing = np.diff(coords)
    if not np.allclose(spacing, spacing[0], rtol=1e-6):
        raise ValueError("uniform_grid: the points are not equally spaced")
    n = coords.size
    return {"coords": coords,
            "d_xi": float(spacing[0]),
            "x_xi": np.ones(n),
            "x_xixi": np.zeros(n)}


def tanh_coords(n, L=1.0, beta=1.3):
    """Positions of the n points of the padded DNS grid.

    The DNS points are cell centres of the stretching function
        x(eps) = L/2 * (1 - tanh(beta * (1 - 2 eps)) / tanh(beta)),
    with one extra (ghost) point at each end, so eps runs from
    -1/(2(n-2)) to 1 + 1/(2(n-2)).
    """
    eps = np.linspace(-1 / (2 * (n - 2)), 1 + 1 / (2 * (n - 2)), n)
    return 0.5 * L * (1 - np.tanh(beta * (1 - 2 * eps)) / np.tanh(beta))


def tanh_grid(coords, L=1.0, beta=1.3):
    """Grid dictionary for the stretched DNS grid.

    The metric terms x_xi and x_xixi are the derivatives of the stretching
    function above, evaluated exactly (as in the DNS notebook). coords itself
    is only used for plotting and for interpolation.
    """
    coords = np.asarray(coords, dtype=np.float64)
    n = coords.size
    eps = np.linspace(-1 / (2 * (n - 2)), 1 + 1 / (2 * (n - 2)), n)
    s = beta * (1 - 2 * eps)
    x_xi = L * beta / np.tanh(beta) / np.cosh(s) ** 2
    x_xixi = 4 * L * beta ** 2 / np.tanh(beta) / np.cosh(s) ** 2 * np.tanh(s)
    return {"coords": coords,
            "d_xi": float(eps[1] - eps[0]),
            "x_xi": x_xi,
            "x_xixi": x_xixi}
