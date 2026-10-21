# -*- coding: utf-8 -*-
"""
derivatives.py
==============

Second-order central differences on a 2-D field f[i, j]
(i along x, j along y), on the uniform PIV grid or the stretched DNS grid.

    gx, gy   the grid dictionaries of x and y (see grid.py)

First derivatives: central difference inside, and a second-order one-sided
difference at the two end points, so the result is defined everywhere.

Second derivatives: central difference inside; the two end points are set to
NaN (not a number). They are never needed by the pressure solve, and a NaN
makes sure that a value that should not be used cannot be mistaken for a real
one.
"""

import numpy as np


def d_dx(f, gx):
    """df/dx of a 2-D field."""
    h = gx["d_xi"]
    df = np.empty_like(f)
    df[1:-1, :] = (f[2:, :] - f[:-2, :]) / (2 * h)               # inside
    df[0, :] = (-3 * f[0, :] + 4 * f[1, :] - f[2, :]) / (2 * h)  # first point
    df[-1, :] = (3 * f[-1, :] - 4 * f[-2, :] + f[-3, :]) / (2 * h)  # last point
    return df / gx["x_xi"][:, None]           # chain rule: divide by dx/dxi


def d_dy(f, gy):
    """df/dy of a 2-D field."""
    h = gy["d_xi"]
    df = np.empty_like(f)
    df[:, 1:-1] = (f[:, 2:] - f[:, :-2]) / (2 * h)
    df[:, 0] = (-3 * f[:, 0] + 4 * f[:, 1] - f[:, 2]) / (2 * h)
    df[:, -1] = (3 * f[:, -1] - 4 * f[:, -2] + f[:, -3]) / (2 * h)
    return df / gy["x_xi"][None, :]


def d2_dx2(f, gx):
    """d2f/dx2 of a 2-D field (NaN on the first and last row)."""
    h = gx["d_xi"]
    df = np.full_like(f, np.nan)        # df/dxi
    d2f = np.full_like(f, np.nan)       # d2f/dxi2
    df[1:-1, :] = (f[2:, :] - f[:-2, :]) / (2 * h)
    d2f[1:-1, :] = (f[2:, :] - 2 * f[1:-1, :] + f[:-2, :]) / h ** 2
    x_xi = gx["x_xi"][:, None]
    x_xixi = gx["x_xixi"][:, None]
    return d2f / x_xi ** 2 - (x_xixi / x_xi ** 3) * df


def d2_dy2(f, gy):
    """d2f/dy2 of a 2-D field (NaN on the first and last column)."""
    h = gy["d_xi"]
    df = np.full_like(f, np.nan)
    d2f = np.full_like(f, np.nan)
    df[:, 1:-1] = (f[:, 2:] - f[:, :-2]) / (2 * h)
    d2f[:, 1:-1] = (f[:, 2:] - 2 * f[:, 1:-1] + f[:, :-2]) / h ** 2
    y_xi = gy["x_xi"][None, :]
    y_xixi = gy["x_xixi"][None, :]
    return d2f / y_xi ** 2 - (y_xixi / y_xi ** 3) * df
