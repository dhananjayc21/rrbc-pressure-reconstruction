# -*- coding: utf-8 -*-
# =============================================================================
# derivatives.py
#
# Space derivatives of a 2-D field f[x][y] (x along the first index, y along
# the second), on the uniform PIV grid or the stretched DNS grid.
# grid_x and grid_y are the grid dictionaries of grid.py.
#
# First derivatives: central difference inside, and a one-sided difference at
# the two end points (so the derivative has a value everywhere).
# Second derivatives: central difference inside; the two end points are NaN,
# they are never used by the pressure solve.
# =============================================================================

import numpy as np


def d_dx(f, grid_x):
    Nx, Ny = f.shape
    h = grid_x["d_xi"]
    df = np.zeros((Nx, Ny))

    df[1:-1, :] = (f[2:, :] - f[:-2, :]) / (2 * h)                  # inside
    df[0, :] = (-3 * f[0, :] + 4 * f[1, :] - f[2, :]) / (2 * h)     # first point
    df[-1, :] = (3 * f[-1, :] - 4 * f[-2, :] + f[-3, :]) / (2 * h)  # last point

    # chain rule: divide by dx/dxi (= 1 on the PIV grid)
    for i in range(Nx):
        df[i, :] = df[i, :] / grid_x["x_xi"][i]
    return df


def d_dy(f, grid_y):
    Nx, Ny = f.shape
    h = grid_y["d_xi"]
    df = np.zeros((Nx, Ny))

    df[:, 1:-1] = (f[:, 2:] - f[:, :-2]) / (2 * h)
    df[:, 0] = (-3 * f[:, 0] + 4 * f[:, 1] - f[:, 2]) / (2 * h)
    df[:, -1] = (3 * f[:, -1] - 4 * f[:, -2] + f[:, -3]) / (2 * h)

    for j in range(Ny):
        df[:, j] = df[:, j] / grid_y["x_xi"][j]
    return df


def d2_dx2(f, grid_x):
    Nx, Ny = f.shape
    h = grid_x["d_xi"]
    df = np.full((Nx, Ny), np.nan)      # df/dxi
    d2f = np.full((Nx, Ny), np.nan)     # d2f/dxi2

    df[1:-1, :] = (f[2:, :] - f[:-2, :]) / (2 * h)
    d2f[1:-1, :] = (f[2:, :] - 2 * f[1:-1, :] + f[:-2, :]) / (h * h)

    # chain rule (on the PIV grid x_xi = 1 and x_xixi = 0, so this is just d2f)
    result = np.full((Nx, Ny), np.nan)
    for i in range(Nx):
        x_xi = grid_x["x_xi"][i]
        x_xixi = grid_x["x_xixi"][i]
        result[i, :] = d2f[i, :] / x_xi ** 2 - (x_xixi / x_xi ** 3) * df[i, :]
    return result


def d2_dy2(f, grid_y):
    Nx, Ny = f.shape
    h = grid_y["d_xi"]
    df = np.full((Nx, Ny), np.nan)
    d2f = np.full((Nx, Ny), np.nan)

    df[:, 1:-1] = (f[:, 2:] - f[:, :-2]) / (2 * h)
    d2f[:, 1:-1] = (f[:, 2:] - 2 * f[:, 1:-1] + f[:, :-2]) / (h * h)

    result = np.full((Nx, Ny), np.nan)
    for j in range(Ny):
        y_xi = grid_y["x_xi"][j]
        y_xixi = grid_y["x_xixi"][j]
        result[:, j] = d2f[:, j] / y_xi ** 2 - (y_xixi / y_xi ** 3) * df[:, j]
    return result
