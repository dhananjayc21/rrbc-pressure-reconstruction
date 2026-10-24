# -*- coding: utf-8 -*-
# =============================================================================
# coefficients.py
#
# The left-hand side of the pressure Poisson equation, as in the notebook:
# at every solved point (x, y)
#
#   A_p * X[x][y] + A_w * X[x-1][y] + A_e * X[x+1][y] + A_s * X[x][y-1] + A_n * X[x][y+1] = B[x][y]
#
# (w/e = west/east = x-1/x+1, s/n = south/north = y-1/y+1)
#
# On the uniform PIV grid this is exactly the notebook:
#     A_p = -2 (1/dx^2 + 1/dy^2),  A_w = A_e = 1/dx^2,  A_s = A_n = 1/dy^2.
# On the stretched DNS grid the weights also contain the grid factors x_xi and
# x_xixi (chain rule, see grid.py).
#
# The pressure is solved at the points MARGIN ... N-MARGIN-1 in each direction
# (MARGIN = 3: X[3:-3, 3:-3], as in the notebook). The ring just outside,
# index 2 and N-3, holds the GHOST values that impose the Neumann conditions:
#     X[2, :]  = X[4, :]  - 2 dx dp_dx[3, :]      (and the same on the other sides)
# =============================================================================

import numpy as np

import settings


def solved_range(N):
    # first and last+1 index of the solved points in one direction
    first = settings.MARGIN
    last = N - settings.MARGIN
    return first, last


def make_coefficients(grid_x, grid_y):
    # A_p, A_w, A_e, A_s, A_n on the full grid (zero outside the solved points)
    Nx = len(grid_x["coords"])
    Ny = len(grid_y["coords"])
    hx = grid_x["d_xi"]
    hy = grid_y["d_xi"]

    A_p = np.zeros((Nx, Ny))
    A_w = np.zeros((Nx, Ny))
    A_e = np.zeros((Nx, Ny))
    A_s = np.zeros((Nx, Ny))
    A_n = np.zeros((Nx, Ny))

    x0, x1 = solved_range(Nx)
    y0, y1 = solved_range(Ny)

    for x in range(x0, x1):
        # chain-rule factors in x:  d2p/dx2 = ax2 * d2p/dxi2 + ax1 * dp/dxi
        ax2 = 1 / grid_x["x_xi"][x] ** 2
        ax1 = -grid_x["x_xixi"][x] / grid_x["x_xi"][x] ** 3

        for y in range(y0, y1):
            ay2 = 1 / grid_y["x_xi"][y] ** 2
            ay1 = -grid_y["x_xixi"][y] / grid_y["x_xi"][y] ** 3

            A_w[x][y] = ax2 / hx ** 2 - ax1 / (2 * hx)
            A_e[x][y] = ax2 / hx ** 2 + ax1 / (2 * hx)
            A_s[x][y] = ay2 / hy ** 2 - ay1 / (2 * hy)
            A_n[x][y] = ay2 / hy ** 2 + ay1 / (2 * hy)
            A_p[x][y] = -2 * (ax2 / hx ** 2 + ay2 / hy ** 2)

    return A_p, A_w, A_e, A_s, A_n


def set_ghost_cells(X, dp_dx, dp_dy, grid_x, grid_y):
    # Fill the ghost ring of X from the Neumann conditions (changes X).
    # Central difference about the first solved point; on the west side:
    #     (X[x0+1] - X[x0-1]) / (2 dxi) = x_xi * dp/dx    ->   X[x0-1] = ...
    Nx, Ny = X.shape
    x0, x1 = solved_range(Nx)
    y0, y1 = solved_range(Ny)
    hx = grid_x["d_xi"]
    hy = grid_y["d_xi"]
    x_xi = grid_x["x_xi"]
    y_xi = grid_y["x_xi"]

    # west and east ghost rows
    X[x0 - 1, y0:y1] = X[x0 + 1, y0:y1] - 2 * hx * x_xi[x0] * dp_dx[x0, y0:y1]
    X[x1, y0:y1] = X[x1 - 2, y0:y1] + 2 * hx * x_xi[x1 - 1] * dp_dx[x1 - 1, y0:y1]

    # south and north ghost columns (these also fill the four corners)
    X[x0 - 1:x1 + 1, y0 - 1] = X[x0 - 1:x1 + 1, y0 + 1] - 2 * hy * y_xi[y0] * dp_dy[x0 - 1:x1 + 1, y0]
    X[x0 - 1:x1 + 1, y1] = X[x0 - 1:x1 + 1, y1 - 2] + 2 * hy * y_xi[y1 - 1] * dp_dy[x0 - 1:x1 + 1, y1 - 1]


def value_from_neighbours(X, B, A_p, A_w, A_e, A_s, A_n):
    # The value every solved point gets from its four neighbours (the Jacobi
    # formula of the notebook). Returned for the solved points only.
    Nx, Ny = X.shape
    x0, x1 = solved_range(Nx)
    y0, y1 = solved_range(Ny)

    X_new = (B[x0:x1, y0:y1]
             - A_w[x0:x1, y0:y1] * X[x0 - 1:x1 - 1, y0:y1]
             - A_e[x0:x1, y0:y1] * X[x0 + 1:x1 + 1, y0:y1]
             - A_s[x0:x1, y0:y1] * X[x0:x1, y0 - 1:y1 - 1]
             - A_n[x0:x1, y0:y1] * X[x0:x1, y0 + 1:y1 + 1]) / A_p[x0:x1, y0:y1]
    return X_new


def residual(X, B, A_p, A_w, A_e, A_s, A_n):
    # How well X satisfies the equations: max |A X - B| / max |B| over the
    # solved points (X must have its ghost cells). About 1e-12 when converged.
    Nx, Ny = X.shape
    x0, x1 = solved_range(Nx)
    y0, y1 = solved_range(Ny)

    r = (A_p[x0:x1, y0:y1] * X[x0:x1, y0:y1]
         + A_w[x0:x1, y0:y1] * X[x0 - 1:x1 - 1, y0:y1]
         + A_e[x0:x1, y0:y1] * X[x0 + 1:x1 + 1, y0:y1]
         + A_s[x0:x1, y0:y1] * X[x0:x1, y0 - 1:y1 - 1]
         + A_n[x0:x1, y0:y1] * X[x0:x1, y0 + 1:y1 + 1]
         - B[x0:x1, y0:y1])
    return np.max(np.abs(r)) / np.max(np.abs(B[x0:x1, y0:y1]))


def remove_mean(X, dp_dx, dp_dy, grid_x, grid_y):
    # The pressure is only known up to a constant: keep its mean at zero over
    # the solved points, then refresh the ghost cells (they move with it).
    Nx, Ny = X.shape
    x0, x1 = solved_range(Nx)
    y0, y1 = solved_range(Ny)
    X[x0:x1, y0:y1] = X[x0:x1, y0:y1] - np.mean(X[x0:x1, y0:y1])
    set_ghost_cells(X, dp_dx, dp_dy, grid_x, grid_y)


def red_black(Nx, Ny):
    # Colour the solved points like a chess board: red where x + y is even,
    # black where it is odd. Used by Gauss-Seidel and SOR.
    x0, x1 = solved_range(Nx)
    y0, y1 = solved_range(Ny)
    red = np.zeros((x1 - x0, y1 - y0), dtype=bool)
    for i in range(x1 - x0):
        for j in range(y1 - y0):
            if (x0 + i + y0 + j) % 2 == 0:
                red[i][j] = True
    black = ~red
    return red, black


def outside_to_nan(X):
    # Points that are not solved and not ghost points are set to NaN
    Nx, Ny = X.shape
    x0, x1 = solved_range(Nx)
    y0, y1 = solved_range(Ny)
    result = np.full((Nx, Ny), np.nan)
    result[x0 - 1:x1 + 1, y0 - 1:y1 + 1] = X[x0 - 1:x1 + 1, y0 - 1:y1 + 1]
    return result
