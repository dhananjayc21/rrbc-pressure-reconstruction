# -*- coding: utf-8 -*-
# =============================================================================
# grid.py
#
# The grid of one direction (x, y or z). A grid is kept in a small dictionary:
#
#     grid["coords"]   positions of the points (dimensionless, x/H)
#     grid["d_xi"]     spacing of the "computational" coordinate xi
#     grid["x_xi"]     dx/dxi at every point
#     grid["x_xixi"]   d2x/dxi2 at every point
#
# Why xi? The DNS grid is stretched (points are closer together near the
# walls). Derivatives are easy on an equally spaced grid, so every point also
# gets a coordinate xi that IS equally spaced. Derivatives are taken with
# respect to xi and then changed back with the chain rule:
#
#     df/dx   = (df/dxi) / x_xi
#     d2f/dx2 = (d2f/dxi2) / x_xi^2  -  (x_xixi / x_xi^3) * (df/dxi)
#
# The PIV grid is already equally spaced, so there we simply take xi = x:
# then x_xi = 1 and x_xixi = 0 and the formulas become the ordinary ones.
# This is why the same derivative code works for PIV and for DNS.
# =============================================================================

import numpy as np


def uniform_grid(coords):
    # Grid of equally spaced points (the PIV grid)
    coords = np.array(coords, dtype=float)
    N = len(coords)

    spacing = coords[1] - coords[0]
    for i in range(1, N):
        if abs((coords[i] - coords[i - 1]) - spacing) > 1e-6 * abs(spacing):
            print("uniform_grid: the points are NOT equally spaced!")
            break

    grid = {}
    grid["coords"] = coords
    grid["d_xi"] = spacing
    grid["x_xi"] = np.ones(N)       # dx/dxi = 1
    grid["x_xixi"] = np.zeros(N)    # d2x/dxi2 = 0
    return grid


def tanh_coords(N, L=1.0, beta=1.3):
    # Positions of the N points of the DNS grid (with one extra point at each end)
    #   x(eps) = L/2 * (1 - tanh(beta * (1 - 2 eps)) / tanh(beta))
    eps = np.linspace(-1 / (2 * (N - 2)), 1 + 1 / (2 * (N - 2)), N)
    x = 0.5 * L * (1 - np.tanh(beta * (1 - 2 * eps)) / np.tanh(beta))
    return x


def tanh_grid(coords, L=1.0, beta=1.3):
    # Grid of the stretched DNS points. x_xi and x_xixi are the exact
    # derivatives of the tanh stretching function (as in the DNS notebook).
    coords = np.array(coords, dtype=float)
    N = len(coords)
    eps = np.linspace(-1 / (2 * (N - 2)), 1 + 1 / (2 * (N - 2)), N)
    s = beta * (1 - 2 * eps)

    grid = {}
    grid["coords"] = coords
    grid["d_xi"] = eps[1] - eps[0]
    grid["x_xi"] = L * beta / np.tanh(beta) / np.cosh(s) ** 2
    grid["x_xixi"] = 4 * L * beta ** 2 / np.tanh(beta) / np.cosh(s) ** 2 * np.tanh(s)
    return grid
