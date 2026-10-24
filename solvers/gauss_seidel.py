# -*- coding: utf-8 -*-
# =============================================================================
# solvers/gauss_seidel.py
#
# Gauss-Seidel: like Jacobi, but a point uses the NEWEST values of its
# neighbours. Going through the points one by one in Python would be very
# slow, so the "red-black" version is used:
#
#   colour the solved points like a chess board (red: x + y even, black: odd).
#   A red point only has black neighbours, and the other way round, so
#     1. all RED points are updated at once, from their black neighbours,
#     2. then all BLACK points at once, already using the new red values.
#
# This needs about half as many iterations as Jacobi, and it has no
# chess-board problem. After each colour the ghost cells are refreshed, after
# each full iteration the mean is removed (see jacobi.py).
# =============================================================================

import numpy as np

from coefficients import (solved_range, set_ghost_cells, value_from_neighbours,
                          residual, remove_mean, red_black, outside_to_nan)


def solve_gauss_seidel(B, dp_dx, dp_dy, grid_x, grid_y, A_p, A_w, A_e, A_s, A_n,
                       tolerance=1e-9, max_iterations=300000):
    Nx, Ny = B.shape
    x0, x1 = solved_range(Nx)
    y0, y1 = solved_range(Ny)
    red, black = red_black(Nx, Ny)

    X = np.zeros((Nx, Ny))
    set_ghost_cells(X, dp_dx, dp_dy, grid_x, grid_y)

    i = 0
    e = 1000
    while e > tolerance and i < max_iterations:

        for colour in [red, black]:
            X_new = value_from_neighbours(X, B, A_p, A_w, A_e, A_s, A_n)
            inside = X[x0:x1, y0:y1]               # a view: changing it changes X
            inside[colour] = X_new[colour]         # only this colour changes
            set_ghost_cells(X, dp_dx, dp_dy, grid_x, grid_y)

        remove_mean(X, dp_dx, dp_dy, grid_x, grid_y)
        i = i + 1

        if i % 100 == 0:
            e = residual(X, B, A_p, A_w, A_e, A_s, A_n)
        if i % 20000 == 0:
            print("    Gauss-Seidel iteration", i, " residual", e)

    return outside_to_nan(X), i, e
