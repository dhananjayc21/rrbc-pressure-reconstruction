# -*- coding: utf-8 -*-
# =============================================================================
# solvers/jacobi.py
#
# The Jacobi method of the notebook: every solved point gets the value its
# four neighbours give it, all points at the same time (from the OLD values),
# then the ghost cells are refreshed, and this is repeated until the
# equations are satisfied.
#
# Two small additions to the notebook:
#
#   1. The mean of X is removed after every iteration. The pressure is only
#      known up to a constant, so without this the whole field can slowly
#      drift up or down.
#
#   2. weight < 1 ("weighted Jacobi"):  X = (1 - weight) X_old + weight X_new.
#      Plain Jacobi (weight = 1) never removes a chess-board pattern
#      (+a, -a, +a, ...): it just flips its sign every iteration. weight = 0.9
#      damps it. Use weight = 1 to get exactly the notebook.
#
# The loop stops when the residual (how well the equations are satisfied,
# see coefficients.residual) is below the tolerance.
# Jacobi is simple but slow: about 10^5 iterations on the PIV grid.
# =============================================================================

import numpy as np

from coefficients import (solved_range, set_ghost_cells, value_from_neighbours,
                          residual, remove_mean, outside_to_nan)


def solve_jacobi(B, dp_dx, dp_dy, grid_x, grid_y, A_p, A_w, A_e, A_s, A_n,
                 tolerance=1e-9, weight=0.9, max_iterations=500000):
    Nx, Ny = B.shape
    x0, x1 = solved_range(Nx)
    y0, y1 = solved_range(Ny)

    X = np.zeros((Nx, Ny))
    set_ghost_cells(X, dp_dx, dp_dy, grid_x, grid_y)

    i = 0           # iteration counter
    e = 1000        # residual
    while e > tolerance and i < max_iterations:

        X_new = value_from_neighbours(X, B, A_p, A_w, A_e, A_s, A_n)
        X[x0:x1, y0:y1] = (1 - weight) * X[x0:x1, y0:y1] + weight * X_new

        remove_mean(X, dp_dx, dp_dy, grid_x, grid_y)      # also refreshes the ghost cells
        i = i + 1

        if i % 100 == 0:                                   # checking costs time: not every iteration
            e = residual(X, B, A_p, A_w, A_e, A_s, A_n)
        if i % 20000 == 0:
            print("    Jacobi iteration", i, " residual", e)

    return outside_to_nan(X), i, e
