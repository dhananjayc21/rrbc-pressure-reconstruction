# -*- coding: utf-8 -*-
# =============================================================================
# solvers/direct.py
#
# Direct solver: sparse LU factorisation (scipy splu), as in the DNS notebook.
#
#   1. All equations are written as one system A x = b (poisson_matrix.py).
#   2. A is factorised ONCE: A = L U. This is the slow step (about a second
#      on the DNS grid) and only depends on the grid.
#   3. Every snapshot then only needs a quick forward and backward
#      substitution with L and U (about 0.01 s for a PIV field).
#
# So: call factorise() once per grid, then solve_direct() for every snapshot.
#
# One point (the centre) is fixed at p = 0, because adding a constant to p
# changes nothing. The mean is removed at the end.
# =============================================================================

import numpy as np
from scipy.sparse.linalg import splu

from coefficients import solved_range, set_ghost_cells, residual, outside_to_nan
from poisson_matrix import boundary_part, fixed_point, matrix_with_fixed_point


def factorise(A, A_p):
    # LU factorisation of the matrix with the fixed point (once per grid)
    A_fixed = matrix_with_fixed_point(A, A_p)
    lu = splu(A_fixed, permc_spec="MMD_AT_PLUS_A")     # same ordering as the DNS notebook
    return lu


def solve_direct(B, dp_dx, dp_dy, grid_x, grid_y, A_p, A_w, A_e, A_s, A_n, lu):
    Nx, Ny = B.shape
    x0, x1 = solved_range(Nx)
    y0, y1 = solved_range(Ny)

    b = B[x0:x1, y0:y1] + boundary_part(dp_dx, dp_dy, A_w, A_e, A_s, A_n, grid_x, grid_y)
    b = b.ravel()
    b[fixed_point(A_p)] = 0.0          # the fixed point: p = 0 there

    x = lu.solve(b)
    x = x - np.mean(x)

    X = np.zeros((Nx, Ny))
    X[x0:x1, y0:y1] = x.reshape((x1 - x0, y1 - y0))
    set_ghost_cells(X, dp_dx, dp_dy, grid_x, grid_y)

    e = residual(X, B, A_p, A_w, A_e, A_s, A_n)
    return outside_to_nan(X), 1, e
