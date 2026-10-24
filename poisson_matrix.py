# -*- coding: utf-8 -*-
# =============================================================================
# poisson_matrix.py
#
# The equations of all solved points written as ONE linear system  A x = b,
# with the ghost values taken out. Used by the direct solver, and to make the
# data consistent before any solver starts.
#
# Taking out the ghost values: on the west side the ghost value is
#     X[ghost] = X[x0+1] - 2 dxi x_xi dp_dx            (known part K = 2 dxi x_xi dp_dx)
# so in the equation of the first solved point
#     A_w X[ghost] = A_w X[x0+1] - A_w K
# the weight A_w moves to the point x0+1 and A_w K moves to the right-hand
# side. The same happens on the other three sides.
#
# Consistent data
# ---------------
# With Neumann conditions on every side the pressure is only defined up to a
# constant, and a solution only exists if B and the boundary values fit
# together (divergence theorem). Measured data never fit exactly, so no
# pressure field satisfies every equation: there is always a small uniform
# leftover "mu" at every point. make_consistent() removes mu from B first, so
# that every solver solves equations that DO have a solution. Without this the
# iterative solvers stall and never reach a small residual.
#
# mu = (l . b) / (sum of l), where l is the "left null vector" of the matrix
# (the vector with A^T l = 0). l only depends on the grid, so it is computed
# once per grid.
# =============================================================================

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from coefficients import solved_range


def build_matrix(A_p, A_w, A_e, A_s, A_n):
    # Sparse matrix of all solved points (ghost values taken out)
    Nx, Ny = A_p.shape
    x0, x1 = solved_range(Nx)
    y0, y1 = solved_range(Ny)
    nx = x1 - x0          # number of solved points in x
    ny = y1 - y0          # number of solved points in y

    rows = []
    cols = []
    vals = []
    for i in range(nx):
        for j in range(ny):
            x = x0 + i            # position on the full grid
            y = y0 + j
            row = i * ny + j      # number of this point in the system

            # neighbours; at the edge the ghost neighbour is replaced by the
            # point on the other side (see the top of this file)
            if i == 0:
                west = (i + 1) * ny + j
            else:
                west = (i - 1) * ny + j
            if i == nx - 1:
                east = (i - 1) * ny + j
            else:
                east = (i + 1) * ny + j
            if j == 0:
                south = i * ny + (j + 1)
            else:
                south = i * ny + (j - 1)
            if j == ny - 1:
                north = i * ny + (j - 1)
            else:
                north = i * ny + (j + 1)

            rows += [row, row, row, row, row]
            cols += [row, west, east, south, north]
            vals += [A_p[x][y], A_w[x][y], A_e[x][y], A_s[x][y], A_n[x][y]]

    # entries with the same (row, col) are added together by coo_matrix
    A = coo_matrix((vals, (rows, cols)), shape=(nx * ny, nx * ny))
    return A.tocsr()


def boundary_part(dp_dx, dp_dy, A_w, A_e, A_s, A_n, grid_x, grid_y):
    # The known part of the ghost values, as it appears on the right-hand side
    # of A x = b (for the solved points only)
    Nx, Ny = A_w.shape
    x0, x1 = solved_range(Nx)
    y0, y1 = solved_range(Ny)
    hx = grid_x["d_xi"]
    hy = grid_y["d_xi"]
    x_xi = grid_x["x_xi"]
    y_xi = grid_y["x_xi"]

    bc = np.zeros((x1 - x0, y1 - y0))
    bc[0, :] = bc[0, :] + A_w[x0, y0:y1] * 2 * hx * x_xi[x0] * dp_dx[x0, y0:y1]
    bc[-1, :] = bc[-1, :] - A_e[x1 - 1, y0:y1] * 2 * hx * x_xi[x1 - 1] * dp_dx[x1 - 1, y0:y1]
    bc[:, 0] = bc[:, 0] + A_s[x0:x1, y0] * 2 * hy * y_xi[y0] * dp_dy[x0:x1, y0]
    bc[:, -1] = bc[:, -1] - A_n[x0:x1, y1 - 1] * 2 * hy * y_xi[y1 - 1] * dp_dy[x0:x1, y1 - 1]
    return bc


def fixed_point(A_p):
    # The point fixed at p = 0 by the direct solver: the centre of the solved points
    Nx, Ny = A_p.shape
    x0, x1 = solved_range(Nx)
    y0, y1 = solved_range(Ny)
    nx = x1 - x0
    ny = y1 - y0
    return (nx // 2) * ny + ny // 2


def matrix_with_fixed_point(A, A_p):
    # A has no inverse (adding a constant to p changes nothing). Fixing one
    # point removes that freedom: its row is replaced by  c * p_k = 0.
    # c is a typical diagonal value, so that this row has the same size as the others.
    k = fixed_point(A_p)
    Nx, Ny = A_p.shape
    x0, x1 = solved_range(Nx)
    y0, y1 = solved_range(Ny)
    c = np.mean(np.abs(A_p[x0:x1, y0:y1]))

    A_fixed = A.tolil()
    A_fixed[k, :] = 0
    A_fixed[k, k] = c
    return A_fixed.tocsc()


def left_null_vector(A, A_p):
    # The vector l with A^T l = 0, with l = 1 at the fixed point.
    # Written out: sum over i (not k) of l_i * row_i = - row_k.  That is solved
    # with the fixed-point matrix (its row k is c * e_k, which gives l_k = 0
    # there), and then l_k is set to 1.
    k = fixed_point(A_p)
    A_fixed = matrix_with_fixed_point(A, A_p)
    row_k = A[k, :].toarray().ravel()
    l = spsolve(A_fixed.T.tocsc(), -row_k)
    l[k] = 1.0
    return l


def make_consistent(B, dp_dx, dp_dy, A_w, A_e, A_s, A_n, grid_x, grid_y, l):
    # Remove the uniform leftover mu from B (see the top of this file).
    # Returns the new B and the "uniform source" |mu| / RMS(B): a measure of
    # how badly the data fit together (it describes the data, not the solver).
    Nx, Ny = B.shape
    x0, x1 = solved_range(Nx)
    y0, y1 = solved_range(Ny)

    b = B[x0:x1, y0:y1] + boundary_part(dp_dx, dp_dy, A_w, A_e, A_s, A_n, grid_x, grid_y)
    b = b.ravel()
    if np.any(np.isnan(b)):
        print("NaN in B or in the boundary values: check the velocity data near the edges!")

    mu = np.sum(l * b) / np.sum(l)
    B_ok = B.copy()
    B_ok[x0:x1, y0:y1] = B_ok[x0:x1, y0:y1] - mu
    uniform_source = abs(mu) / np.sqrt(np.mean(B[x0:x1, y0:y1] ** 2))
    return B_ok, uniform_source
