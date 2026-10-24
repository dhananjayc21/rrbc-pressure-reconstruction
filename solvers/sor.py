# -*- coding: utf-8 -*-
# =============================================================================
# solvers/sor.py
#
# SOR (successive over-relaxation): Gauss-Seidel that takes a bigger step.
# Gauss-Seidel moves a point to its new value X_new; SOR moves it further in
# the same direction:
#
#     X = (1 - omega) X_old + omega X_new,        1 < omega < 2
#
# omega = 1 is exactly Gauss-Seidel. With a good omega the number of
# iterations drops from about 10^5 to about 10^3 on the PIV grid.
#
# The best omega depends on the grid:
#     omega_best = 2 / (1 + sqrt(1 - rho^2))
# where rho is how much one Jacobi iteration shrinks the slowest error
# pattern (the smoothest wave that fits in the domain). best_omega()
# estimates it from the number of points and the weights A_w ... A_n.
# =============================================================================

import numpy as np

from coefficients import (solved_range, set_ghost_cells, value_from_neighbours,
                          residual, remove_mean, red_black, outside_to_nan)


def best_omega(A_w, A_e, A_s, A_n):
    Nx, Ny = A_w.shape
    x0, x1 = solved_range(Nx)
    y0, y1 = solved_range(Ny)
    nx = x1 - x0
    ny = y1 - y0
    wx = np.mean(A_w[x0:x1, y0:y1] + A_e[x0:x1, y0:y1])     # weight of the x-neighbours
    wy = np.mean(A_s[x0:x1, y0:y1] + A_n[x0:x1, y0:y1])     # weight of the y-neighbours
    rho = (wx * np.cos(np.pi / nx) + wy * np.cos(np.pi / ny)) / (wx + wy)
    omega = 2 / (1 + np.sqrt(1 - rho ** 2))
    return omega


def solve_sor(B, dp_dx, dp_dy, grid_x, grid_y, A_p, A_w, A_e, A_s, A_n,
              tolerance=1e-9, omega=None, max_iterations=100000):
    Nx, Ny = B.shape
    x0, x1 = solved_range(Nx)
    y0, y1 = solved_range(Ny)
    red, black = red_black(Nx, Ny)
    if omega is None:
        omega = best_omega(A_w, A_e, A_s, A_n)

    X = np.zeros((Nx, Ny))
    set_ghost_cells(X, dp_dx, dp_dy, grid_x, grid_y)

    i = 0
    e = 1000
    while e > tolerance and i < max_iterations:

        for colour in [red, black]:
            X_new = value_from_neighbours(X, B, A_p, A_w, A_e, A_s, A_n)
            inside = X[x0:x1, y0:y1]
            inside[colour] = (1 - omega) * inside[colour] + omega * X_new[colour]
            set_ghost_cells(X, dp_dx, dp_dy, grid_x, grid_y)

        remove_mean(X, dp_dx, dp_dy, grid_x, grid_y)
        i = i + 1

        if i % 10 == 0:
            e = residual(X, B, A_p, A_w, A_e, A_s, A_n)
        if i % 1000 == 0:
            print("    SOR iteration", i, " residual", e)

    return outside_to_nan(X), i, e
