# -*- coding: utf-8 -*-
# =============================================================================
# solve_pressure.py
#
# The two functions the run scripts call:
#
#   prepare_poisson(grid_x, grid_y, method)
#       Everything that only depends on the grid: the coefficients A_p ... A_n,
#       the matrix, the left null vector (for the consistency fix) and, for
#       the direct solver, the LU factorisation. Call it ONCE per grid.
#
#   compute_pressure(u, v, dt, grid_x, grid_y, Ro, rotation, poisson, method)
#       Velocity -> pressure for one snapshot:
#         1. dp_dx, dp_dy and B          (equations.py)
#         2. make B consistent           (poisson_matrix.py)
#         3. solve with the chosen method:
#              "direct"        sparse LU (default: exact and by far the fastest)
#              "jacobi"        Jacobi iteration
#              "gauss-seidel"  red-black Gauss-Seidel
#              "sor"           red-black SOR
#
# All four methods solve exactly the same equations and give the same
# pressure (compare_solvers.py shows this).
# =============================================================================

from equations import pressure_equation
from coefficients import make_coefficients
from poisson_matrix import build_matrix, left_null_vector, make_consistent
from solvers.direct import factorise, solve_direct
from solvers.jacobi import solve_jacobi
from solvers.gauss_seidel import solve_gauss_seidel
from solvers.sor import solve_sor

METHODS = ["direct", "jacobi", "gauss-seidel", "sor"]


def prepare_poisson(grid_x, grid_y, method="direct"):
    A_p, A_w, A_e, A_s, A_n = make_coefficients(grid_x, grid_y)
    A = build_matrix(A_p, A_w, A_e, A_s, A_n)
    l = left_null_vector(A, A_p)

    lu = None
    if method == "direct":
        lu = factorise(A, A_p)

    poisson = {}
    poisson["A_p"] = A_p
    poisson["A_w"] = A_w
    poisson["A_e"] = A_e
    poisson["A_s"] = A_s
    poisson["A_n"] = A_n
    poisson["l"] = l
    poisson["lu"] = lu
    return poisson


def compute_pressure(u, v, dt, grid_x, grid_y, Ro, rotation, poisson,
                     method="direct", z_terms=None, tolerance=1e-9):
    # Returns: pressure X, vorticity Vor, number of iterations, final residual,
    # and the "uniform source" (how badly the data fit together, see poisson_matrix.py)

    A_p = poisson["A_p"]
    A_w = poisson["A_w"]
    A_e = poisson["A_e"]
    A_s = poisson["A_s"]
    A_n = poisson["A_n"]

    # 1. the equations
    dp_dx, dp_dy, B, Vor, Div_xy = pressure_equation(u, v, dt, grid_x, grid_y, Ro, rotation, z_terms)

    # 2. make the data consistent
    B, uniform_source = make_consistent(B, dp_dx, dp_dy, A_w, A_e, A_s, A_n, grid_x, grid_y, poisson["l"])

    # 3. solve
    if method == "direct":
        if poisson["lu"] is None:
            print("For method 'direct', call prepare_poisson(grid_x, grid_y, 'direct') first")
            return None
        X, iterations, e = solve_direct(B, dp_dx, dp_dy, grid_x, grid_y, A_p, A_w, A_e, A_s, A_n,
                                        poisson["lu"])
    elif method == "jacobi":
        X, iterations, e = solve_jacobi(B, dp_dx, dp_dy, grid_x, grid_y, A_p, A_w, A_e, A_s, A_n,
                                        tolerance)
    elif method == "gauss-seidel":
        X, iterations, e = solve_gauss_seidel(B, dp_dx, dp_dy, grid_x, grid_y, A_p, A_w, A_e, A_s, A_n,
                                              tolerance)
    elif method == "sor":
        X, iterations, e = solve_sor(B, dp_dx, dp_dy, grid_x, grid_y, A_p, A_w, A_e, A_s, A_n,
                                     tolerance)
    else:
        print("Unknown method:", method, " choose one of", METHODS)
        return None

    return X, Vor, iterations, e, uniform_source
