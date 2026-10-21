# -*- coding: utf-8 -*-
"""
solve_pressure.py
=================

One function that the run scripts call: velocity on a plane -> pressure.
It computes the equation terms (equations.py) and hands them to the solver
chosen by name:

    "splu"          direct sparse LU (default; exact and fastest)
    "jacobi"        weighted Jacobi iteration
    "gauss-seidel"  red-black Gauss-Seidel iteration
    "sor"           red-black SOR iteration

All four solve exactly the same discrete equations, so they give the same
pressure (to within the tolerance of the iterative ones). compare_solvers.py
checks that.
"""

from equations import pressure_equation_terms
from solver_splu import factorise, solve_splu
from solver_jacobi import solve_jacobi
from solver_gauss_seidel import solve_gauss_seidel
from solver_sor import solve_sor

SOLVERS = ("splu", "jacobi", "gauss-seidel", "sor")


def compute_pressure(plane, Ro, rotation_sign, setup, method="splu",
                     lu_data=None, use_z_terms=True, **solver_options):
    """Pressure for one plane.

    plane          dictionary from load_piv.py or load_dns.py
    Ro             Rossby number
    rotation_sign  settings.CLOCKWISE (experiments) or settings.ANTICLOCKWISE (DNS)
    setup          from poisson_setup.make_poisson_setup (once per grid)
    method         one of SOLVERS
    lu_data        for "splu": from solver_splu.factorise(setup) (once per grid);
                   made here if not given, but that repeats the slow step
    use_z_terms    keep term 4 (only has an effect for DNS data)
    solver_options passed on to the iterative solvers, e.g. tolerance=1e-8

    Returns (terms, info): terms is the dictionary of equations.py with the
    pressure added as terms["p"]; info describes the solve.
    """
    terms = pressure_equation_terms(plane, Ro, rotation_sign, use_z_terms)
    args = (terms["rhs"], terms["dp_dx"], terms["dp_dy"], setup)

    if method == "splu":
        if lu_data is None:
            lu_data = factorise(setup)
        p, info = solve_splu(*args, lu_data)
    elif method == "jacobi":
        p, info = solve_jacobi(*args, **solver_options)
    elif method == "gauss-seidel":
        p, info = solve_gauss_seidel(*args, **solver_options)
    elif method == "sor":
        p, info = solve_sor(*args, **solver_options)
    else:
        raise ValueError(f"unknown solver {method!r}; choose one of {SOLVERS}")

    terms["p"] = p
    return terms, info
