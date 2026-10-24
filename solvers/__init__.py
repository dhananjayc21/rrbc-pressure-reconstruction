# -*- coding: utf-8 -*-
# =============================================================================
# solvers
#
# The four ways of solving the pressure Poisson equation. All of them use the
# coefficients and ghost cells of coefficients.py and solve exactly the same
# equations:
#
#     direct.py         sparse LU (default)
#     jacobi.py         Jacobi iteration (as in the notebook)
#     gauss_seidel.py   red-black Gauss-Seidel iteration
#     sor.py            red-black SOR iteration
#
# solve_pressure.compute_pressure() picks one of them by name.
# =============================================================================
