"""
solvers
=======

The four solvers of the discrete pressure Poisson equation. All of them use
the stencil, ghost cells and consistency fix of poisson_setup.py and solve
exactly the same equations:

    solver_splu.py          direct sparse LU (default)
    solver_jacobi.py        weighted Jacobi iteration
    solver_gauss_seidel.py  red-black Gauss-Seidel iteration
    solver_sor.py           red-black SOR iteration

solve_pressure.compute_pressure() picks one of them by name.
"""
