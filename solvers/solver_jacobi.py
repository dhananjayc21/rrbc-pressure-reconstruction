# -*- coding: utf-8 -*-
"""
solvers/solver_jacobi.py
========================

Jacobi iteration (the method of Pressure_Poisson_PIV.ipynb), fully vectorised
with numpy: in every iteration ALL points get a new value computed from the
OLD values of their four neighbours,

    p_new[i,j] = (rhs - cW p[i-1,j] - cE p[i+1,j] - cS p[i,j-1] - cN p[i,j+1]) / cP

then the ghost cells are refreshed from the Neumann condition and the mean is
removed (the pressure is only defined up to a constant, and this fixes it).
Before the first iteration the data are made consistent
(poisson_setup.make_consistent); without that the iteration can never reach
a small residual.

weight
------
weight = 1 is the plain Jacobi of the notebook. Plain Jacobi has one pattern
it never removes: a chess-board pattern (+a, -a, +a, ...) just flips its sign
every iteration, so with pure Neumann conditions the iteration never fully
settles. weight < 1 ("weighted Jacobi") mixes in some of the old value,

    p = (1 - weight) * p_old + weight * p_new,

which damps that pattern and costs only a little speed. The default 0.9 does
that; use weight=1 to reproduce the notebook.

Jacobi is simple but slow: on the PIV grid it needs of the order of 10^5
iterations. Use it to understand or to cross-check, not for production runs.
"""

import time
import numpy as np

from poisson_setup import (make_consistent, new_value_from_neighbours, set_ghost_cells,
                           remove_mean, residual, start_field, finish_field)


def solve_jacobi(rhs, dp_dx, dp_dy, setup, weight=0.9, tolerance=1e-9,
                 max_iterations=500000, check_every=100, verbose=False):
    """Pressure on the full grid (NaN outside the solved region + ghost ring).

    Stops when the relative residual (poisson_setup.residual) is below
    tolerance, or after max_iterations.
    """
    t0 = time.perf_counter()
    s = setup
    rhs, uniform = make_consistent(rhs, dp_dx, dp_dy, s)   # see poisson_setup.py
    p = start_field(s)
    set_ghost_cells(p, dp_dx, dp_dy, s)
    rel_res, it = np.inf, 0

    for it in range(1, max_iterations + 1):
        p_new = new_value_from_neighbours(p, rhs, s)      # uses the OLD values only
        p[s["inside"]] = (1 - weight) * p[s["inside"]] + weight * p_new
        remove_mean(p, dp_dx, dp_dy, s)                  # also refreshes the ghosts

        if it % check_every == 0:
            rel_res = residual(p, rhs, s)
            if verbose and it % (100 * check_every) == 0:
                print(f"    Jacobi iteration {it}: residual {rel_res:.2e}")
            if rel_res < tolerance:
                break

    info = {"method": f"jacobi (weight {weight})", "iterations": it,
            "time": time.perf_counter() - t0, "residual": rel_res,
            "uniform_source": uniform, "converged": rel_res < tolerance}
    return finish_field(p, s), info
