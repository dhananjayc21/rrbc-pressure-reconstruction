# -*- coding: utf-8 -*-
"""
solver_gauss_seidel.py
======================

Gauss-Seidel iteration, vectorised with the red-black ordering.

Plain Gauss-Seidel goes through the points one by one and always uses the
NEWEST values of the neighbours. That needs a slow Python loop over every
point. The red-black version gives exactly the same kind of method, but fast:

    colour the solved points like a chess board, red where i + j is even and
    black where it is odd. Every red point has only black neighbours (and the
    other way round), so

      1. all RED points are updated at once, from their black neighbours;
      2. all BLACK points are updated at once, from the red values just made.

Each half-step is one numpy expression, like Jacobi, but the second half
already uses the new red values. That makes it about twice as fast as Jacobi
(per iteration), and it does not have Jacobi's chess-board problem.

After each colour the ghost cells are refreshed (the boundary points need the
newest values), and after each full iteration the mean is removed (the free
constant, see solver_jacobi.py).
"""

import time
import numpy as np

from poisson_setup import (make_consistent, new_value_from_neighbours, set_ghost_cells,
                           remove_mean, residual, start_field, finish_field)


def solve_gauss_seidel(rhs, dp_dx, dp_dy, setup, tolerance=1e-9,
                       max_iterations=300000, check_every=100, verbose=False):
    """Pressure on the full grid (NaN outside the solved region + ghost ring)."""
    t0 = time.perf_counter()
    s = setup
    rhs, uniform = make_consistent(rhs, dp_dx, dp_dy, s)   # see poisson_setup.py
    p = start_field(s)
    set_ghost_cells(p, dp_dx, dp_dy, s)
    inside = p[s["inside"]]          # a view: changing "inside" changes p
    rel_res, it = np.inf, 0

    for it in range(1, max_iterations + 1):
        for colour in ("red", "black"):
            p_new = new_value_from_neighbours(p, rhs, s)
            mask = s[colour]
            inside[mask] = p_new[mask]                # only this colour changes
            set_ghost_cells(p, dp_dx, dp_dy, s)
        remove_mean(p, dp_dx, dp_dy, s)

        if it % check_every == 0:
            rel_res = residual(p, rhs, s)
            if verbose and it % (100 * check_every) == 0:
                print(f"    Gauss-Seidel iteration {it}: residual {rel_res:.2e}")
            if rel_res < tolerance:
                break

    info = {"method": "gauss-seidel (red-black)", "iterations": it,
            "time": time.perf_counter() - t0, "residual": rel_res,
            "uniform_source": uniform, "converged": rel_res < tolerance}
    return finish_field(p, s), info
