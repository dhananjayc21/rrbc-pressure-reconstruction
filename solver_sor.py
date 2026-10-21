# -*- coding: utf-8 -*-
"""
solver_sor.py
=============

SOR (successive over-relaxation), vectorised with the red-black ordering of
solver_gauss_seidel.py.

SOR is Gauss-Seidel that takes a bigger step: Gauss-Seidel moves every point
to its new value p_new; SOR moves it further in the same direction,

    p = (1 - omega) * p_old + omega * p_new,     1 < omega < 2.

omega = 1 is exactly Gauss-Seidel. With a good omega the number of iterations
drops from about 10^5 to about 10^3 on the PIV grid.

The best omega depends on the grid. For the Poisson equation it is

    omega_best = 2 / (1 + sqrt(1 - rho^2)),

where rho is how much one Jacobi iteration reduces the slowest error pattern
(the smoothest wave that fits in the domain). best_omega() estimates rho from
the number of points and the stencil weights; on the stretched DNS grid this
is only an estimate, but a good one.
"""

import time
import numpy as np

from poisson_setup import (make_consistent, new_value_from_neighbours, set_ghost_cells,
                           remove_mean, residual, start_field, finish_field)


def best_omega(setup):
    """Estimate of the optimal SOR factor for this grid."""
    ni, nj = setup["cP"].shape
    wx = np.mean(setup["cW"] + setup["cE"])   # stencil weight of the x-neighbours
    wy = np.mean(setup["cS"] + setup["cN"])   # and of the y-neighbours
    rho = (wx * np.cos(np.pi / ni) + wy * np.cos(np.pi / nj)) / (wx + wy)
    return 2 / (1 + np.sqrt(1 - rho ** 2))


def solve_sor(rhs, dp_dx, dp_dy, setup, omega=None, tolerance=1e-9,
              max_iterations=100000, check_every=10, verbose=False):
    """Pressure on the full grid (NaN outside the solved region + ghost ring).

    omega=None uses best_omega(setup).
    """
    t0 = time.perf_counter()
    s = setup
    if omega is None:
        omega = best_omega(s)
    rhs, uniform = make_consistent(rhs, dp_dx, dp_dy, s)   # see poisson_setup.py
    p = start_field(s)
    set_ghost_cells(p, dp_dx, dp_dy, s)
    inside = p[s["inside"]]          # a view: changing "inside" changes p
    rel_res, it = np.inf, 0

    for it in range(1, max_iterations + 1):
        for colour in ("red", "black"):
            p_new = new_value_from_neighbours(p, rhs, s)
            mask = s[colour]
            inside[mask] = (1 - omega) * inside[mask] + omega * p_new[mask]
            set_ghost_cells(p, dp_dx, dp_dy, s)
        remove_mean(p, dp_dx, dp_dy, s)

        if it % check_every == 0:
            rel_res = residual(p, rhs, s)
            if verbose and it % (100 * check_every) == 0:
                print(f"    SOR iteration {it}: residual {rel_res:.2e}")
            if rel_res < tolerance:
                break

    info = {"method": f"sor (omega {omega:.4f})", "iterations": it,
            "time": time.perf_counter() - t0, "residual": rel_res,
            "uniform_source": uniform, "converged": rel_res < tolerance}
    return finish_field(p, s), info
