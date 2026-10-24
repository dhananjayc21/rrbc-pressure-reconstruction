# -*- coding: utf-8 -*-
"""
solvers/solver_splu.py
======================

Direct solver: sparse LU factorisation (scipy.sparse.linalg.splu, the SuperLU
library), as in the DNS notebook.

How it works
------------
1. The discrete equations of all solved points are written as one linear
   system A p = b, with the ghost values eliminated
   (poisson_setup.build_matrix and boundary_correction).
2. A is factorised once, A = L U (lower x upper triangular). This is the slow
   step (about a second on the DNS grid) and depends only on the grid.
3. Every snapshot then only needs a forward and a backward substitution with
   L and U: about 0.01 s for a PIV field.

So: call factorise() ONCE per grid, then solve_splu() for every snapshot.

A on its own has no inverse (adding a constant to p changes nothing), so one
point, the centre, is fixed at p = 0 (poisson_setup.matrix_with_fixed_point).
Because the data are first made consistent (poisson_setup.make_consistent),
fixing that point does not distort the solution; the mean is removed at the
end.
"""

import time
from scipy.sparse.linalg import splu

from poisson_setup import (matrix_with_fixed_point, boundary_correction, make_consistent,
                           set_ghost_cells, residual, finish_field)
import numpy as np


def factorise(setup):
    """LU factorisation of the matrix, done once per grid."""
    A_fixed, _ = matrix_with_fixed_point(setup)
    return splu(A_fixed, permc_spec="MMD_AT_PLUS_A")   # same ordering as the notebook


def solve_splu(rhs, dp_dx, dp_dy, setup, lu):
    """Pressure on the full grid (NaN outside the solved region + ghost ring).

    lu: from factorise(setup).
    """
    t0 = time.perf_counter()
    s = setup
    rhs_ok, uniform = make_consistent(rhs, dp_dx, dp_dy, s)
    b = (rhs_ok[s["inside"]] + boundary_correction(dp_dx, dp_dy, s)).ravel()
    b[s["fixed_point"]] = 0.0                # the fixed point: p = 0 there
    x = lu.solve(b)
    x -= x.mean()

    p = np.zeros(s["shape"])
    p[s["inside"]] = x.reshape(s["cP"].shape)
    set_ghost_cells(p, dp_dx, dp_dy, s)
    info = {"method": "splu", "iterations": 1, "time": time.perf_counter() - t0,
            "residual": residual(p, rhs_ok, s), "uniform_source": uniform, "converged": True}
    return finish_field(p, s), info
