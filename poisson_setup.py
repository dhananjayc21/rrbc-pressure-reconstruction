# -*- coding: utf-8 -*-
"""
poisson_setup.py
================

Everything the four solvers (SPLU, Jacobi, Gauss-Seidel, SOR) have in common:
the five-point stencil, the ghost-cell boundary condition, making the data
consistent, and the residual.

The discrete Poisson equation at a solved point (i, j) is

    cW p[i-1,j] + cE p[i+1,j] + cS p[i,j-1] + cN p[i,j+1] + cP p[i,j] = rhs[i,j]

(W/E = west/east = i-1/i+1, S/N = south/north = j-1/j+1). On the uniform PIV
grid cW = cE = 1/dx^2, cS = cN = 1/dy^2 and cP = -2/dx^2 - 2/dy^2. On the
stretched DNS grid the weights come from the chain rule (see grid.py).

The margin
----------
The pressure is solved at the points  margin ... n-margin-1  in each direction.
The ring of points just outside them holds the GHOST values, which impose the
Neumann boundary condition with a central difference about the first solved
point; on the west side, with g = margin - 1:

    (p[g+2] - p[g]) / (2 dxi) = x_xi[g+1] * dp/dx[g+1]
    ->  p[g] = p[g+2] - 2 dxi x_xi[g+1] dp/dx[g+1]

and the same on the other three sides. margin = 3 is what both original codes
used.

Consistent data
---------------
With Neumann conditions on every side the pressure is only defined up to a
constant, and a solution exists only if the boundary values and the
right-hand side fit together (the discrete form of the divergence theorem).
Measured data never fit exactly: no pressure field satisfies every equation;
the best possible field leaves one small uniform leftover mu at every point.
make_consistent() removes that mu from the right-hand side before solving, so
that every solver solves equations that DO have a solution. Without this the
iterative solvers can never reach a small residual (they stall), and the
direct solver depends on round-off. mu is reported as the "uniform source";
it measures the data, not the solver.

mu is computed exactly from the left null vector l of the system matrix A
(the vector with A^T l = 0):  mu = (l . b) / (l . 1). l depends only on the
grid, so it is found once, in make_poisson_setup().
"""

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve


def make_poisson_setup(gx, gy, margin=3):
    """Everything about the grid that the solvers need, in one dictionary.

    It depends only on the grid, so make it once and use it for every
    snapshot of a data set (it takes a second or two on the DNS grid).
    """
    if margin < 2:
        raise ValueError("margin must be at least 2")
    nx, ny = gx["coords"].size, gy["coords"].size
    i0, i1 = margin, nx - margin            # solved points: i0 ... i1-1
    j0, j1 = margin, ny - margin            #                j0 ... j1-1

    # chain-rule factors at the solved points: d2p/dx2 = ax2 * p_xixi + ax1 * p_xi
    ax2 = 1 / gx["x_xi"][i0:i1] ** 2
    ax1 = -gx["x_xixi"][i0:i1] / gx["x_xi"][i0:i1] ** 3
    ay2 = 1 / gy["x_xi"][j0:j1] ** 2
    ay1 = -gy["x_xixi"][j0:j1] / gy["x_xi"][j0:j1] ** 3
    hx, hy = gx["d_xi"], gy["d_xi"]

    ones = np.ones((i1 - i0, j1 - j0))
    setup = {
        "gx": gx, "gy": gy, "margin": margin, "shape": (nx, ny),
        "i0": i0, "i1": i1, "j0": j0, "j1": j1,
        "cW": (ax2 / hx ** 2 - ax1 / (2 * hx))[:, None] * ones,
        "cE": (ax2 / hx ** 2 + ax1 / (2 * hx))[:, None] * ones,
        "cS": (ay2 / hy ** 2 - ay1 / (2 * hy))[None, :] * ones,
        "cN": (ay2 / hy ** 2 + ay1 / (2 * hy))[None, :] * ones,
        "cP": -2 * (ax2[:, None] / hx ** 2 + ay2[None, :] / hy ** 2) * ones,
        # index ranges of the solved points and of their four neighbours
        "inside": (slice(i0, i1), slice(j0, j1)),
        "west": (slice(i0 - 1, i1 - 1), slice(j0, j1)),
        "east": (slice(i0 + 1, i1 + 1), slice(j0, j1)),
        "south": (slice(i0, i1), slice(j0 - 1, j1 - 1)),
        "north": (slice(i0, i1), slice(j0 + 1, j1 + 1)),
    }

    # red-black colouring (like a chess board) of the solved points, used by
    # Gauss-Seidel and SOR: a red point only has black neighbours and the
    # other way round, so all points of one colour can be updated at once
    I, J = np.meshgrid(np.arange(i0, i1), np.arange(j0, j1), indexing="ij")
    setup["red"] = (I + J) % 2 == 0
    setup["black"] = ~setup["red"]

    # the point that the direct solver fixes at p = 0 (the centre), and the
    # left null vector of the matrix (for make_consistent)
    ni, nj = ones.shape
    setup["fixed_point"] = (ni // 2) * nj + nj // 2
    setup["left_null"] = left_null_vector(setup)
    return setup


# -----------------------------------------------------------------------------
# The system matrix (ghost values eliminated)
# -----------------------------------------------------------------------------
def build_matrix(setup):
    """The sparse matrix A of all solved points, ghost values eliminated.

    A ghost value is p[ghost] = p[point on the other side] - (known), so the
    weight of a ghost neighbour is added to the point on the other side, and
    the known part goes to the right-hand side (boundary_correction).
    """
    s = setup
    ni, nj = s["cP"].shape
    I, J = np.meshgrid(np.arange(ni), np.arange(nj), indexing="ij")

    def number(i, j):                     # point (i, j) -> row number in A
        return (i * nj + j).ravel()

    west = np.where(I == 0, I + 1, I - 1)
    east = np.where(I == ni - 1, I - 1, I + 1)
    south = np.where(J == 0, J + 1, J - 1)
    north = np.where(J == nj - 1, J - 1, J + 1)

    rows = np.tile(number(I, J), 5)
    cols = np.concatenate([number(I, J), number(west, J), number(east, J),
                           number(I, south), number(I, north)])
    vals = np.concatenate([s["cP"].ravel(), s["cW"].ravel(), s["cE"].ravel(),
                           s["cS"].ravel(), s["cN"].ravel()])
    # entries with the same (row, col) are added together by coo_matrix
    return coo_matrix((vals, (rows, cols)), shape=(ni * nj, ni * nj))


def boundary_correction(dp_dx, dp_dy, setup):
    """The known part of the ghost relations, as it appears on the right-hand
    side of A p = b (shape of the solved points)."""
    s = setup
    gx, gy = s["gx"], s["gy"]
    i0, i1, j0, j1 = s["i0"], s["i1"], s["j0"], s["j1"]
    bc = np.zeros(s["cP"].shape)
    bc[0, :] += s["cW"][0, :] * 2 * gx["d_xi"] * gx["x_xi"][i0] * dp_dx[i0, j0:j1]
    bc[-1, :] -= s["cE"][-1, :] * 2 * gx["d_xi"] * gx["x_xi"][i1 - 1] * dp_dx[i1 - 1, j0:j1]
    bc[:, 0] += s["cS"][:, 0] * 2 * gy["d_xi"] * gy["x_xi"][j0] * dp_dy[i0:i1, j0]
    bc[:, -1] -= s["cN"][:, -1] * 2 * gy["d_xi"] * gy["x_xi"][j1 - 1] * dp_dy[i0:i1, j1 - 1]
    return bc


def matrix_with_fixed_point(setup):
    """A with the row of the fixed point replaced by 'c * p_k = 0'.

    A itself has no inverse (adding a constant to p changes nothing); fixing
    one point removes that freedom. c keeps the row the same size as the rest.
    """
    A = build_matrix(setup)
    k = setup["fixed_point"]
    c = float(np.mean(np.abs(setup["cP"])))
    keep = A.row != k
    return coo_matrix((np.append(A.data[keep], c),
                       (np.append(A.row[keep], k), np.append(A.col[keep], k))),
                      shape=A.shape).tocsc(), A.tocsr()


def left_null_vector(setup):
    """The vector l with A^T l = 0, normalised so that l = 1 at the fixed point.

    With row k (the fixed point) of A called a_k:  A^T l = 0  means
    sum over i != k of l_i a_i = -a_k. Exactly that is solved by
    A_fixed^T y = -a_k (y_k comes out 0); then l = y with l_k = 1.
    """
    A_fixed, A = matrix_with_fixed_point(setup)
    k = setup["fixed_point"]
    row_k = A[[k], :].toarray().ravel()
    l = spsolve(A_fixed.T.tocsc(), -row_k)
    l[k] = 1.0
    return l


# -----------------------------------------------------------------------------
# Used by every solver
# -----------------------------------------------------------------------------
def make_consistent(rhs, dp_dx, dp_dy, setup):
    """Remove the uniform leftover mu from the right-hand side (see the top).

    Returns (rhs with mu removed at the solved points, uniform source), where
    the uniform source is |mu| / RMS(rhs): the size of the mismatch of the data.
    """
    s = setup
    b = (rhs[s["inside"]] + boundary_correction(dp_dx, dp_dy, s)).ravel()
    if not np.all(np.isfinite(b)):
        raise ValueError("NaN in the right-hand side or boundary values; "
                         "check the velocity data near the edges")
    l = s["left_null"]
    mu = (l @ b) / l.sum()
    rhs_ok = rhs.copy()
    rhs_ok[s["inside"]] -= mu
    return rhs_ok, float(abs(mu) / np.sqrt(np.mean(rhs[s["inside"]] ** 2)))


def set_ghost_cells(p, dp_dx, dp_dy, setup):
    """Fill the ghost ring of p from the Neumann condition (changes p in place).

    First the two x-ghost rows, then the two y-ghost columns, which also give
    the four corners (the same order as in the original notebooks).
    """
    gx, gy = setup["gx"], setup["gy"]
    i0, i1, j0, j1 = setup["i0"], setup["i1"], setup["j0"], setup["j1"]
    js = slice(j0, j1)
    p[i0 - 1, js] = p[i0 + 1, js] - 2 * gx["d_xi"] * gx["x_xi"][i0] * dp_dx[i0, js]
    p[i1, js] = p[i1 - 2, js] + 2 * gx["d_xi"] * gx["x_xi"][i1 - 1] * dp_dx[i1 - 1, js]
    iz = slice(i0 - 1, i1 + 1)
    p[iz, j0 - 1] = p[iz, j0 + 1] - 2 * gy["d_xi"] * gy["x_xi"][j0] * dp_dy[iz, j0]
    p[iz, j1] = p[iz, j1 - 2] + 2 * gy["d_xi"] * gy["x_xi"][j1 - 1] * dp_dy[iz, j1 - 1]


def new_value_from_neighbours(p, rhs, setup):
    """The value every solved point would get from its four neighbours:

        p[i,j] = (rhs - cW p_west - cE p_east - cS p_south - cN p_north) / cP

    This one line is the heart of Jacobi, Gauss-Seidel and SOR.
    """
    s = setup
    return (rhs[s["inside"]]
            - s["cW"] * p[s["west"]] - s["cE"] * p[s["east"]]
            - s["cS"] * p[s["south"]] - s["cN"] * p[s["north"]]) / s["cP"]


def remove_mean(p, dp_dx, dp_dy, setup):
    """Subtract the mean of the solved points (the free constant) and refresh
    the ghost cells, which move with it."""
    p[setup["inside"]] -= p[setup["inside"]].mean()
    set_ghost_cells(p, dp_dx, dp_dy, setup)


def residual(p, rhs, setup):
    """How well p satisfies the discrete equations: max |A p - rhs| / max |rhs|
    over the solved points (p must have its ghost cells, rhs must be the
    consistent one). About 1e-12 when fully converged."""
    s = setup
    r = (s["cP"] * p[s["inside"]]
         + s["cW"] * p[s["west"]] + s["cE"] * p[s["east"]]
         + s["cS"] * p[s["south"]] + s["cN"] * p[s["north"]]
         - rhs[s["inside"]])
    return float(np.max(np.abs(r)) / np.max(np.abs(rhs[s["inside"]])))


def start_field(setup):
    """Starting guess p = 0 for the iterative solvers (full grid)."""
    return np.zeros(setup["shape"])


def finish_field(p, setup):
    """Points outside the solved region and its ghost ring are set to NaN:
    the pressure is not computed there."""
    out = np.full(setup["shape"], np.nan)
    ring = (slice(setup["i0"] - 1, setup["i1"] + 1), slice(setup["j0"] - 1, setup["j1"] + 1))
    out[ring] = p[ring]
    return out
