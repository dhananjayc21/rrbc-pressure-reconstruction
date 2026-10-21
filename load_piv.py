# -*- coding: utf-8 -*-
"""
load_piv.py
===========

Reading PIVlab TECPLOT exports (PIVlab_XXXX.dat) and turning them into the
"plane" dictionary that equations.py works with:

    plane["u"], plane["v"]   velocity at three times, shape (3, nx, ny):
                             index 0 = t-1, 1 = t, 2 = t+1
    plane["dt"]              time between the three fields (non-dimensional)
    plane["grid_x"], plane["grid_y"]   grid dictionaries (grid.py)
    plane["z_terms"]         None: planar PIV has no w or z-derivatives
    plane["name"]            the file name of the middle field
    plane["n_filled"]        how many flagged vectors were filled in

Units: lengths in H, velocities in the free-fall velocity U_REF, time in
T_REF = H / U_REF (see settings.py).
"""

import os
import re
import numpy as np
import pandas as pd
from scipy.interpolate import griddata

import settings
from grid import uniform_grid


def read_pivlab_dat(path):
    """One PIVlab file -> x_px (nx,), y_px (ny,), U (nx, ny), V (nx, ny).

    U and V are in px/frame. The grid is taken from the distinct x and y values
    in the file (not hard-coded), so other PIVlab settings also work. Vectors
    that PIVlab flagged (last column = 1) are returned as NaN.
    """
    table = pd.read_csv(path, sep=r"\s+", skiprows=6, header=None,
                        names=["x", "y", "u", "v", "flag"])
    x_px = np.unique(table["x"].to_numpy())
    y_px = np.unique(table["y"].to_numpy())
    i = np.searchsorted(x_px, table["x"].to_numpy())     # column index of every row
    j = np.searchsorted(y_px, table["y"].to_numpy())

    U = np.full((x_px.size, y_px.size), np.nan)
    V = np.full((x_px.size, y_px.size), np.nan)
    good = table["flag"].to_numpy() == 0
    U[i[good], j[good]] = table["u"].to_numpy()[good]
    V[i[good], j[good]] = table["v"].to_numpy()[good]
    return x_px, y_px, U, V


def fill_missing(F):
    """Fill NaN entries from their neighbours (linear interpolation, or the
    nearest value where that is not possible). Returns (filled field, count)."""
    missing = ~np.isfinite(F)
    count = int(missing.sum())
    if count == 0:
        return F, 0
    I, J = np.indices(F.shape)
    points = np.column_stack([I[~missing], J[~missing]])
    values = F[~missing]
    G = F.copy()
    G[missing] = griddata(points, values, (I[missing], J[missing]), method="linear")
    still = ~np.isfinite(G)
    if still.any():
        G[still] = griddata(points, values, (I[still], J[still]), method="nearest")
    return G, count


def frame_number(path):
    """The number at the end of a file name, e.g. PIVlab_0200.dat -> 200."""
    match = re.search(r"(\d+)\D*$", os.path.basename(path))
    return int(match.group(1)) if match else None


def has_consecutive_neighbours(files, t):
    """True if files[t-1], files[t], files[t+1] are consecutive frames.

    The time derivative needs the fields exactly one frame before and after.
    A folder with gaps (missing frames, or several short excerpts) would
    otherwise silently give a wrong d/dt.
    """
    n0, n1, n2 = (frame_number(f) for f in files[t - 1:t + 2])
    if None in (n0, n1, n2):
        return True                    # no numbers in the names: cannot check
    return n1 - n0 == 1 and n2 - n1 == 1


def load_piv_snapshot(files, t, cache=None):
    """The plane dictionary for the field files[t], using files[t-1] and files[t+1].

    cache: pass the same empty dictionary {} for every call of a run. Each file
    is then read only once, although three neighbouring fields use it (reading
    is by far the slowest part of the PIV run).
    """
    if not 1 <= t <= len(files) - 2:
        raise ValueError("t needs a field on either side (1 <= t <= number of files - 2)")

    velocity_scale = settings.M_PER_PX / settings.IMAGE_DT / settings.U_REF   # px/frame -> U_REF
    u_list, v_list, n_filled = [], [], 0
    for path in files[t - 1:t + 2]:
        if cache is not None and path in cache:
            x_px, y_px, U, V, n = cache[path]
        else:
            x_px, y_px, U, V = read_pivlab_dat(path)
            U, nu = fill_missing(U)
            V, nv = fill_missing(V)
            n = max(nu, nv)
            if cache is not None:
                cache[path] = (x_px, y_px, U, V, n)
        u_list.append(U * velocity_scale)
        v_list.append(V * velocity_scale)
        n_filled += n

    if cache is not None:              # keep only the files the next field can use
        for old in [k for k in cache if k not in files[t - 1:t + 3]]:
            del cache[old]

    to_H = settings.M_PER_PX / settings.H                                  # px -> x/H
    return {"u": np.stack(u_list),
            "v": np.stack(v_list),
            "dt": settings.SNAPSHOT_DT / settings.T_REF,
            "grid_x": uniform_grid(x_px * to_H),
            "grid_y": uniform_grid(y_px * to_H),
            "z_terms": None,
            "p_dns": None,
            "name": os.path.splitext(os.path.basename(files[t]))[0],
            "n_filled": n_filled}
