# -*- coding: utf-8 -*-
# =============================================================================
# read_piv.py
#
# Reading the PIVlab files (PIVlab_XXXX.dat) and making the velocity
# dimensionless, as in the PIV notebook.
#
# The main function is read_three_fields(files, t): it reads the field t and
# the fields t-1 and t+1 (needed for d/dt) and returns
#
#     u, v      velocity, shape (3, Nx, Ny): index 0 = t-1, 1 = t, 2 = t+1
#     grid_x    grid dictionaries (grid.py)
#     grid_y
#
# Units: lengths in H, velocity in U_ndim, time in T_ndim (see settings.py).
# =============================================================================

import os
import numpy as np
import pandas as pd
from scipy.interpolate import griddata

import settings
from grid import uniform_grid


def read_piv_file(path):
    # One PIVlab file -> x and y of the vectors (px) and U, V (px/frame).
    # Vectors flagged by PIVlab (last column = 1) are set to NaN.
    df = pd.read_csv(path, sep=r"\s+", skiprows=6, names=["x", "y", "u", "v", "flag"])

    x_px = np.unique(df["x"].to_numpy())     # the different x positions
    y_px = np.unique(df["y"].to_numpy())     # the different y positions
    Nx = len(x_px)
    Ny = len(y_px)

    # index of every vector in the grid
    ix = np.searchsorted(x_px, df["x"].to_numpy())
    iy = np.searchsorted(y_px, df["y"].to_numpy())

    U = np.full((Nx, Ny), np.nan)
    V = np.full((Nx, Ny), np.nan)
    good = df["flag"].to_numpy() == 0
    U[ix[good], iy[good]] = df["u"].to_numpy()[good]
    V[ix[good], iy[good]] = df["v"].to_numpy()[good]
    return x_px, y_px, U, V


def fill_missing(F):
    # Fill the NaN values from their neighbours (linear interpolation; where
    # that is not possible, the nearest value). Returns the field and how many
    # values were filled.
    missing = np.isnan(F)
    count = int(np.sum(missing))
    if count == 0:
        return F, 0

    I, J = np.indices(F.shape)
    known_points = np.column_stack([I[~missing], J[~missing]])
    known_values = F[~missing]

    G = F.copy()
    G[missing] = griddata(known_points, known_values, (I[missing], J[missing]), method="linear")
    still_missing = np.isnan(G)
    if np.any(still_missing):
        G[still_missing] = griddata(known_points, known_values,
                                    (I[still_missing], J[still_missing]), method="nearest")
    return G, count


def frame_number(path):
    # PIVlab_0200.dat -> 200
    name = os.path.splitext(os.path.basename(path))[0]
    digits = ""
    for character in name:
        if character.isdigit():
            digits = digits + character
    if digits == "":
        return None
    return int(digits)


def frames_are_consecutive(files, t):
    # True if files[t-1], files[t], files[t+1] are three frames in a row.
    # d/dt needs exactly the frame before and the frame after.
    n0 = frame_number(files[t - 1])
    n1 = frame_number(files[t])
    n2 = frame_number(files[t + 1])
    if n0 is None or n1 is None or n2 is None:
        return True                   # no numbers in the names: cannot check
    if n1 - n0 == 1 and n2 - n1 == 1:
        return True
    else:
        return False


def read_three_fields(files, t, already_read=None):
    # Velocity of the fields t-1, t, t+1 (dimensionless) and the grid.
    #
    # already_read: give the same empty dictionary {} for every call of a run.
    # Then every file is read only once, although three neighbouring fields use
    # it (reading the .dat files is the slowest part of a PIV run).

    # px/frame -> m/s -> dimensionless
    to_velocity = settings.px_to_m / settings.dt_frame / settings.U_ndim

    u = []
    v = []
    n_filled = 0
    for k in range(t - 1, t + 2):
        path = files[k]
        if already_read is not None and path in already_read:
            x_px, y_px, U, V, n = already_read[path]
        else:
            x_px, y_px, U, V = read_piv_file(path)
            U, n_u = fill_missing(U)
            V, n_v = fill_missing(V)
            n = max(n_u, n_v)
            if already_read is not None:
                already_read[path] = (x_px, y_px, U, V, n)
        u.append(U * to_velocity)
        v.append(V * to_velocity)
        n_filled = n_filled + n

    # forget the files that the next field will not need any more
    if already_read is not None:
        for path in list(already_read.keys()):
            if path not in files[t - 1:t + 3]:
                del already_read[path]

    u = np.array(u)
    v = np.array(v)

    # px -> x/H
    to_H = settings.px_to_m / settings.H
    grid_x = uniform_grid(x_px * to_H)
    grid_y = uniform_grid(y_px * to_H)
    return u, v, grid_x, grid_y, n_filled
