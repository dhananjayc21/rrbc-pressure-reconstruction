# -*- coding: utf-8 -*-
# =============================================================================
# run_piv.py
#
# Pressure and vorticity of a PIV recording (a folder of PIVlab_XXXX.dat files).
#
# Change the settings below and run:   python run_piv.py
#
# For every STEP-th field it saves
#     <output>/pressure/P_PIVlab_XXXX.npy      pressure
#     <output>/vorticity/W_PIVlab_XXXX.npy     vertical vorticity
# on the full PIV grid, with 0 where nothing is computed (the outer rows),
# the same layout as before, so the correlation scripts read them unchanged.
# =============================================================================

import os
import glob
import time
import numpy as np

import settings
from read_piv import read_three_fields, frames_are_consecutive
from solve_pressure import prepare_poisson, compute_pressure
from plotting import plot_fields


# ==== SETTINGS ===============================================================
PIV_FOLDER = os.path.join(settings.DATA_FOLDER, "PIV Data sets", "Run9")   # the .dat files

Ro = 0.05            # Rossby number of the recording   ----> CHANGE IT FOR EVERY RECORDING

STEP = 20            # solve every STEP-th field (1 = all of them)
METHOD = "direct"    # "direct", "jacobi", "gauss-seidel" or "sor"
OUTPUT_FOLDER = os.path.join(settings.DATA_FOLDER, "PIV Data sets", "Rossby_" + format(Ro, ".2f"))
MAKE_FIGURE = True   # quick-look figure of the first field
# =============================================================================


files = sorted(glob.glob(os.path.join(PIV_FOLDER, "PIVlab_*.dat")))
print("PIV folder:", PIV_FOLDER)
print("Number of files:", len(files))
if len(files) < 3:
    print("At least 3 PIVlab_*.dat files are needed")
    raise SystemExit

print("Rossby number =", Ro)
print("Ekman number =", settings.ekman_number(Ro))
print("Rotation rate =", settings.rotation_rate(Ro), "rad/s")

os.makedirs(os.path.join(OUTPUT_FOLDER, "pressure"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_FOLDER, "vorticity"), exist_ok=True)

dt = settings.dt_fields / settings.T_ndim       # time between the fields (dimensionless)
already_read = {}                               # every .dat file is read only once
poisson = None                                  # prepared at the first field
start_time = time.time()
count = 0

# t needs a field before and after it: t = 1 ... len(files) - 2
for t in range(1, len(files) - 1, STEP):

    if not frames_are_consecutive(files, t):
        print("  skipping", os.path.basename(files[t]), ": its neighbours are not the next frames")
        continue

    u, v, grid_x, grid_y, n_filled = read_three_fields(files, t, already_read)

    if poisson is None:                         # the grid is the same for the whole recording
        poisson = prepare_poisson(grid_x, grid_y, METHOD)

    result = compute_pressure(u, v, dt, grid_x, grid_y, Ro, settings.CLOCKWISE, poisson, METHOD)
    X, Vor, iterations, e, uniform_source = result

    # save, with 0 instead of NaN where nothing is computed
    name = os.path.splitext(os.path.basename(files[t]))[0]
    np.save(os.path.join(OUTPUT_FOLDER, "pressure", "P_" + name + ".npy"), np.nan_to_num(X, nan=0.0))
    np.save(os.path.join(OUTPUT_FOLDER, "vorticity", "W_" + name + ".npy"), np.nan_to_num(Vor, nan=0.0))
    count = count + 1

    if MAKE_FIGURE and count == 1:
        plot_fields([X, Vor], ["pressure p", "vorticity"], grid_x, grid_y,
                    os.path.join(OUTPUT_FOLDER, "quicklook_" + name + ".png"),
                    u=u[1], v=v[1], title=name)

    if count == 1 or count % 50 == 0:
        print("  ", name, ": iterations", iterations, ", residual", format(e, ".1e"),
              ", filled vectors", n_filled, ",", round(time.time() - start_time), "s so far")

print("Done:", count, "fields written to", OUTPUT_FOLDER)
