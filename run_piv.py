# -*- coding: utf-8 -*-
"""
run_piv.py
==========

Pressure and vorticity for a PIV recording (a folder of PIVlab_XXXX.dat files).

How to use: change the USER SETTINGS below and run
    python run_piv.py

For every STEP-th field it saves
    <output>/pressure/P_PIVlab_XXXX.npy     pressure
    <output>/vorticity/W_PIVlab_XXXX.npy    vertical vorticity
on the full PIV grid, with 0 where nothing is computed (the outer rows), the
same layout as piv_fields.py, so the correlation scripts read them unchanged.
A quick-look figure of the first field is saved as well.
"""

import os
import glob
import time
import numpy as np

import settings
from load_piv import load_piv_snapshot, has_consecutive_neighbours
from poisson_setup import make_poisson_setup
from solver_splu import factorise
from solve_pressure import compute_pressure
from plotting import plot_fields


# ==== USER SETTINGS ==========================================================
PIV_FOLDER = os.path.join(settings.DATA_FOLDER, "PIV Data sets", "Run9")  # the .dat files
RO = 0.05                  # Rossby number of the recording  ----> CHANGE FOR EVERY RECORDING
STEP = 20                  # solve every STEP-th field (1 = all). d/dt always uses the
                           # fields directly before and after (0.1 s apart)
FIRST, LAST = 1, None      # first and last field index (None = up to the end)
MARGIN = 3                 # rows left out at each edge (see poisson_setup.py)
SOLVER = "splu"            # "splu", "jacobi", "gauss-seidel" or "sor"
OUTPUT_FOLDER = None       # None = a folder "Rossby_<Ro>" next to PIV_FOLDER
MAKE_FIGURE = True         # quick-look figure of the first field
# =============================================================================


def save_fields(pressure, vorticity, output_folder, name):
    """Save in the layout of piv_fields.py: full grid, 0 instead of NaN."""
    for sub, prefix, field in (("pressure", "P", pressure), ("vorticity", "W", vorticity)):
        folder = os.path.join(output_folder, sub)
        os.makedirs(folder, exist_ok=True)
        np.save(os.path.join(folder, f"{prefix}_{name}.npy"), np.nan_to_num(field, nan=0.0))


def run_piv(piv_folder, Ro, step=20, first=1, last=None, margin=3, solver="splu",
            output_folder=None, make_figure=True):
    files = sorted(glob.glob(os.path.join(piv_folder, "PIVlab_*.dat")))
    if len(files) < 3:
        raise SystemExit(f"need at least 3 PIVlab_*.dat files in {piv_folder}")
    if output_folder is None:
        output_folder = os.path.join(os.path.dirname(os.path.abspath(piv_folder)), f"Rossby_{Ro:.2f}")
    last = len(files) - 2 if last is None else min(last, len(files) - 2)

    # fields to solve: every step-th one, and only where the neighbours are
    # the frames directly before and after (needed for d/dt)
    wanted = list(range(max(first, 1), last + 1, step))
    fields = [t for t in wanted if has_consecutive_neighbours(files, t)]
    skipped = [os.path.basename(files[t]) for t in wanted if t not in fields]

    print(f"PIV folder: {piv_folder}")
    print(f"  {len(files)} files, solving {len(fields)} fields (every {step}), solver {solver}, margin {margin}")
    if skipped:
        print(f"  skipping {len(skipped)} field(s) without consecutive neighbours: {skipped[:5]}")
    print(f"  Ro = {Ro}, E = {settings.ekman_number(Ro):.3e}, "
          f"rotation rate = {settings.rotation_rate(Ro):.3f} rad/s, U = {settings.U_REF:.4f} m/s")

    cache = {}                 # each .dat file is read only once
    setup = lu_data = None
    t_start = time.time()
    for count, t in enumerate(fields, 1):
        plane = load_piv_snapshot(files, t, cache)
        if setup is None:      # the grid is the same for the whole recording
            setup = make_poisson_setup(plane["grid_x"], plane["grid_y"], margin)
            lu_data = factorise(setup) if solver == "splu" else None
        terms, info = compute_pressure(plane, Ro, settings.CLOCKWISE, setup, solver, lu_data)
        save_fields(terms["p"], terms["vorticity"], output_folder, plane["name"])

        if make_figure and count == 1:
            plot_fields(plane, {"pressure $p$": terms["p"], r"vorticity $\omega_z$": terms["vorticity"]},
                        os.path.join(output_folder, f"quicklook_{plane['name']}.png"), crop=margin - 1)
        if count == 1 or count % 50 == 0 or count == len(fields):
            print(f"  [{count}/{len(fields)}] {plane['name']}: {info['iterations']} it, "
                  f"residual {info['residual']:.1e}, filled vectors {plane['n_filled']}, "
                  f"{time.time() - t_start:.0f} s so far")
    print(f"  written to {output_folder}")


if __name__ == "__main__":
    run_piv(PIV_FOLDER, RO, STEP, FIRST, LAST, MARGIN, SOLVER, OUTPUT_FOLDER, MAKE_FIGURE)
