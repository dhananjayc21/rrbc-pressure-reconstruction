# -*- coding: utf-8 -*-
"""
run_dns_error_vs_z.py
=====================

Error of the reconstructed pressure against height: every plane of one DNS
snapshot (k = 2 ... 384, the padded plane index of the notebook) is solved
full 3D and XY-only and compared with the DNS pressure.

How to use: change the USER SETTINGS below and run
    python run_dns_error_vs_z.py

The snapshot is read in blocks of planes, so the 384^3 field is never in
memory at once. Planes next to the plates use the ghost planes beyond the
plates (no-slip values, as in the notebook; see load_dns.read_dns_planes).

Writes DNS_export/results_pressure/error_vs_z_Ro<Ro>.csv and .png
"""

import os
import csv
import glob
import time
import numpy as np
import h5py

import settings
from grid import tanh_grid
from load_dns import dns_coordinates, read_dns_planes, plane_from_blocks
from poisson_setup import make_poisson_setup
from solver_splu import factorise
from solve_pressure import compute_pressure
from error_measures import rms_error_percent
from plotting import plot_error_vs_z


# ==== USER SETTINGS ==========================================================
RO = 0.11                  # Rossby number: reads DNS_DRIVE_FOLDER/"Ro = <RO>"
FIRST_FILE = 0             # index of the first of the three consecutive .h5 files
K_FIRST, K_LAST = 2, 384   # planes to solve (padded index)
BLOCK = 48                 # planes read at once (more = faster, but more memory)
MARGIN = 3
SOLVER = "splu"            # the iterative solvers work too, but take minutes per plane
OUTPUT_FOLDER = settings.DNS_RESULTS_FOLDER
# =============================================================================


def run_error_vs_z(Ro=RO, first_file=FIRST_FILE, k_first=K_FIRST, k_last=K_LAST):
    folder = os.path.join(settings.DNS_DRIVE_FOLDER, f"Ro = {Ro:g}")
    names = sorted(os.path.basename(p) for p in glob.glob(os.path.join(folder, "*.h5")))
    names = names[first_file:first_file + 3]
    if len(names) < 3:
        raise SystemExit(f"need three .h5 files in {folder}")
    files = [h5py.File(os.path.join(folder, n), "r") for n in names]

    X, Y, Z = dns_coordinates(files[0])
    L, b = settings.DNS_L, settings.DNS_BETA
    gx, gy, gz = tanh_grid(X, L, b), tanh_grid(Y, L, b), tanh_grid(Z, L, b)
    setup = make_poisson_setup(gx, gy, MARGIN)
    lu_data = factorise(setup) if SOLVER == "splu" else None
    print(f"Ro = {Ro}: {names}")

    rows, t0 = [], time.time()
    for b0 in range(k_first, k_last + 1, BLOCK):
        ks = range(b0, min(b0 + BLOCK, k_last + 1))
        k0, k1 = ks[0] - 1, ks[-1] + 1                   # planes needed for the block
        u, v, w, P = (read_dns_planes(files, name, k0, k1, X, Y, Z)
                      for name in ("Vx", "Vy", "Vz", "P"))
        for k in ks:
            plane = plane_from_blocks(u, v, w, P, k, k0, gx, gy, gz, Ro, float(Z[k]))
            full, info_full = compute_pressure(plane, Ro, settings.ANTICLOCKWISE, setup,
                                               SOLVER, lu_data, use_z_terms=True)
            xy, info_xy = compute_pressure(plane, Ro, settings.ANTICLOCKWISE, setup,
                                           SOLVER, lu_data, use_z_terms=False)
            rows.append({"k": k, "zH": float(Z[k]),
                         "err_3d_vs_dns": rms_error_percent(plane["p_dns"], full["p"], MARGIN),
                         "err_xy_vs_dns": rms_error_percent(plane["p_dns"], xy["p"], MARGIN),
                         "err_3d_vs_xy": rms_error_percent(full["p"], xy["p"], MARGIN),
                         "uniform_source_3d": info_full["uniform_source"],
                         "uniform_source_xy": info_xy["uniform_source"]})
        del u, v, w, P
        r = rows[-1]
        print(f"  planes {ks[0]}-{ks[-1]} done ({time.time() - t0:.0f} s); k = {r['k']}: "
              f"3D {r['err_3d_vs_dns']:.2f} %, XY {r['err_xy_vs_dns']:.2f} %")
    for f in files:
        f.close()

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    csv_path = os.path.join(OUTPUT_FOLDER, f"error_vs_z_Ro{Ro:g}.csv")
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    snapshot = os.path.splitext(names[1])[0].rstrip("0").rstrip(".")    # the middle one
    plot_error_vs_z(rows, csv_path.replace(".csv", ".png"), Ro, snapshot)
    print(f"  written {csv_path} and the .png")
    return rows


if __name__ == "__main__":
    run_error_vs_z()
