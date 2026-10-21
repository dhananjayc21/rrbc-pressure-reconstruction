# -*- coding: utf-8 -*-
"""
run_dns.py
==========

Pressure on chosen DNS planes, reconstructed twice:
    full 3D   with term 4 (w and the z-derivatives), and
    XY-only   without it, i.e. from exactly what planar PIV can measure,
and both compared with the DNS pressure.

How to use: change the USER SETTINGS below and run
    python run_dns.py

Writes to DNS_export/results_pressure/:
    dns_pressure_Ro<Ro>_z<z>.npz   the three pressure fields, vorticity, grid
    dns_pressure_Ro<Ro>_z<z>.png   the three fields side by side
    dns_rms_errors.csv             one line per plane (appended)
"""

import os
import csv
import glob
import numpy as np

import settings
from load_dns import load_dns_export, load_dns_h5
from poisson_setup import make_poisson_setup
from solver_splu import factorise
from solve_pressure import compute_pressure
from error_measures import rms_error_percent
from plotting import plot_fields


# ==== USER SETTINGS ==========================================================
SOURCE = "export"          # "export" (the small .npz files) or "h5" (the snapshots)

# for SOURCE = "export":
EXPORT_FILES = [os.path.join(settings.DNS_EXPORT_FOLDER, "dns_export_Ro0.11.npz")]
# for SOURCE = "h5": three consecutive snapshots of one run
RO = 0.11                  # Rossby number of the run
FIRST_FILE = 0             # index of the first of the three .h5 files

Z_TARGETS = [0.05, 0.5]    # heights z/H (the two PIV heights)
MARGIN = 3
SOLVER = "splu"            # "splu", "jacobi", "gauss-seidel" or "sor"
OUTPUT_FOLDER = settings.DNS_RESULTS_FOLDER
MAKE_FIGURES = True
# =============================================================================


def load_planes():
    """The list of (plane, Ro) to work on, from the settings above."""
    planes = []
    if SOURCE == "export":
        for path in EXPORT_FILES:
            for z in Z_TARGETS:
                planes.append(load_dns_export(path, z))
    elif SOURCE == "h5":
        folder = os.path.join(settings.DNS_DRIVE_FOLDER, f"Ro = {RO:g}")
        files = sorted(glob.glob(os.path.join(folder, "*.h5")))[FIRST_FILE:FIRST_FILE + 3]
        if len(files) < 3:
            raise SystemExit(f"need three .h5 files in {folder}")
        print("  snapshots:", [os.path.basename(f) for f in files])
        for z in Z_TARGETS:
            planes.append((load_dns_h5(files, z, RO), RO))
    else:
        raise SystemExit("SOURCE must be 'export' or 'h5'")
    return planes


def run_dns():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    rows = []
    setup = lu_data = None
    for plane, Ro in load_planes():
        if setup is None:          # every DNS plane has the same x-y grid
            setup = make_poisson_setup(plane["grid_x"], plane["grid_y"], MARGIN)
            lu_data = factorise(setup) if SOLVER == "splu" else None

        full, info_full = compute_pressure(plane, Ro, settings.ANTICLOCKWISE, setup,
                                           SOLVER, lu_data, use_z_terms=True)
        xy, info_xy = compute_pressure(plane, Ro, settings.ANTICLOCKWISE, setup,
                                       SOLVER, lu_data, use_z_terms=False)
        p_dns = plane["p_dns"]
        errors = {"err_3d_vs_dns": rms_error_percent(p_dns, full["p"], MARGIN),
                  "err_xy_vs_dns": rms_error_percent(p_dns, xy["p"], MARGIN),
                  "err_3d_vs_xy": rms_error_percent(full["p"], xy["p"], MARGIN)}
        rows.append({"Ro": Ro, "zH": plane["zH"], **errors})

        tag = f"Ro{Ro:g}_z{plane['zH']:.3f}"
        np.savez_compressed(os.path.join(OUTPUT_FOLDER, f"dns_pressure_{tag}.npz"),
                            p_full3d=full["p"], p_xy=xy["p"], p_dns=p_dns,
                            vorticity=full["vorticity"],
                            x=plane["grid_x"]["coords"], y=plane["grid_y"]["coords"],
                            zH=plane["zH"], Ro=Ro, margin=MARGIN, **errors)
        if MAKE_FIGURES:
            plot_fields(plane, {"DNS": p_dns, "full 3D": full["p"], "XY only": xy["p"]},
                        os.path.join(OUTPUT_FOLDER, f"dns_pressure_{tag}.png"),
                        crop=MARGIN, arrows=False)
        print(f"  {plane['name']}: 3D vs DNS {errors['err_3d_vs_dns']:.2f} %, "
              f"XY vs DNS {errors['err_xy_vs_dns']:.2f} %, 3D vs XY {errors['err_3d_vs_xy']:.2f} %")

    csv_path = os.path.join(OUTPUT_FOLDER, "dns_rms_errors.csv")
    new_file = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        if new_file:
            writer.writeheader()
        writer.writerows(rows)
    print(f"  written to {OUTPUT_FOLDER}")


if __name__ == "__main__":
    run_dns()
