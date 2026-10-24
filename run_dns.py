# -*- coding: utf-8 -*-
# =============================================================================
# run_dns.py
#
# Pressure on DNS planes, reconstructed twice:
#     full 3D   with term 4 (w and the z-derivatives), and
#     XY-only   without it, i.e. from exactly what planar PIV can measure,
# and both compared with the DNS pressure.
#
# Change the settings below and run:   python run_dns.py
#
# Writes to DNS_export/results_pressure/:
#     dns_pressure_Ro<Ro>_z<z>.npz    the three pressure fields, vorticity, grid
#     dns_pressure_Ro<Ro>_z<z>.png    the three fields side by side
#     dns_rms_errors.csv              one line per plane (added to the file)
# =============================================================================

import os
import glob
import numpy as np

import settings
from read_dns import read_dns_export, read_dns_h5
from solve_pressure import prepare_poisson, compute_pressure
from error_measures import rms_error_percent
from plotting import plot_fields


# ==== SETTINGS ===============================================================
SOURCE = "export"    # "export" (the small .npz files) or "h5" (the snapshots)

# for SOURCE = "export":
EXPORT_FILES = [os.path.join(settings.DNS_EXPORT_FOLDER, "dns_export_Ro0.11.npz")]

# for SOURCE = "h5": three consecutive snapshots of one run
Ro = 0.11            # Rossby number of the run   ----> CHANGE IT FOR EVERY RUN
FIRST_FILE = 0       # index of the first of the three .h5 files

Z_TARGETS = [0.05, 0.5]      # heights z/H (the two PIV heights)
METHOD = "direct"            # "direct", "jacobi", "gauss-seidel" or "sor"
OUTPUT_FOLDER = settings.DNS_RESULTS_FOLDER
MAKE_FIGURES = True
# =============================================================================


# ---- read the planes --------------------------------------------------------
planes = []
if SOURCE == "export":
    for path in EXPORT_FILES:
        for z in Z_TARGETS:
            planes.append(read_dns_export(path, z))
elif SOURCE == "h5":
    folder = os.path.join(settings.DNS_DRIVE_FOLDER, "Ro = " + format(Ro, "g"))
    files = sorted(glob.glob(os.path.join(folder, "*.h5")))
    files = files[FIRST_FILE:FIRST_FILE + 3]
    if len(files) < 3:
        print("Three .h5 files are needed in", folder)
        raise SystemExit
    print("Snapshots:", files)
    for z in Z_TARGETS:
        planes.append(read_dns_h5(files, z, Ro))
else:
    print("SOURCE must be 'export' or 'h5'")
    raise SystemExit


# ---- solve every plane ------------------------------------------------------
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
csv_path = os.path.join(OUTPUT_FOLDER, "dns_rms_errors.csv")
if not os.path.exists(csv_path):
    with open(csv_path, "w") as f:
        f.write("Ro,zH,err_3d_vs_dns,err_xy_vs_dns,err_3d_vs_xy\n")

poisson = None
for dns in planes:
    if poisson is None:                  # every DNS plane has the same x-y grid
        poisson = prepare_poisson(dns["grid_x"], dns["grid_y"], METHOD)

    # full 3D: with the z_terms (term 4)
    X_3d, Vor, it, e, us = compute_pressure(dns["u"], dns["v"], dns["dt"], dns["grid_x"], dns["grid_y"],
                                            dns["Ro"], settings.ANTICLOCKWISE, poisson, METHOD,
                                            z_terms=dns["z_terms"])
    # XY-only: without them, like PIV
    X_xy, Vor, it, e, us = compute_pressure(dns["u"], dns["v"], dns["dt"], dns["grid_x"], dns["grid_y"],
                                            dns["Ro"], settings.ANTICLOCKWISE, poisson, METHOD,
                                            z_terms=None)
    p_dns = dns["p_dns"]

    err_3d_vs_dns = rms_error_percent(p_dns, X_3d)
    err_xy_vs_dns = rms_error_percent(p_dns, X_xy)
    err_3d_vs_xy = rms_error_percent(X_3d, X_xy)
    print(" ", dns["name"], ":  3D vs DNS", round(err_3d_vs_dns, 2), "%,  XY vs DNS",
          round(err_xy_vs_dns, 2), "%,  3D vs XY", round(err_3d_vs_xy, 2), "%")

    tag = "Ro" + format(dns["Ro"], "g") + "_z" + format(dns["zH"], ".3f")
    np.savez_compressed(os.path.join(OUTPUT_FOLDER, "dns_pressure_" + tag + ".npz"),
                        p_full3d=X_3d, p_xy=X_xy, p_dns=p_dns, vorticity=Vor,
                        x=dns["grid_x"]["coords"], y=dns["grid_y"]["coords"],
                        zH=dns["zH"], Ro=dns["Ro"],
                        err_3d_vs_dns=err_3d_vs_dns, err_xy_vs_dns=err_xy_vs_dns, err_3d_vs_xy=err_3d_vs_xy)

    with open(csv_path, "a") as f:
        f.write(format(dns["Ro"], "g") + "," + format(dns["zH"], ".5f") + "," + format(err_3d_vs_dns, ".5f")
                + "," + format(err_xy_vs_dns, ".5f") + "," + format(err_3d_vs_xy, ".5f") + "\n")

    if MAKE_FIGURES:
        plot_fields([p_dns, X_3d, X_xy], ["DNS", "full 3D", "XY only"], dns["grid_x"], dns["grid_y"],
                    os.path.join(OUTPUT_FOLDER, "dns_pressure_" + tag + ".png"),
                    title=dns["name"], crop=settings.MARGIN)

print("Written to", OUTPUT_FOLDER)
