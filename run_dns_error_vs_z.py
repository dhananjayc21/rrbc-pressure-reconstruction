# -*- coding: utf-8 -*-
# =============================================================================
# run_dns_error_vs_z.py
#
# Error of the reconstructed pressure against height: every plane of one DNS
# snapshot (k = 2 ... 384, the padded plane index of the DNS notebook) is
# solved full 3D and XY-only and compared with the DNS pressure.
#
# Change the settings below and run:   python run_dns_error_vs_z.py
#
# The snapshot is read in blocks of planes, so that the 384^3 field is never
# in memory all at once.
#
# Writes DNS_export/results_pressure/error_vs_z_Ro<Ro>.csv and .png
# =============================================================================

import os
import glob
import time
import numpy as np
import h5py

import settings
from grid import tanh_grid
from read_dns import dns_coordinates, read_dns_planes, dns_plane
from solve_pressure import prepare_poisson, compute_pressure
from error_measures import rms_error_percent
from plotting import plot_error_vs_z


# ==== SETTINGS ===============================================================
Ro = 0.11            # Rossby number: reads DNS_DRIVE_FOLDER/"Ro = <Ro>"   ----> CHANGE IT
FIRST_FILE = 0       # index of the first of the three consecutive .h5 files
K_FIRST = 2          # first plane to solve (padded index)
K_LAST = 384         # last plane to solve
BLOCK = 48           # planes read at once (more = faster, but more memory)
METHOD = "direct"    # the iterative methods also work, but take minutes per plane
OUTPUT_FOLDER = settings.DNS_RESULTS_FOLDER
# =============================================================================


folder = os.path.join(settings.DNS_DRIVE_FOLDER, "Ro = " + format(Ro, "g"))
paths = sorted(glob.glob(os.path.join(folder, "*.h5")))
paths = paths[FIRST_FILE:FIRST_FILE + 3]
if len(paths) < 3:
    print("Three .h5 files are needed in", folder)
    raise SystemExit
print("Ro =", Ro, ":", paths)

files = []
for path in paths:
    files.append(h5py.File(path, "r"))

X, Y, Z = dns_coordinates(files[0])
grid_x = tanh_grid(X, settings.DNS_L, settings.DNS_BETA)
grid_y = tanh_grid(Y, settings.DNS_L, settings.DNS_BETA)
grid_z = tanh_grid(Z, settings.DNS_L, settings.DNS_BETA)
poisson = prepare_poisson(grid_x, grid_y, METHOD)

k_list = []
z_list = []
err_3d_vs_dns = []
err_xy_vs_dns = []
err_3d_vs_xy = []
start_time = time.time()

block_start = K_FIRST
while block_start <= K_LAST:
    block_end = min(block_start + BLOCK - 1, K_LAST)
    k0 = block_start - 1                 # the block also needs the plane below ...
    k1 = block_end + 1                   # ... and the plane above
    u = read_dns_planes(files, "Vx", k0, k1, X, Y, Z)
    v = read_dns_planes(files, "Vy", k0, k1, X, Y, Z)
    w = read_dns_planes(files, "Vz", k0, k1, X, Y, Z)
    P = read_dns_planes(files, "P", k0, k1, X, Y, Z)

    for k in range(block_start, block_end + 1):
        dns = dns_plane(u, v, w, P, k, k0, grid_x, grid_y, grid_z, Ro, float(Z[k]))
        X_3d, *rest = compute_pressure(dns["u"], dns["v"], dns["dt"], grid_x, grid_y, Ro,
                                       settings.ANTICLOCKWISE, poisson, METHOD, z_terms=dns["z_terms"])
        X_xy, *rest = compute_pressure(dns["u"], dns["v"], dns["dt"], grid_x, grid_y, Ro,
                                       settings.ANTICLOCKWISE, poisson, METHOD, z_terms=None)
        k_list.append(k)
        z_list.append(float(Z[k]))
        err_3d_vs_dns.append(rms_error_percent(dns["p_dns"], X_3d))
        err_xy_vs_dns.append(rms_error_percent(dns["p_dns"], X_xy))
        err_3d_vs_xy.append(rms_error_percent(X_3d, X_xy))

    print("  planes", block_start, "-", block_end, "done (", round(time.time() - start_time), "s ):",
          "3D", round(err_3d_vs_dns[-1], 2), "%, XY", round(err_xy_vs_dns[-1], 2), "%")
    block_start = block_end + 1

for f in files:
    f.close()

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
csv_path = os.path.join(OUTPUT_FOLDER, "error_vs_z_Ro" + format(Ro, "g") + ".csv")
with open(csv_path, "w") as f:
    f.write("k,zH,err_3d_vs_dns,err_xy_vs_dns,err_3d_vs_xy\n")
    for i in range(len(k_list)):
        f.write(str(k_list[i]) + "," + format(z_list[i], ".6f") + "," + format(err_3d_vs_dns[i], ".6f")
                + "," + format(err_xy_vs_dns[i], ".6f") + "," + format(err_3d_vs_xy[i], ".6f") + "\n")

plot_error_vs_z(np.array(z_list), np.array(err_xy_vs_dns), np.array(err_3d_vs_dns),
                np.array(err_3d_vs_xy), csv_path.replace(".csv", ".png"), Ro)
print("Written", csv_path, "and the .png")
