# -*- coding: utf-8 -*-
# =============================================================================
# compare_solvers.py
#
# Solves the same plane with the four methods and shows that they agree:
# how many iterations and how much time each one needs, and how far its
# pressure is from the direct solution.
#
# Change the settings below and run:   python compare_solvers.py
# Prints a table and saves solver_comparison_<name>.png in OUTPUT_FOLDER.
# Jacobi takes a few minutes; the others are quicker.
# =============================================================================

import os
import glob
import time
import numpy as np

import settings
from read_piv import read_three_fields
from read_dns import read_dns_export
from solve_pressure import prepare_poisson, compute_pressure
from plotting import plot_solver_check


# ==== SETTINGS ===============================================================
DATA = "piv"         # "piv" or "dns"

PIV_FOLDER = os.path.join(settings.DATA_FOLDER, "PIV Data sets", "Run9")
PIV_FIELD = 1        # index of the field (needs a field on either side)
PIV_RO = 0.05

DNS_EXPORT = os.path.join(settings.DNS_EXPORT_FOLDER, "dns_export_Ro0.11.npz")
DNS_Z = 0.05

TOLERANCE = 1e-9                                       # for the iterative methods
METHODS = ["direct", "sor", "gauss-seidel", "jacobi"]  # the first one is the reference
OUTPUT_FOLDER = os.path.join(settings.DATA_FOLDER, "Results_solver_comparison")
# =============================================================================


# ---- the plane ---------------------------------------------------------------
if DATA == "piv":
    files = sorted(glob.glob(os.path.join(PIV_FOLDER, "PIVlab_*.dat")))
    u, v, grid_x, grid_y, n_filled = read_three_fields(files, PIV_FIELD)
    dt = settings.dt_fields / settings.T_ndim
    Ro = PIV_RO
    rotation = settings.CLOCKWISE
    z_terms = None
    name = os.path.splitext(os.path.basename(files[PIV_FIELD]))[0]
else:
    dns = read_dns_export(DNS_EXPORT, DNS_Z)
    u, v, dt = dns["u"], dns["v"], dns["dt"]
    grid_x, grid_y = dns["grid_x"], dns["grid_y"]
    Ro = dns["Ro"]
    rotation = settings.ANTICLOCKWISE
    z_terms = dns["z_terms"]
    name = dns["name"]

poisson = prepare_poisson(grid_x, grid_y, "direct")


# ---- solve with every method ------------------------------------------------
pressures = []
labels = []
print()
print("method          iterations   time (s)   residual   max diff from", METHODS[0])
for method in METHODS:
    start = time.time()
    X, Vor, iterations, e, us = compute_pressure(u, v, dt, grid_x, grid_y, Ro, rotation, poisson,
                                                 method, z_terms, TOLERANCE)
    seconds = time.time() - start

    m = settings.MARGIN
    p = X[m:-m, m:-m] - np.mean(X[m:-m, m:-m])
    if len(pressures) == 0:
        reference = p
    difference = np.max(np.abs(p - reference)) / (np.max(reference) - np.min(reference)) * 100

    print(format(method, "15s"), format(iterations, "10d"), format(seconds, "10.2f"),
          format(e, "10.1e"), "  ", format(difference, ".1e"), "% of the pressure range")
    pressures.append(X)
    labels.append(method + "\n" + str(iterations) + " it, " + format(seconds, ".2f") + " s")

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
path = os.path.join(OUTPUT_FOLDER, "solver_comparison_" + name.replace(" ", "_").replace("/", "") + ".png")
plot_solver_check(pressures, labels, grid_x, grid_y, path, title=name)
print()
print("Figure:", path)
