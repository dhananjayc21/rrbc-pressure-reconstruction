# -*- coding: utf-8 -*-
"""
compare_solvers.py
==================

Solves the same plane with all four solvers and shows that they agree:
how many iterations and how much time each one needs, and how far its
pressure is from the SPLU (direct) solution.

How to use: change the USER SETTINGS below and run
    python compare_solvers.py

Prints a table and saves solver_comparison_<name>.png in OUTPUT_FOLDER.
Jacobi takes a few minutes; the others are quicker.
"""

import os
import glob
import numpy as np

import settings
from load_piv import load_piv_snapshot
from load_dns import load_dns_export
from poisson_setup import make_poisson_setup
from solve_pressure import compute_pressure, SOLVERS
from plotting import plot_solver_check


# ==== USER SETTINGS ==========================================================
DATA = "piv"               # "piv" or "dns"
PIV_FOLDER = os.path.join(settings.DATA_FOLDER, "PIV Data sets", "Run9")
PIV_FIELD = 1              # index of the field (needs a field on either side)
PIV_RO = 0.05
DNS_EXPORT = os.path.join(settings.DNS_EXPORT_FOLDER, "dns_export_Ro0.11.npz")
DNS_Z = 0.05
MARGIN = 3
TOLERANCE = 1e-9           # for the iterative solvers (relative residual)
SOLVERS_TO_RUN = ["splu", "sor", "gauss-seidel", "jacobi"]   # splu first: the reference
# (on the larger DNS grid Jacobi and Gauss-Seidel take about an hour; SOR a few minutes)
OUTPUT_FOLDER = os.path.join(settings.DATA_FOLDER, "Results_solver_comparison")
# =============================================================================


def main():
    if DATA == "piv":
        files = sorted(glob.glob(os.path.join(PIV_FOLDER, "PIVlab_*.dat")))
        plane, Ro, sign = load_piv_snapshot(files, PIV_FIELD), PIV_RO, settings.CLOCKWISE
    else:
        plane, Ro = load_dns_export(DNS_EXPORT, DNS_Z)
        sign = settings.ANTICLOCKWISE
    setup = make_poisson_setup(plane["grid_x"], plane["grid_y"], MARGIN)
    inside = setup["inside"]

    results = {}
    for method in SOLVERS_TO_RUN:
        if method not in SOLVERS:
            raise SystemExit(f"unknown solver {method!r}; choose from {SOLVERS}")
        options = {} if method == "splu" else {"tolerance": TOLERANCE}
        print(f"  solving with {method} ...", flush=True)
        terms, info = compute_pressure(plane, Ro, sign, setup, method, **options)
        results[method] = (terms["p"], info)

    first = SOLVERS_TO_RUN[0]                  # the reference
    ref = results[first][0][inside]
    ref = ref - ref.mean()
    span = ref.max() - ref.min()
    print(f"\n{plane['name']}, Ro = {Ro}, grid {setup['shape'][0]} x {setup['shape'][1]}")
    print(f"  {'solver':28s} {'iterations':>10s} {'time (s)':>9s} {'residual':>9s}   max diff from {first}")
    for method, (p, info) in results.items():
        q = p[inside] - p[inside].mean()
        diff = np.max(np.abs(q - ref)) / span * 100
        print(f"  {info['method']:28s} {info['iterations']:10d} {info['time']:9.2f} "
              f"{info['residual']:9.1e}   {diff:.1e} % of the pressure range")

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    path = os.path.join(OUTPUT_FOLDER, f"solver_comparison_{plane['name'].replace(' ', '_').replace('/', '')}.png")
    plot_solver_check(plane, results, path, crop=MARGIN - 1)
    print(f"\n  figure: {path}")


if __name__ == "__main__":
    main()
