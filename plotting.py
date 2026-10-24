# -*- coding: utf-8 -*-
# =============================================================================
# plotting.py
#
# The figures, kept apart from the calculation:
#
#     plot_fields()         contour plots of one or more fields side by side
#     plot_error_vs_z()     reconstruction error against height (DNS)
#     plot_solver_check()   the pressure of every solver and its difference
# =============================================================================

import numpy as np
import matplotlib
matplotlib.use("Agg")            # only save to files, no window
import matplotlib.pyplot as plt


def plot_fields(fields, titles, grid_x, grid_y, path, u=None, v=None, title="", crop=2):
    # fields: list of 2-D arrays, titles: list of panel titles.
    # Every field is shown with its mean removed, on a colour scale that is
    # symmetric about zero and set by the 99th percentile of |field|.
    # If u and v are given, velocity arrows are drawn on top.
    X, Y = np.meshgrid(grid_x["coords"], grid_y["coords"], indexing="ij")
    X = X[crop:-crop, crop:-crop]
    Y = Y[crop:-crop, crop:-crop]

    n = len(fields)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 4.6), constrained_layout=True)
    if n == 1:
        axes = [axes]

    for k in range(n):
        ax = axes[k]
        F = fields[k][crop:-crop, crop:-crop]
        F = F - np.nanmean(F)
        limit = np.nanpercentile(np.abs(F), 99)
        if limit == 0:
            limit = 1.0
        levels = np.linspace(-limit, limit, 41)
        cf = ax.contourf(X, Y, F, levels=levels, cmap="RdBu_r", extend="both")

        if u is not None and v is not None:
            step = max(1, min(X.shape) // 22)          # about 22 arrows across
            U = u[crop:-crop, crop:-crop]
            V = v[crop:-crop, crop:-crop]
            ax.quiver(X[::step, ::step], Y[::step, ::step], U[::step, ::step], V[::step, ::step],
                      color="k", width=0.003, alpha=0.8)

        ax.set_title(titles[k])
        ax.set_xlabel(r"$x/H$")
        ax.set_ylabel(r"$y/H$")
        ax.set_aspect("equal")
        fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.03)

    fig.suptitle(title)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_error_vs_z(z, err_xy_vs_dns, err_3d_vs_dns, err_3d_vs_xy, path, Ro):
    # The three errors against height, on a logarithmic error axis
    fig, ax = plt.subplots(figsize=(7.5, 4.6), constrained_layout=True)
    ax.plot(z, err_xy_vs_dns, "-", color="#2a78d6", lw=2, label="XY-only vs DNS")
    ax.plot(z, err_3d_vs_dns, "--", color="#eb6834", lw=2, label="Full 3D vs DNS")
    ax.plot(z, err_3d_vs_xy, ":", color="#1baf7a", lw=2, label="Full 3D vs XY-only")
    ax.set_yscale("log")
    ax.set_xlim(0, 1)

    for z_piv in [0.05, 0.5]:                 # heights of the PIV planes
        ax.axvline(z_piv, color="#c3c2b7", lw=0.8, zorder=0)

    ax.set_xlabel(r"$z/H$")
    ax.set_ylabel("RMS error (% of max pressure)")
    ax.set_title("Pressure reconstruction error vs height, DNS Ro = " + format(Ro, "g"))
    ax.grid(True, color="#e4e3de", lw=0.6)
    ax.legend(frameon=False)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_solver_check(pressures, names, grid_x, grid_y, path, title="", crop=2):
    # Top row: the pressure of every solver. Bottom row: its difference from the
    # first one, in % of the pressure range of the first one.
    X, Y = np.meshgrid(grid_x["coords"], grid_y["coords"], indexing="ij")
    X = X[crop:-crop, crop:-crop]
    Y = Y[crop:-crop, crop:-crop]

    ref = pressures[0][crop:-crop, crop:-crop]
    ref = ref - np.nanmean(ref)
    span = np.nanmax(ref) - np.nanmin(ref)
    limit = np.nanpercentile(np.abs(ref), 99)

    n = len(pressures)
    fig, axes = plt.subplots(2, n, figsize=(3.6 * n, 6.2), constrained_layout=True)
    if n == 1:
        axes = np.array(axes).reshape(2, 1)

    for k in range(n):
        p = pressures[k][crop:-crop, crop:-crop]
        p = p - np.nanmean(p)
        cf = axes[0, k].contourf(X, Y, p, levels=np.linspace(-limit, limit, 41),
                                 cmap="RdBu_r", extend="both")
        axes[0, k].set_title(names[k], fontsize=9)

        difference = (p - ref) / span * 100
        d = np.nanmax(np.abs(difference))
        scale = max(d, 1e-12)                  # a colour scale needs a non-zero range
        cd = axes[1, k].contourf(X, Y, difference, levels=np.linspace(-scale, scale, 21), cmap="PuOr")
        axes[1, k].set_title("difference from " + names[0] + "\nmax " + format(d, ".1e") + " % of range",
                             fontsize=9)
        fig.colorbar(cd, ax=axes[1, k], fraction=0.046, pad=0.03)

        for row in [0, 1]:
            axes[row, k].set_aspect("equal")
            axes[row, k].set_xticks([])
            axes[row, k].set_yticks([])

    fig.colorbar(cf, ax=axes[0, -1], fraction=0.046, pad=0.03)
    fig.suptitle(title)
    fig.savefig(path, dpi=150)
    plt.close(fig)
