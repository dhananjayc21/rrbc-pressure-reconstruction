# -*- coding: utf-8 -*-
"""
plotting.py
===========

All the figures, kept apart from the calculation so that the calculation can
run without matplotlib and the figures can be changed without touching it.

    plot_fields()        filled contours of one or more 2-D fields (quick look)
    plot_error_vs_z()    reconstruction error against height (DNS sweep)
    plot_solver_check()  pressure of each solver and its difference from SPLU
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")            # draw to files only (no window needed)
import matplotlib.pyplot as plt


def plot_fields(plane, fields, path, crop=2, arrows=True, title=None):
    """One panel per entry of fields = {panel title: 2-D array}.

    Each field is shown with its mean removed, on a symmetric colour scale set
    by the 99th percentile of |field| (so a few extreme points do not wash out
    the rest). With arrows=True the velocity at time t is drawn on top.
    """
    s = (slice(crop, -crop), slice(crop, -crop))
    X, Y = np.meshgrid(plane["grid_x"]["coords"], plane["grid_y"]["coords"], indexing="ij")
    X, Y = X[s], Y[s]
    U, V = plane["u"][1][s], plane["v"][1][s]
    step = max(1, min(X.shape) // 22)                 # about 22 arrows across

    fig, axes = plt.subplots(1, len(fields), figsize=(5.2 * len(fields), 4.6),
                             constrained_layout=True, squeeze=False)
    for ax, (name, F) in zip(axes[0], fields.items()):
        F = F[s] - np.nanmean(F[s])
        limit = np.nanpercentile(np.abs(F), 99) or 1.0
        cf = ax.contourf(X, Y, F, levels=np.linspace(-limit, limit, 41),
                         cmap="RdBu_r", extend="both")
        if arrows:
            ax.quiver(X[::step, ::step], Y[::step, ::step], U[::step, ::step], V[::step, ::step],
                      color="k", width=0.003, alpha=0.8)
        ax.set_title(name)
        ax.set_xlabel(r"$x/H$")
        ax.set_ylabel(r"$y/H$")
        ax.set_aspect("equal")
        fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.03)
    fig.suptitle(title or plane["name"])
    fig.savefig(path, dpi=200)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Error against height
# -----------------------------------------------------------------------------
ERROR_SERIES = [  # (column of the results, label, colour, line style)
    ("err_xy_vs_dns", "XY-only vs DNS", "#2a78d6", "-"),
    ("err_3d_vs_dns", "Full 3D vs DNS", "#eb6834", "--"),
    ("err_3d_vs_xy", "Full 3D vs XY-only", "#1baf7a", ":"),
]


def plot_error_vs_z(rows, path, Ro, snapshot=""):
    """rows: list of dictionaries with zH and the three error columns."""
    z = np.array([r["zH"] for r in rows])
    fig, ax = plt.subplots(figsize=(7.5, 4.6), constrained_layout=True)
    curves = []
    for key, label, colour, style in ERROR_SERIES:
        e = np.array([r[key] for r in rows])
        ax.plot(z, e, style, color=colour, lw=2, label=label)
        curves.append(np.log10(e))
    ax.set_yscale("log")                          # the errors span several decades
    ax.set_xlim(0, 1)
    for zp in (0.05, 0.5):                        # heights of the PIV planes
        ax.axvline(zp, color="#c3c2b7", lw=0.8, zorder=0)
        ax.text(zp, 0.80, f" PIV plane\n z/H = {zp:g}", transform=ax.get_xaxis_transform(),
                fontsize=8, color="#73726c", va="bottom")
    ax.set_xlabel(r"$z/H$")
    ax.set_ylabel("RMS error (% of max pressure)")
    ax.set_title(f"Pressure reconstruction error vs height, DNS Ro = {Ro:g}\n"
                 f"{('snapshot ' + snapshot + ', ') if snapshot else ''}"
                 f"planes k = {rows[0]['k']}-{rows[-1]['k']}", fontsize=11)
    ax.grid(True, which="major", color="#e4e3de", lw=0.6)
    ax.grid(True, which="minor", color="#f0efec", lw=0.4)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="upper center", ncol=3)
    # last, so that the axes have their final size when the labels are placed
    place_labels(ax, z, curves, [label for _, label, _, _ in ERROR_SERIES])
    fig.savefig(path, dpi=200)
    plt.close(fig)


def place_labels(ax, z, curves, labels, pad=0.012):
    """Write each curve's name next to it, where it touches nothing else.

    Positions along each curve (above and below it) are tried; a position is
    rejected if the label would come too close to another curve or overlap the
    legend, another text or a label already placed. The best remaining one is
    used. If no position is free (e.g. two curves on top of each other), the
    curve gets no label of its own; the legend still names it.
    All sizes are in "axes fraction" units (0 to 1 across the plot).
    """
    fig = ax.figure
    fig.canvas.draw()                                   # so that sizes are known
    renderer = fig.canvas.get_renderer()
    axes_box = ax.get_window_extent(renderer)

    def as_fraction(box):                               # screen box -> (x0, x1, y0, y1)
        return ((box.x0 - axes_box.x0) / axes_box.width, (box.x1 - axes_box.x0) / axes_box.width,
                (box.y0 - axes_box.y0) / axes_box.height, (box.y1 - axes_box.y0) / axes_box.height)

    taken = [as_fraction(t.get_window_extent(renderer)) for t in ax.texts]
    if ax.get_legend() is not None:
        taken.append(as_fraction(ax.get_legend().get_window_extent(renderer)))
    lo, hi = np.log10(ax.get_ylim())
    F = [(c - lo) / (hi - lo) for c in curves]          # curves in axes fraction

    def overlap(a, b):
        return a[0] < b[1] and b[0] < a[1] and a[2] < b[3] and b[2] < a[3]

    for i, label in enumerate(labels):
        text = ax.text(0, 0, label, transform=ax.transAxes, fontsize=9,
                       color="#3d3d3a", ha="center", va="center")
        x0, x1, y0, y1 = as_fraction(text.get_window_extent(renderer))
        w, h = x1 - x0, y1 - y0
        best = None
        for x in np.arange(0.04 + w / 2, 0.96 - w / 2, 0.005):
            xs = np.linspace(x - w / 2, x + w / 2, 15)
            own = np.interp(xs, z, F[i])
            for side in (1, -1):
                y = own.max() + pad + h / 2 if side > 0 else own.min() - pad - h / 2
                box = (x - w / 2, x + w / 2, y - h / 2, y + h / 2)
                if box[2] < 0.02 or box[3] > 0.98 or any(overlap(box, t) for t in taken):
                    continue
                room = 1.0                               # distance to the nearest other curve
                for j, other in enumerate(F):
                    if j == i:
                        continue
                    vals = np.interp(xs, z, other)
                    inside = (vals >= box[2]) & (vals <= box[3])
                    gap = np.where(inside, 0.0, np.minimum(np.abs(vals - box[2]), np.abs(vals - box[3])))
                    room = min(room, gap.min())
                if room < 1.5 * pad:
                    continue
                score = min(room, 0.15) - abs(y - np.interp(x, z, F[i]))
                if best is None or score > best[0]:
                    best = (score, x, y, box)
        if best is None:
            text.remove()
        else:
            text.set_position((best[1], best[2]))
            taken.append(best[3])


# -----------------------------------------------------------------------------
# Solver comparison
# -----------------------------------------------------------------------------
def plot_solver_check(plane, results, path, crop=2):
    """results = {solver name: (pressure, info)}; the first one is the reference.

    Top row: the pressure of every solver. Bottom row: its difference from the
    reference, as % of the reference pressure range.
    """
    s = (slice(crop, -crop), slice(crop, -crop))
    X, Y = np.meshgrid(plane["grid_x"]["coords"], plane["grid_y"]["coords"], indexing="ij")
    X, Y = X[s], Y[s]
    names = list(results)
    ref = results[names[0]][0][s]
    ref = ref - np.nanmean(ref)
    span = np.nanmax(ref) - np.nanmin(ref)
    limit = np.nanpercentile(np.abs(ref), 99)

    fig, axes = plt.subplots(2, len(names), figsize=(3.6 * len(names), 6.2),
                             constrained_layout=True, squeeze=False)
    for col, name in enumerate(names):
        p, info = results[name]
        p = p[s] - np.nanmean(p[s])
        cf = axes[0, col].contourf(X, Y, p, levels=np.linspace(-limit, limit, 41),
                                   cmap="RdBu_r", extend="both")
        axes[0, col].set_title(f"{name}\n{info['iterations']} it, {info['time']:.2f} s", fontsize=9)
        diff = (p - ref) / span * 100
        d = np.nanmax(np.abs(diff))
        scale = max(d, 1e-12)                 # a colour scale needs a non-zero range
        cd = axes[1, col].contourf(X, Y, diff, levels=np.linspace(-scale, scale, 21), cmap="PuOr")
        axes[1, col].set_title(f"difference from {names[0]}\nmax {d:.1e} % of range", fontsize=9)
        fig.colorbar(cd, ax=axes[1, col], fraction=0.046, pad=0.03)
        for row in (0, 1):
            axes[row, col].set_aspect("equal")
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
    fig.colorbar(cf, ax=axes[0, -1], fraction=0.046, pad=0.03)
    fig.suptitle(plane["name"])
    fig.savefig(path, dpi=150)
    plt.close(fig)
