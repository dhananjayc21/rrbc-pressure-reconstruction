# -*- coding: utf-8 -*-
"""
load_dns.py
===========

Reading DNS data into the same "plane" dictionary as load_piv.py, with two
extra entries that planar PIV does not have:

    plane["z_terms"]   w and the z-derivatives at the plane and time t
                       (term 4 of the Poisson equation), see z_terms()
    plane["p_dns"]     the DNS pressure on the plane, to check the result

Two sources:
    load_dns_export()  the small .npz exports in DNS_export/ (two heights)
    load_dns_h5()      three consecutive .h5 snapshots (any height)

The grid is the padded DNS grid of the notebook: the stored coordinates lose
their first and last entries, get one extra point at each end, and those two
end points are put on the tanh stretching function.
"""

import os
import numpy as np

import settings
from grid import tanh_grid, tanh_coords


def z_terms(u3, v3, w3, gz, k):
    """w and the z-derivatives of u and v at plane k, time t.

    u3, v3, w3: shape (nx, ny, 3) = planes k-1, k, k+1 at time t.
    gz: grid dictionary of z. Second-order central differences in z.
    """
    h = gz["d_xi"]
    z_xi, z_xixi = gz["x_xi"][k], gz["x_xixi"][k]

    def first(f):
        return (f[..., 2] - f[..., 0]) / (2 * h) / z_xi

    def second(f):
        df = (f[..., 2] - f[..., 0]) / (2 * h)
        d2f = (f[..., 2] - 2 * f[..., 1] + f[..., 0]) / h ** 2
        return d2f / z_xi ** 2 - (z_xixi / z_xi ** 3) * df

    return {"w": w3[..., 1],
            "du_dz": first(u3), "dv_dz": first(v3),
            "d2u_dz2": second(u3), "d2v_dz2": second(v3)}


# -----------------------------------------------------------------------------
# From the .npz exports
# -----------------------------------------------------------------------------
def load_dns_export(path, z_target):
    """Plane dictionary and Ro from an export written by dns_extract_slabs.py.

    z_target: one of the heights in the file (0.05 or 0.5). The slabs have
    shape (3 times, n, n, 3 planes); time 1 and plane 1 are the middle ones.
    """
    data = np.load(path, allow_pickle=True)
    key = f"z{z_target:g}"
    if f"{key}_u" not in data.files:
        raise KeyError(f"{path} has no plane for z/H = {z_target}")

    gx = tanh_grid(data["X"], float(data["Lx"]), float(data["beta_x"]))
    gy = tanh_grid(data["Y"], float(data["Ly"]), float(data["beta_y"]))
    gz = tanh_grid(data["Z"], float(data["Lz"]), float(data["beta_z"]))
    k = int(data[f"{key}_index"])
    t = int(data["t"])
    u, v, w, P = (data[f"{key}_{name}"].astype(np.float64) for name in "uvwP")

    Ro = float(data["Ro"])
    zH = float(data[f"{key}_zH"])
    plane = {"u": u[:, :, :, 1], "v": v[:, :, :, 1], "dt": float(data["dt"]),
             "grid_x": gx, "grid_y": gy,
             "z_terms": z_terms(u[t], v[t], w[t], gz, k),
             "p_dns": P[t, :, :, 1],
             "name": f"DNS Ro={Ro:g} z/H={zH:.4f}", "zH": zH, "k": k}
    return plane, Ro


# -----------------------------------------------------------------------------
# From the .h5 snapshots
# -----------------------------------------------------------------------------
def dns_coordinates(h5_file):
    """Padded x, y, z coordinates, rebuilt as in the DNS notebook."""
    out = []
    for name in "XYZ":
        c = np.pad(h5_file[name][1:-1].astype(np.float64), 1, mode="edge")
        c[[0, -1]] = tanh_coords(c.size, settings.DNS_L, settings.DNS_BETA)[[0, -1]]
        out.append(c)
    return out


def read_dns_planes(h5_files, name, k0, k1, X, Y, Z):
    """Padded planes k0 ... k1 (inclusive) of one field, all three snapshots.

    Returns shape (3, n, n, k1 - k0 + 1). Padded plane k is data plane k - 1.
    The extra ring of ghost points around the data gets the values of the DNS
    notebook: a no-slip reflection for the velocity (Vx, Vy, Vz), a copy for
    the pressure (P). The ghost planes 0 and 385 (beyond the bottom and top
    plates) are made the same way when they are asked for.
    """
    L = settings.DNS_L
    velocity = name != "P"
    n_data = h5_files[0][name].shape[2]
    d0, d1 = max(k0 - 1, 0), min(k1 - 1, n_data - 1)
    a = np.stack([f[name][:, :, d0:d1 + 1].astype(np.float64) for f in h5_files])
    a = np.pad(a, ((0, 0), (1, 1), (1, 1), (0, 0)), mode="edge")
    if velocity:                                         # side walls
        a[:, -1] = -((X[-1] - L) / (L - X[-2])) * a[:, -2]
        a[:, 0] = (X[0] / X[1]) * a[:, 1]
        a[:, :, -1] = -((Y[-1] - L) / (L - Y[-2])) * a[:, :, -2]
        a[:, :, 0] = (Y[0] / Y[1]) * a[:, :, 1]
    if k0 == 0:                                          # below the bottom plate
        ghost = (Z[0] / Z[1]) * a[..., :1] if velocity else a[..., :1]
        a = np.concatenate([ghost, a], axis=-1)
    if k1 == Z.size - 1:                                 # above the top plate
        ghost = -((Z[-1] - L) / (L - Z[-2])) * a[..., -1:] if velocity else a[..., -1:]
        a = np.concatenate([a, ghost], axis=-1)
    return a


def plane_from_blocks(u, v, w, P, k, k0, gx, gy, gz, Ro, zH):
    """Plane dictionary for plane k, from blocks read with read_dns_planes
    (which start at padded plane k0)."""
    s = slice(k - 1 - k0, k + 2 - k0)                     # planes k-1, k, k+1
    return {"u": u[..., k - k0], "v": v[..., k - k0], "dt": settings.DNS_DT,
            "grid_x": gx, "grid_y": gy,
            "z_terms": z_terms(u[1][..., s], v[1][..., s], w[1][..., s], gz, k),
            "p_dns": P[1, :, :, k - k0],
            "name": f"DNS Ro={Ro:g} z/H={zH:.4f}", "zH": zH, "k": k}


def load_dns_h5(file_paths, z_target, Ro):
    """Plane dictionary for the grid plane nearest to z_target, from three
    consecutive .h5 snapshots (in time order)."""
    import h5py
    files = [h5py.File(p, "r") for p in file_paths]
    X, Y, Z = dns_coordinates(files[0])
    k = int(np.argmin(np.abs(Z - z_target)))
    L, b = settings.DNS_L, settings.DNS_BETA
    gx, gy, gz = tanh_grid(X, L, b), tanh_grid(Y, L, b), tanh_grid(Z, L, b)
    u, v, w, P = (read_dns_planes(files, name, k - 1, k + 1, X, Y, Z)
                  for name in ("Vx", "Vy", "Vz", "P"))
    for f in files:
        f.close()
    return plane_from_blocks(u, v, w, P, k, k - 1, gx, gy, gz, Ro, float(Z[k]))
