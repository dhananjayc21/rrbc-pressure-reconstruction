# -*- coding: utf-8 -*-
# =============================================================================
# read_dns.py
#
# Reading the DNS data. The DNS gives more than planar PIV:
#   - w and the z-derivatives of u and v (term 4 of the Poisson equation)
#   - the DNS pressure itself, to check the reconstructed pressure
#
# Two kinds of files:
#   read_dns_export()   the small .npz exports in DNS_export/ (two heights)
#   read_dns_h5()       three consecutive .h5 snapshots (any height)
#
# Both return a dictionary "dns" with
#     dns["u"], dns["v"]   velocity at three times, shape (3, N, N)
#     dns["dt"]            time between the three snapshots
#     dns["grid_x"], dns["grid_y"]
#     dns["z_terms"]       w and the z-derivatives at time t (see z_derivatives)
#     dns["p_dns"]         DNS pressure on the plane at time t
#     dns["zH"], dns["k"]  height and plane index
#     dns["name"]          text for prints and figures
# =============================================================================

import os
import numpy as np

import settings
from grid import tanh_grid, tanh_coords


def z_derivatives(u3, v3, w3, grid_z, k):
    # w and the z-derivatives of u and v at plane k.
    # u3, v3, w3 have shape (N, N, 3): the planes k-1, k, k+1 at time t.
    h = grid_z["d_xi"]
    z_xi = grid_z["x_xi"][k]
    z_xixi = grid_z["x_xixi"][k]

    du_dxi = (u3[:, :, 2] - u3[:, :, 0]) / (2 * h)
    dv_dxi = (v3[:, :, 2] - v3[:, :, 0]) / (2 * h)
    d2u_dxi2 = (u3[:, :, 2] - 2 * u3[:, :, 1] + u3[:, :, 0]) / (h * h)
    d2v_dxi2 = (v3[:, :, 2] - 2 * v3[:, :, 1] + v3[:, :, 0]) / (h * h)

    z_terms = {}
    z_terms["w"] = w3[:, :, 1]
    z_terms["du_dz"] = du_dxi / z_xi
    z_terms["dv_dz"] = dv_dxi / z_xi
    z_terms["d2u_dz2"] = d2u_dxi2 / z_xi ** 2 - (z_xixi / z_xi ** 3) * du_dxi
    z_terms["d2v_dz2"] = d2v_dxi2 / z_xi ** 2 - (z_xixi / z_xi ** 3) * dv_dxi
    return z_terms


# -----------------------------------------------------------------------------
# The .npz exports
# -----------------------------------------------------------------------------
def read_dns_export(path, z_target):
    # z_target: one of the heights stored in the file (0.05 or 0.5).
    # The stored blocks have shape (3 times, N, N, 3 planes); the middle time
    # and the middle plane are the ones we solve.
    data = np.load(path, allow_pickle=True)
    key = "z" + format(z_target, "g")
    if key + "_u" not in data.files:
        print("The file", path, "has no plane at z/H =", z_target)
        return None

    grid_x = tanh_grid(data["X"], float(data["Lx"]), float(data["beta_x"]))
    grid_y = tanh_grid(data["Y"], float(data["Ly"]), float(data["beta_y"]))
    grid_z = tanh_grid(data["Z"], float(data["Lz"]), float(data["beta_z"]))
    k = int(data[key + "_index"])
    t = int(data["t"])

    u = data[key + "_u"].astype(float)
    v = data[key + "_v"].astype(float)
    w = data[key + "_w"].astype(float)
    P = data[key + "_P"].astype(float)

    dns = {}
    dns["Ro"] = float(data["Ro"])
    dns["zH"] = float(data[key + "_zH"])
    dns["k"] = k
    dns["u"] = u[:, :, :, 1]                 # the middle plane, all three times
    dns["v"] = v[:, :, :, 1]
    dns["dt"] = float(data["dt"])
    dns["grid_x"] = grid_x
    dns["grid_y"] = grid_y
    dns["z_terms"] = z_derivatives(u[t], v[t], w[t], grid_z, k)
    dns["p_dns"] = P[t, :, :, 1]
    dns["name"] = "DNS Ro=" + format(dns["Ro"], "g") + " z/H=" + format(dns["zH"], ".4f")
    return dns


# -----------------------------------------------------------------------------
# The .h5 snapshots
# -----------------------------------------------------------------------------
def dns_coordinates(h5_file):
    # x, y, z of the padded DNS grid, as in the DNS notebook: the stored
    # coordinates lose their first and last entry, get one extra point at each
    # end, and those two end points are put on the tanh stretching function.
    coordinates = []
    for name in ["X", "Y", "Z"]:
        c = h5_file[name][1:-1].astype(float)
        c = np.pad(c, 1, mode="edge")
        ends = tanh_coords(len(c), settings.DNS_L, settings.DNS_BETA)
        c[0] = ends[0]
        c[-1] = ends[-1]
        coordinates.append(c)
    return coordinates[0], coordinates[1], coordinates[2]


def read_dns_planes(h5_files, name, k0, k1, X, Y, Z):
    # Padded planes k0 ... k1 (k1 included) of one field ("Vx", "Vy", "Vz" or
    # "P"), for all three snapshots. Result shape: (3, N, N, k1 - k0 + 1).
    # Padded plane k is data plane k - 1.
    #
    # The ring of ghost points around the data gets the values of the DNS
    # notebook: a no-slip reflection for the velocity, a copy for the pressure.
    # The ghost planes below the bottom plate (k = 0) and above the top plate
    # are made the same way when they are asked for.
    L = settings.DNS_L
    is_velocity = (name != "P")
    n_data = h5_files[0][name].shape[2]
    d0 = max(k0 - 1, 0)
    d1 = min(k1 - 1, n_data - 1)

    blocks = []
    for f in h5_files:
        blocks.append(f[name][:, :, d0:d1 + 1].astype(float))
    a = np.array(blocks)
    a = np.pad(a, ((0, 0), (1, 1), (1, 1), (0, 0)), mode="edge")

    if is_velocity:                      # side walls (no slip)
        a[:, -1] = -((X[-1] - L) / (L - X[-2])) * a[:, -2]
        a[:, 0] = (X[0] / X[1]) * a[:, 1]
        a[:, :, -1] = -((Y[-1] - L) / (L - Y[-2])) * a[:, :, -2]
        a[:, :, 0] = (Y[0] / Y[1]) * a[:, :, 1]

    if k0 == 0:                          # ghost plane below the bottom plate
        if is_velocity:
            ghost = (Z[0] / Z[1]) * a[:, :, :, :1]
        else:
            ghost = a[:, :, :, :1]
        a = np.concatenate([ghost, a], axis=3)

    if k1 == len(Z) - 1:                 # ghost plane above the top plate
        if is_velocity:
            ghost = -((Z[-1] - L) / (L - Z[-2])) * a[:, :, :, -1:]
        else:
            ghost = a[:, :, :, -1:]
        a = np.concatenate([a, ghost], axis=3)
    return a


def dns_plane(u, v, w, P, k, k0, grid_x, grid_y, grid_z, Ro, zH):
    # The dictionary "dns" for plane k, from blocks read with read_dns_planes
    # (which start at the padded plane k0).
    m = k - k0                           # position of plane k inside the block
    dns = {}
    dns["Ro"] = Ro
    dns["zH"] = zH
    dns["k"] = k
    dns["u"] = u[:, :, :, m]
    dns["v"] = v[:, :, :, m]
    dns["dt"] = settings.DNS_DT
    dns["grid_x"] = grid_x
    dns["grid_y"] = grid_y
    dns["z_terms"] = z_derivatives(u[1][:, :, m - 1:m + 2], v[1][:, :, m - 1:m + 2],
                                   w[1][:, :, m - 1:m + 2], grid_z, k)
    dns["p_dns"] = P[1, :, :, m]
    dns["name"] = "DNS Ro=" + format(Ro, "g") + " z/H=" + format(zH, ".4f")
    return dns


def read_dns_h5(file_paths, z_target, Ro):
    # The plane nearest to z_target, from three consecutive .h5 snapshots
    import h5py
    files = []
    for path in file_paths:
        files.append(h5py.File(path, "r"))

    X, Y, Z = dns_coordinates(files[0])
    k = int(np.argmin(np.abs(Z - z_target)))
    grid_x = tanh_grid(X, settings.DNS_L, settings.DNS_BETA)
    grid_y = tanh_grid(Y, settings.DNS_L, settings.DNS_BETA)
    grid_z = tanh_grid(Z, settings.DNS_L, settings.DNS_BETA)

    u = read_dns_planes(files, "Vx", k - 1, k + 1, X, Y, Z)
    v = read_dns_planes(files, "Vy", k - 1, k + 1, X, Y, Z)
    w = read_dns_planes(files, "Vz", k - 1, k + 1, X, Y, Z)
    P = read_dns_planes(files, "P", k - 1, k + 1, X, Y, Z)
    for f in files:
        f.close()
    return dns_plane(u, v, w, P, k, k - 1, grid_x, grid_y, grid_z, Ro, float(Z[k]))
