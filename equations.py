# -*- coding: utf-8 -*-
# =============================================================================
# equations.py
#
# The physics, as in the PIV notebook. From the velocity on one plane:
#
#   1. dp/dx and dp/dy from the momentum equations
#      (these give the Neumann boundary conditions of the pressure), and
#   2. B, the right-hand side of the pressure Poisson equation
#
#          d2p/dx2 + d2p/dy2 = B = term1 + term2 + term3 + term4 + term5
#
#      term1 = -(du/dx^2 + 2 du/dy dv/dx + dv/dy^2)              velocity gradients
#      term2 = -(dDiv/dt + u dDiv/dx + v dDiv/dy)                 in-plane divergence
#      term3 = sqrt(Pr/Ra) (d2Div/dx2 + d2Div/dy2)                viscous
#      term4 = w and z-derivatives                                 (DNS only)
#      term5 = rotation * sqrt(Pr/(Ra E^2)) * Vor                 Coriolis
#
#   with Div_xy = du/dx + dv/dy and Vor = dv/dx - du/dy.
#
# sqrt(Pr/(Ra E^2)) = 1/Ro. rotation = settings.CLOCKWISE (= -1) for the
# experiments and settings.ANTICLOCKWISE (= +1) for the DNS: it sets the sign
# of the Coriolis terms, exactly as "-np.sqrt(Pr/(Ra*E*E))" in the notebook.
#
# Planar PIV has no w and no z-derivatives, so term 4 is always 0 for PIV
# ("XY-only"). For the DNS it can be kept ("full 3D") or left out.
# =============================================================================

import numpy as np

import settings
from derivatives import d_dx, d_dy, d2_dx2, d2_dy2


def pressure_equation(u, v, dt, grid_x, grid_y, Ro, rotation, z_terms=None):
    # u, v: velocity at three times, shape (3, Nx, Ny). t = 1 is the middle one.
    # z_terms: None for PIV; the dictionary of read_dns.z_derivatives for DNS
    #          (give None to leave term 4 out, also for DNS).
    #
    # Returns dp_dx, dp_dy, B, Vor, Div_xy (all of shape (Nx, Ny)).

    E = settings.ekman_number(Ro)
    viscous = np.sqrt(settings.Pr / settings.Ra)                    # sqrt(Pr/Ra)
    coriolis = np.sqrt(settings.Pr / (settings.Ra * E * E))         # = 1/Ro

    t = 1

    # time derivatives (central difference)
    du_dt = (u[t + 1] - u[t - 1]) / (2 * dt)
    dv_dt = (v[t + 1] - v[t - 1]) / (2 * dt)

    # space derivatives at time t
    du_dx = d_dx(u[t], grid_x)
    dv_dx = d_dx(v[t], grid_x)
    du_dy = d_dy(u[t], grid_y)
    dv_dy = d_dy(v[t], grid_y)

    d2u_dx2 = d2_dx2(u[t], grid_x)
    d2v_dx2 = d2_dx2(v[t], grid_x)
    d2u_dy2 = d2_dy2(u[t], grid_y)
    d2v_dy2 = d2_dy2(v[t], grid_y)

    Div_xy = du_dx + dv_dy        # in-plane divergence
    Vor = dv_dx - du_dy           # vertical vorticity

    # ---- dp/dx and dp/dy (momentum equations) ------------------------------
    term1 = -(du_dt + u[t] * du_dx + v[t] * du_dy)
    term2 = viscous * (d2u_dx2 + d2u_dy2)
    term3 = rotation * coriolis * v[t]
    dp_dx = term1 + term2 + term3

    term1 = -(dv_dt + u[t] * dv_dx + v[t] * dv_dy)
    term2 = viscous * (d2v_dx2 + d2v_dy2)
    term3 = -rotation * coriolis * u[t]
    dp_dy = term1 + term2 + term3

    # ---- right-hand side B of the Poisson equation ------------------------
    term1 = -(du_dx ** 2 + 2 * du_dy * dv_dx + dv_dy ** 2)

    # dDiv/dt is written as d/dx(du/dt) + d/dy(dv/dt) (the same thing, as in
    # the notebook), so only space derivatives at time t are needed
    term2 = -((d_dx(du_dt, grid_x) + d_dy(dv_dt, grid_y))
              + u[t] * d_dx(Div_xy, grid_x) + v[t] * d_dy(Div_xy, grid_y))

    term3 = viscous * (d2_dx2(Div_xy, grid_x) + d2_dy2(Div_xy, grid_y))

    term5 = rotation * coriolis * Vor

    B = term1 + term2 + term3 + term5

    # ---- term 4: w and the z-derivatives (DNS only) ------------------------
    if z_terms is not None:
        w = z_terms["w"]
        du_dz = z_terms["du_dz"]
        dv_dz = z_terms["dv_dz"]
        d2u_dz2 = z_terms["d2u_dz2"]
        d2v_dz2 = z_terms["d2v_dz2"]

        dp_dx = dp_dx - w * du_dz + viscous * d2u_dz2
        dp_dy = dp_dy - w * dv_dz + viscous * d2v_dz2

        dDiv_dz = d_dx(du_dz, grid_x) + d_dy(dv_dz, grid_y)
        d2Div_dz2 = d_dx(d2u_dz2, grid_x) + d_dy(d2v_dz2, grid_y)
        term4 = (-(d_dx(w, grid_x) * du_dz + d_dy(w, grid_y) * dv_dz + w * dDiv_dz)
                 + viscous * d2Div_dz2)
        B = B + term4

    return dp_dx, dp_dy, B, Vor, Div_xy
