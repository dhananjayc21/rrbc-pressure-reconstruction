# -*- coding: utf-8 -*-
"""
equations.py
============

The physics: from the velocity on one plane, compute

  1. the pressure gradient dp/dx, dp/dy from the momentum equations
     (these are the Neumann boundary values of the Poisson equation), and
  2. the right-hand side of the pressure Poisson equation.

Non-dimensional equations of Chapter 3 of the thesis, with
    nu* = sqrt(Pr/Ra)            (viscous coefficient)
    C   = sqrt(Pr/(Ra E^2)) = 1/Ro   (Coriolis coefficient)
    s   = +1 for anticlockwise rotation (DNS), -1 for clockwise (experiments)

Momentum equations, solved for the pressure gradient:

    dp/dx = -(du/dt + u du/dx + v du/dy + w du/dz) + nu* lap(u) + s C v
    dp/dy = -(dv/dt + u dv/dx + v dv/dy + w dv/dz) + nu* lap(v) - s C u

Pressure Poisson equation (horizontal divergence of the above):

    d2p/dx2 + d2p/dy2 = term1 + term2 + term3 + term4 + term5

    term1 = -(ux^2 + 2 uy vx + vy^2)                       velocity gradients
    term2 = -(dD/dt + u dD/dx + v dD/dy)                    in-plane divergence D
    term3 = nu* (d2D/dx2 + d2D/dy2)                         viscous
    term4 = -(wx uz + wy vz + w dD/dz) + nu* d2D/dz2        out-of-plane (DNS only)
    term5 = s C omega_z                                     Coriolis

with D = du/dx + dv/dy and omega_z = dv/dx - du/dy.

Term 4 and the w-terms in dp/dx, dp/dy need w and z-derivatives. Planar PIV
does not have them, so for PIV they are always left out ("XY-only"). For the
DNS they can be kept (use_z_terms=True, "full 3D") or left out.
"""

import settings
from derivatives import d_dx, d_dy, d2_dx2, d2_dy2


def pressure_equation_terms(plane, Ro, rotation_sign, use_z_terms=True):
    """Boundary values and right-hand side for one plane at time t.

    plane: dictionary from load_piv.py or load_dns.py
    Returns a dictionary with the 2-D arrays
        dp_dx, dp_dy   pressure gradient (Neumann boundary values)
        rhs            right-hand side of the Poisson equation
        vorticity      omega_z = dv/dx - du/dy
        divergence     D = du/dx + dv/dy
    """
    gx, gy = plane["grid_x"], plane["grid_y"]
    nu = settings.viscous_coefficient()
    C = settings.coriolis_coefficient(Ro)
    s = rotation_sign

    # velocity at time t, and its time derivative (centred difference)
    u = plane["u"][1]
    v = plane["v"][1]
    du_dt = (plane["u"][2] - plane["u"][0]) / (2 * plane["dt"])
    dv_dt = (plane["v"][2] - plane["v"][0]) / (2 * plane["dt"])

    # space derivatives at time t
    du_dx, du_dy = d_dx(u, gx), d_dy(u, gy)
    dv_dx, dv_dy = d_dx(v, gx), d_dy(v, gy)
    lap_u = d2_dx2(u, gx) + d2_dy2(u, gy)
    lap_v = d2_dx2(v, gx) + d2_dy2(v, gy)

    divergence = du_dx + dv_dy
    vorticity = dv_dx - du_dy

    # ---- momentum equations -> pressure gradient (in-plane part) ------------
    dp_dx = -(du_dt + u * du_dx + v * du_dy) + nu * lap_u + s * C * v
    dp_dy = -(dv_dt + u * dv_dx + v * dv_dy) + nu * lap_v - s * C * u

    # ---- right-hand side of the Poisson equation ----------------------------
    term1 = -(du_dx ** 2 + 2 * du_dy * dv_dx + dv_dy ** 2)
    # dD/dt is written as d/dx(du/dt) + d/dy(dv/dt): exactly the same thing
    # (derivatives are linear), but only needs space derivatives at time t
    term2 = -((d_dx(du_dt, gx) + d_dy(dv_dt, gy))
              + u * d_dx(divergence, gx) + v * d_dy(divergence, gy))
    term3 = nu * (d2_dx2(divergence, gx) + d2_dy2(divergence, gy))
    term5 = s * C * vorticity
    rhs = term1 + term2 + term3 + term5

    # ---- term 4: w and z-derivatives (DNS only) -----------------------------
    z = plane["z_terms"]
    if use_z_terms and z is not None:
        w = z["w"]
        dp_dx = dp_dx - w * z["du_dz"] + nu * z["d2u_dz2"]
        dp_dy = dp_dy - w * z["dv_dz"] + nu * z["d2v_dz2"]
        dD_dz = d_dx(z["du_dz"], gx) + d_dy(z["dv_dz"], gy)
        d2D_dz2 = d_dx(z["d2u_dz2"], gx) + d_dy(z["d2v_dz2"], gy)
        term4 = (-(d_dx(w, gx) * z["du_dz"] + d_dy(w, gy) * z["dv_dz"] + w * dD_dz)
                 + nu * d2D_dz2)
        rhs = rhs + term4

    return {"dp_dx": dp_dx, "dp_dy": dp_dy, "rhs": rhs,
            "vorticity": vorticity, "divergence": divergence}
