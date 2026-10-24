# -*- coding: utf-8 -*-
# =============================================================================
# settings.py
#
# All the fixed numbers of the project: folders, fluid properties, the PIV
# calibration and the DNS grid. Every other file imports what it needs from
# here, so a number only has to be changed in one place.
#
# The things that change from run to run (which folder, which Rossby number,
# which solver) are NOT here. They are at the top of the run_*.py scripts.
# =============================================================================

import os
import numpy as np


# -----------------------------------------------------------------------------
# Folders
# -----------------------------------------------------------------------------
# The data are not inside this code folder. By default they are in the folder
# that CONTAINS the code folder (PIV_PROJECT). Change DATA_FOLDER if needed.

CODE_FOLDER = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.dirname(CODE_FOLDER)

DNS_EXPORT_FOLDER = os.path.join(DATA_FOLDER, "DNS_export")           # small .npz exports
DNS_RESULTS_FOLDER = os.path.join(DNS_EXPORT_FOLDER, "results_pressure")
DNS_DRIVE_FOLDER = r"G:\My Drive\DNS Data sets"                       # the DNS .h5 files


# -----------------------------------------------------------------------------
# Water properties and the experiment (as in the PIV notebook)
# -----------------------------------------------------------------------------
g = 9.81           # gravity (m/s^2)
beta = 0.000303    # thermal expansion coefficient of water at 302 K (1/K)
mu = 0.000803      # dynamic viscosity (kg/m-s)
rho = 995.7        # density (kg/m^3)
dT = 4             # temperature difference between hot and cold plate (K)
H = 0.200          # height of the cell from the middle of the roughness elements (m)
nu = mu / rho      # kinematic viscosity (m^2/s)

L_ndim = H                              # length used to non-dimensionalise
U_ndim = np.sqrt(g * beta * dT * H)     # velocity used to non-dimensionalise
T_ndim = L_ndim / U_ndim                # time used to non-dimensionalise


# -----------------------------------------------------------------------------
# Dimensionless numbers (the same for the experiments and the DNS)
# -----------------------------------------------------------------------------
Pr = 5.63          # Prandtl number of water at 302 K
Ra = 7.23e8        # Rayleigh number


def ekman_number(Ro):
    # E = sqrt(Pr/Ra) * Ro
    E = np.sqrt(Pr / Ra) * Ro
    return E


def rotation_rate(Ro):
    # Angular velocity of the rotating table (rad/s), only printed for information
    E = ekman_number(Ro)
    Omega = nu / (2 * E * H * H)
    return Omega


# Sense of rotation. It sets the sign of the Coriolis terms.
#   The experiments rotate CLOCKWISE, the DNS rotates ANTICLOCKWISE.
CLOCKWISE = -1
ANTICLOCKWISE = +1


# -----------------------------------------------------------------------------
# PIV calibration
# -----------------------------------------------------------------------------
px_to_m = 1.0 / (32 * 1000)    # 32 px = 1 mm
dt_frame = 0.1                 # s between image A and image B of one PIV pair
dt_fields = 0.1                # s between two consecutive velocity fields (for d/dt)


# -----------------------------------------------------------------------------
# DNS
# -----------------------------------------------------------------------------
DNS_L = 1.0        # size of the DNS box in x, y and z (dimensionless)
DNS_BETA = 1.3     # stretching parameter of the tanh grid
DNS_DT = 0.1       # time between the three DNS snapshots (dimensionless)


# -----------------------------------------------------------------------------
# Pressure solve
# -----------------------------------------------------------------------------
MARGIN = 3         # the pressure is solved from MARGIN points in from every edge
                   # (index 3 to N-4), the ghost points are at index 2 and N-3,
                   # exactly as in the notebook: X_jacobi[3:-3, 3:-3]
