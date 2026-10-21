# -*- coding: utf-8 -*-
"""
settings.py
===========

All the fixed numbers of the project in one place: the flow parameters, the
PIV calibration, the DNS grid and the folders. Every other file imports what
it needs from here, so a number is changed in one place only.

The numbers that change from run to run (which folder, which Rossby number,
which solver) are NOT here; they are at the top of the run_*.py scripts.
"""

import os
import numpy as np


# -----------------------------------------------------------------------------
# Folders
# -----------------------------------------------------------------------------
# The data are NOT part of this code: they live in DATA_FOLDER, which by
# default is the folder that contains this code folder. Change DATA_FOLDER if
# your data are somewhere else. Results are written next to the data, so the
# code folder stays clean.
CODE_FOLDER = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.dirname(CODE_FOLDER)
DNS_EXPORT_FOLDER = os.path.join(DATA_FOLDER, "DNS_export")      # the small .npz exports
DNS_RESULTS_FOLDER = os.path.join(DNS_EXPORT_FOLDER, "results_pressure")
DNS_DRIVE_FOLDER = r"G:\My Drive\DNS Data sets"                 # the DNS .h5 files


# -----------------------------------------------------------------------------
# Flow parameters (the same for the experiments and the DNS)
# -----------------------------------------------------------------------------
RA = 7.23e8          # Rayleigh number
PR = 5.63            # Prandtl number of water at about 302 K

# Sense of rotation. It decides the sign of the Coriolis term:
#   the DNS rotates anticlockwise, the experiments rotate clockwise.
ANTICLOCKWISE = +1
CLOCKWISE = -1


def ekman_number(Ro):
    """E = sqrt(Pr/Ra) * Ro."""
    return np.sqrt(PR / RA) * Ro


def viscous_coefficient():
    """Coefficient of the viscous terms, sqrt(Pr/Ra)."""
    return np.sqrt(PR / RA)


def coriolis_coefficient(Ro):
    """Coefficient of the Coriolis terms, sqrt(Pr/(Ra E^2)). This equals 1/Ro."""
    return np.sqrt(PR / (RA * ekman_number(Ro) ** 2))


# -----------------------------------------------------------------------------
# Experiment (PIV): water at about 302 K, cell height H
# -----------------------------------------------------------------------------
G = 9.81                 # gravity, m/s^2
BETA = 0.000303          # thermal expansion coefficient of water, 1/K
DELTA_T = 4.0            # temperature difference between the plates, K
H = 0.200                # height from mid-roughness to the top plate, m
MU = 0.000803            # dynamic viscosity, kg/(m s)
RHO = 995.7              # density, kg/m^3

U_REF = np.sqrt(G * BETA * DELTA_T * H)     # free-fall velocity, m/s
T_REF = H / U_REF                           # free-fall time, s

PX_PER_MM = 32.0          # PIV image calibration
M_PER_PX = 1.0 / (PX_PER_MM * 1000.0)
IMAGE_DT = 0.1            # s between images A and B of one PIV pair (for the velocity)
SNAPSHOT_DT = 0.1         # s between two consecutive velocity fields (for d/dt)


def rotation_rate(Ro):
    """Angular velocity of the rotating table, rad/s (for information only)."""
    nu = MU / RHO
    return nu / (2 * ekman_number(Ro) * H ** 2)


# -----------------------------------------------------------------------------
# DNS
# -----------------------------------------------------------------------------
DNS_L = 1.0               # domain size in x, y and z (non-dimensional)
DNS_BETA = 1.3            # stretching parameter of the tanh grid
DNS_DT = 0.1              # time between the three DNS snapshots (non-dimensional)
