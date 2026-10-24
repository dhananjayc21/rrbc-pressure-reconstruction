# -*- coding: utf-8 -*-
# =============================================================================
# error_measures.py
#
# How far a reconstructed pressure is from a reference pressure (the
# definition used in the thesis):
#
#   both fields are taken over the solved points [3:-3, 3:-3], each is
#   shifted by its own minimum (the pressure is only known up to a constant),
#   and then
#
#       error = RMS(p_test - p_ref) / max(p_ref) * 100        (in %)
# =============================================================================

import numpy as np

import settings


def rms_error_percent(p_ref, p_test):
    m = settings.MARGIN
    a = p_ref[m:-m, m:-m]
    b = p_test[m:-m, m:-m]

    a = a - np.min(a)
    b = b - np.min(b)

    error = np.sqrt(np.mean((a - b) ** 2)) / np.max(a) * 100
    return error
