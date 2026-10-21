# -*- coding: utf-8 -*-
"""
error_measures.py
=================

How far a reconstructed pressure is from a reference pressure.
"""

import numpy as np


def rms_error_percent(p_ref, p_test, crop=3):
    """RMS difference in % of the reference: the definition used in the thesis.

    Both fields are taken over the solved points [crop:-crop, crop:-crop] and
    each is shifted by its own minimum (the pressure is only defined up to a
    constant), then

        error = RMS(p_test - p_ref) / max(p_ref) * 100.
    """
    a = p_ref[crop:-crop, crop:-crop]
    b = p_test[crop:-crop, crop:-crop]
    a = a - a.min()
    b = b - b.min()
    return float(np.sqrt(np.mean((a - b) ** 2)) / a.max() * 100)
