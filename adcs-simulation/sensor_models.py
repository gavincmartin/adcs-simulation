# -*- coding: utf-8 -*-
"""Sensor Models module for attitude determination and control system.

This module implements models for sensor estimates in inertial coordinates
that are subsequently used by the actual sensor simulation in `sensors.py`.
"""

import numpy as np
from math_utils import get_DCM_i2NED, normalize


def compute_local_magnetic_field(r):
    """Computes the local magnetic field using a simple dipole model
    
    Args:
        r (numpy ndarray): the inertial position of the spacecraft (m)
    
    Returns:
        numpy ndarray: the vector representing the local magnetic field
            measured in inertial coordinates
    """
    r_mag = np.linalg.norm(r)
    B_0 = 3.12e-5
    R_e = 6.3781e6
    DCM_i2n = get_DCM_i2NED(r)
    latitude = np.arcsin(r[2] / r_mag)
    B_n = (B_0 * (R_e / r_mag)**3 * np.array(
        [np.cos(latitude), 0, -2 * np.sin(latitude)]))
    B_i = np.matmul(np.transpose(DCM_i2n), B_n)
    return B_i


def compute_earth_direction(r):
    """Estimates the direction vector to the Earth
        
        Args:
            r (numpy ndarray): the inertial position of the spacecraft (m)
        
        Returns:
            numpy ndarray: the direction vector to the Earth as measured by the
                sensor in inertial coordinates
        """
    return normalize(-r)