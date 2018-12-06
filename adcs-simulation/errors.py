# -*- coding: utf-8 -*-
"""Errors module for attitude determination and control system.

This module computes the attitude and attitude rate errors given the estimated
and desired states (attitude and angular velocity) at a given time.
"""

import numpy as np
from math_utils import quaternion_multiply, cross


def calculate_attitude_error(DCM_desired, DCM_estimated):
    """Computes the attitude error of a spacecraft at a given time

    Args:
        DCM_desired (numpy ndarray): the DCM that represents the nominal attitude (from the inertial to body frame) of the spacecraft at a given time
        DCM_estimated (numpy ndarray): the DCM that represents the estimated attitude (from the inertial to body frame) of the spacecraft at a given time

    Returns:
        numpy ndarray: the attitude error (3x1) of the system at the given time
    """
    delta_DCM = np.matmul(DCM_estimated, np.transpose(DCM_desired))
    attitude_err = -0.5 * np.array([
        delta_DCM[2, 1] - delta_DCM[1, 2],
        delta_DCM[0, 2] - delta_DCM[2, 0],
        delta_DCM[1, 0] - delta_DCM[0, 1],
    ])
    return attitude_err


def calculate_attitude_rate_error(w_desired, w_estimated, attitude_err):
    """Computes the attitude rate error of a spacecraft at a given time

    Args:
        w_desired (numpy ndarray): the nominal angular velocity (rad/s) (3x1) in body coordinates of the spacecraft at a given time
        w_estimated (numpy ndarray): the estimated angular velocity (rad/s) (3x1) in body coordinates of the spacecraft at a given time
        attitude_err (numpy ndarray): the attitude error (3x1) of the spacecraft at a given time

    Returns:
        attitude_rate_error: the attitude rate error (3x1) of the system at the given time
    """
    delta_w = w_estimated - w_desired
    attitude_rate_err = -cross(w_estimated, attitude_err) + delta_w
    return attitude_rate_err
