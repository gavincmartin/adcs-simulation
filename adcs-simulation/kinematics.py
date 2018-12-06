# -*- coding: utf-8 -*-
"""Kinematics module for attitude determination and control system.

This module models the kinematic evolution of the quaternion describing 
spacecraft attitude over time as a function of its current attitude and
angular velocity.
"""

import numpy as np
from math_utils import quaternion_multiply


def quaternion_derivative(q, w):
    """Computes the time derivative of a satellite's attitude (in quaternion form)

    Args:
        q (numpy ndarray): the quaternion representing the attitude (from the inertial to body frame) of the spacecraft
        w (numpy ndarray): the angular velocity (rad/s) (3x1) in body coordinates of the spacecraft

    Returns:
        numpy ndarray: the time derivative of the quaternion (4x1)
    """
    q_w = np.array([*w, 0])
    return 0.5 * quaternion_multiply(q_w, q)