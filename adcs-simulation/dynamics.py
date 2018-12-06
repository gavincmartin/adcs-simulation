# -*- coding: utf-8 -*-
"""Dynamics module for attitude determination and control system.

This module models the evolution of the angular velocity of a spacecraft given
a set of external torques, the spacecraft inertia tensor, and its current
angular velocity.
"""

import numpy as np
from math_utils import cross


def angular_velocity_derivative(J, w, M_list=[]):
    """Computes the time derivative of a satellite's angular velocity (in rad/s^2 in body coordinates)

    Args:
        J (numpy ndarray): the satellite's inertia tensor (3x3) (kg * m^2)
        w (numpy ndarray): the angular velocity (rad/s) (3x1) in body
            coordinates of the spacecraft
        M_list (list): a list of (3x1) numpy ndarrays where each element in the
            list is a torque (N * m); an empty list signifies no torques

    Returns:
        numpy ndarray: the angular acceleration (3x1) produced by the sum of 
            the torques on the spacecraft
    """
    # if the list of torques is empty, the total torque is [0, 0, 0]
    if len(M_list) == 0:
        M_total = np.array([0, 0, 0])
    # else, sum the torques
    else:
        M_total = np.sum(M_list, axis=0)
    J_inv = np.linalg.inv(J)
    return (-np.matmul(J_inv, cross(w, np.matmul(J, w))) +
            np.matmul(J_inv, M_total))
