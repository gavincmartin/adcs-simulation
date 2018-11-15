import numpy as np
from math_utils import quaternion_multiply, cross


def calculate_attitude_error(q_desired, q_current):
    """Computes the attitude error of a spacecraft at a given time

    Args:
        q_desired (numpy ndarray): the quaternion that represents the nominal attitude (from the inertial to body frame) of the spacecraft at a given time
        q_current (numpy ndarray): the quaternion that represents the current attitude (from the inertial to body frame) of the spacecraft at a given time

    Returns:
        numpy ndarray: the attitude error (3x1) of the system at the given time
    """
    q_desired_conjugate = np.array([*(-q_desired[0:3]), q_desired[3]])
    delta_q = quaternion_multiply(q_current, q_desired_conjugate)
    attitude_err = 2 * delta_q[0:3] / delta_q[3]
    return attitude_err


def calculate_attitude_rate_error(w_desired, w_current, attitude_err):
    """Computes the attitude rate error of a spacecraft at a given time

    Args:
        w_desired (numpy ndarray): the nominal angular velocity (rad/s) (3x1) in body coordinates of the spacecraft at a given time
        w_current (numpy ndarray): the current angular velocity (rad/s) (3x1) in body coordinates of the spacecraft at a given time
        attitude_err (numpy ndarray): the attitude error (3x1) of the spacecraft at a given time

    Returns:
        attitude_rate_error: the attitude rate error (3x1) of the system at the given time
    """
    delta_w = w_current - w_desired
    attitude_rate_err = -cross(w_current, attitude_err) + delta_w
    return attitude_rate_err
