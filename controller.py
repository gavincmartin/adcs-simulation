import numpy as np


def calculate_PD_control_torque(attitude_err, attitude_rate_err, k_d, k_p, J):
    """Calculates the control torques supplied by a PD controller given input errors and gains

    Args:
        attitude_err (numpy ndarray): the attitude error (3x1) of the spacecraft at a given time
        attitude_rate_err (numpy ndarray): the attitude rateerror (3x1) of the spacecraft at a given time
        k_d (numpy ndarray): the gains matrix (3x3) for the derivative control
        k_p (numpy ndarray): the gains matrix (3x3) for the proportional control
        J (numpy ndarray): the satellite's inertia tensor (3x3) (kg * m^2)

    Returns:
        numpy ndarray: the control torque (3x1) produced by the PD controller (N * m)
    """
    u = -np.matmul(k_d, attitude_rate_err) - np.matmul(k_p, attitude_err)
    return np.matmul(J, u)