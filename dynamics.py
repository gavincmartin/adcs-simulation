import numpy as np


def angular_velocity_derivative(J, w, M_list):
    """Computes the time derivative of a satellite's angular velocity (in rad/s^2 in body coordinates)

    Args:
        J (numpy ndarray): the satellite's inertia tensor (3x3) (kg * m^2)
        w (numpy ndarray): the angular velocity (rad/s) (3x1) in body coordinates of the spacecraft
        M_list (list): a list of (3x1) numpy ndarrays where each element in the list is a torque (N * m); an empty list signifies no torques

    Returns:
        [type]: [description]
    """
    M_total = np.sum(M_list, axis=0)
    J_inv = np.linalg.inv(J)
    if M_total == 0:
        return -np.matmul(J_inv, np.cross(w, np.matmul(J, w)))
    else:
        return (-np.matmul(J_inv, np.cross(w, np.matmul(J, w))) +
                np.matmul(J_inv, M_total))
