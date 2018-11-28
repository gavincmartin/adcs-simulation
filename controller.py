import numpy as np


class PDController(object):
    def __init__(self, k_d, k_p):
        """Constructs a Proportional Derivative Controller object

        Args:
            k_d (numpy ndarray): the gains matrix (3x3) for the derivative
                control
            k_p (numpy ndarray): the gains matrix (3x3) for the proportional
                control
        """
        self.k_d = k_d
        self.k_p = k_p

    def calculate_control_torques(self, attitude_err, attitude_rate_err, J):
        """Calculates the necessary control torques given input errors

        Args:
            attitude_err (numpy ndarray): the attitude error (3x1) of the
                spacecraft at a given time
            attitude_rate_err (numpy ndarray): the attitude rate error (3x1) of
                the spacecraft at a given time
            J (numpy ndarray): the spacecraft's inertia tensor (3x3) (kg * m^2)

        Returns:
            numpy ndarray: the control torque (3x1) produced by the PD
                controller (N * m)
        """
        u = -np.matmul(self.k_d, attitude_rate_err) - np.matmul(
            self.k_p, attitude_err)
        return np.matmul(J, u)
