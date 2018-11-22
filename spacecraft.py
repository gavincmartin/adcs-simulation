import numpy as np
from math_utils import cross
from controller import PDController
from actuators import Actuators


class Spacecraft(object):
    def __init__(self,
                 J=np.diag([100, 100, 100]),
                 controller=PDController(
                     k_d=np.diag([.01, .01, .01]), k_p=np.diag([.1, .1, .1])),
                 sensors=None,
                 actuators=Actuators(
                     rxwl_mass=14,
                     rxwl_radius=0.1845,
                     rxwl_max_torque=0.68,
                     noise_factor=0.01),
                 q=np.array([0, 0, 0, 1]),
                 w=np.array([0, 0, 0])):
        """Constructs a Spacecraft object to store system objects, and state
        
        Args:
            J (numpy ndarray): the spacecraft's inertia tensor (3x3) (kg * m^2)
            controller (PDController): the controller that will compute control
                torques to meet desired pointing and angular velocity
                requirements
            sensors (---): ---
            actuators (Actuators): an object that stores reaction wheel state
                and related methods; actually applies control torques generated
                by the controller object
            q (numpy ndarray): the quaternion representing the attitude (from
                the inertial to body frame) of the spacecraft (at a given time)
            w (numpy ndarray): the angular velocity (rad/s) (3x1) in body
                coordinates of the spacecraft (at a given time)
        """
        self.J = J
        self.controller = controller
        self.sensors = sensors
        self.actuators = actuators
        self.q = q
        self.w = w

    def calculate_control_torques(self, attitude_err, attitude_rate_err):
        """Wrapper method for Controller.calculate_control_torques
        
        Args:
            attitude_err (numpy ndarray): the attitude error (3x1) of the
                spacecraft at a given time
            attitude_rate_err (numpy ndarray): the attitude rate error (3x1) of
                the spacecraft at a given time

        Returns:
            numpy ndarray: the control torque (3x1) produced by the PD
                controller (N * m) or [0, 0, 0] if there is no controller
        """
        if self.controller is None:
            return np.array([0, 0, 0])
        return self.controller.calculate_control_torques(
            attitude_err, attitude_rate_err, self.J)

    def apply_control_torques(self, M_ctrl, t, delta_t):
        """Wrapper method for Actuators.apply_control_torques
        
        Args:
            M_ctrl (numpy ndarray): the control torque (3x1) produced by the
                PD controller (N * m)
            w_sc (numpy ndarray): the angular velocity (rad/s) (3x1) in body
                coordinates of the spacecraft (at a given time)
            t (float): the current simulation time in seconds
            delta_t (float): the time between user-defined integrator steps
                (not the internal/adaptive integrator steps) in seconds
        
        Returns:
            numpy ndarray: the control moment (3x1) as actually applied on
                the reaction wheels (the input control torque with some
                Gaussian noise applied) (N * m)
            numpy ndarray: the angular acceleration of the 3 reaction wheels
                applied to achieve the applied torque (rad/s^2)
        """
        if self.actuators is None:
            return np.array([0, 0, 0]), np.array([0, 0, 0])
        return self.actuators.apply_control_torques(M_ctrl, self.w, t, delta_t)
