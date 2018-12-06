# -*- coding: utf-8 -*-
"""Spacecraft module for attitude determination and control system.

This module stores a number of composed objects that describe the overall
spacecraft and its associated controller, gyros, etc. It is primarily used to
simplify the process of passing parameters between methods, and it has aliases
for a number of methods of its sub-objects for simplicity.
"""

import numpy as np
from math_utils import cross, triad, dcm_to_quaternion, normalize, quaternion_to_dcm
from sensor_models import compute_earth_direction, compute_local_magnetic_field
from controller import PDController
from actuators import Actuators
from perturbations import gravity_gradient_torque, magnetic_field_torque


class Spacecraft(object):
    """A class to store system objects and state over time
        
        Args:
            J (numpy ndarray): the spacecraft's inertia tensor (3x3) (kg * m^2)
            controller (PDController): the controller that will compute control
                torques to meet desired pointing and angular velocity
                requirements
            gyros (Gyros): an object that models gyroscopes and simulates
                estimated angular velocity by introducing bias and noise to
                angular velocity measurements
            actuators (Actuators): an object that stores reaction wheel state
                and related methods; actually applies control torques generated
                by the controller object
            dipole (numpy ndarray): the spacecraft's residual magnetic dipole
                vector (A * m^2)
            q (numpy ndarray): the quaternion representing the attitude (from
                the inertial to body frame) of the spacecraft (at a given time)
            w (numpy ndarray): the angular velocity (rad/s) (3x1) in body
                coordinates of the spacecraft (at a given time)
            r (numpy ndarray): the inertial position of the spacecraft (m)
            v (numpy ndarray): the inertial velocity of the spacecraft (m/s)
        
        Attributes:
            J (numpy ndarray): the spacecraft's inertia tensor (3x3) (kg * m^2)
            controller (PDController): the controller that will compute control
                torques to meet desired pointing and angular velocity
                requirements
            gyros (Gyros): an object that models gyroscopes and simulates
                estimated angular velocity by introducing bias and noise to
                angular velocity measurements
            actuators (Actuators): an object that stores reaction wheel state
                and related methods; actually applies control torques generated
                by the controller object
            dipole (numpy ndarray): the spacecraft's residual magnetic dipole
                vector (A * m^2)
            q (numpy ndarray): the quaternion representing the attitude (from
                the inertial to body frame) of the spacecraft (at a given time)
            w (numpy ndarray): the angular velocity (rad/s) (3x1) in body
                coordinates of the spacecraft (at a given time)
            r (numpy ndarray): the inertial position of the spacecraft (m)
            v (numpy ndarray): the inertial velocity of the spacecraft (m/s)
    """

    def __init__(self,
                 J=np.diag([100, 100, 100]),
                 controller=PDController(
                     k_d=np.diag([.01, .01, .01]), k_p=np.diag([.1, .1, .1])),
                 gyros=None,
                 magnetometer=None,
                 earth_horizon_sensor=None,
                 actuators=Actuators(
                     rxwl_mass=14,
                     rxwl_radius=0.1845,
                     rxwl_max_torque=0.68,
                     noise_factor=0.01),
                 dipole=np.array([0, 0, 0]),
                 q=np.array([0, 0, 0, 1]),
                 w=np.array([0, 0, 0]),
                 r=np.array([0, 0, 0]),
                 v=np.array([0, 0, 0])):
        """Constructs a Spacecraft object to store system objects, and state
        
        Args:
            J (numpy ndarray): the spacecraft's inertia tensor (3x3) (kg * m^2)
            controller (PDController): the controller that will compute control
                torques to meet desired pointing and angular velocity
                requirements
            gyros (Gyros): an object that models gyroscopes and simulates
                estimated angular velocity by introducing bias and noise to
                angular velocity measurements
            actuators (Actuators): an object that stores reaction wheel state
                and related methods; actually applies control torques generated
                by the controller object
            dipole (numpy ndarray): the spacecraft's residual magnetic dipole
                vector (A * m^2)
            q (numpy ndarray): the quaternion representing the attitude (from
                the inertial to body frame) of the spacecraft (at a given time)
            w (numpy ndarray): the angular velocity (rad/s) (3x1) in body
                coordinates of the spacecraft (at a given time)
            r (numpy ndarray): the inertial position of the spacecraft (m)
            v (numpy ndarray): the inertial velocity of the spacecraft (m/s)
        """
        self.J = J
        self.controller = controller
        self.gyros = gyros
        self.magnetometer = magnetometer
        self.earth_horizon_sensor = earth_horizon_sensor
        self.actuators = actuators
        self.dipole = dipole
        self.q = q
        self.w = w
        self.r = r
        self.v = v

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

    def estimate_angular_velocity(self, t, delta_t):
        """Wrapper method for gyros.estimate_angular_velocity
        
        Args:
            t (float): the current simulation time in seconds
            delta_t (float): the time between user-defined integrator steps
                (not the internal/adaptive integrator steps) in seconds
        
        Returns:
            numpy ndarray: the estimated angular velocity in rad/s
        """
        if self.gyros is None:
            return self.w
        else:
            return self.gyros.estimate_angular_velocity(self.w, t, delta_t)

    def estimate_attitude(self, t, delta_t):
        """Provides an estimated attitude (adding measurement noise to the actual)

        This method uses the TRIAD algorithm to compute the attitude from
        two direction measurements.
        
        Args:
            t (float): the current simulation time in seconds
            delta_t (float): the time between user-defined integrator steps
                (not the internal/adaptive integrator steps) in seconds
        
        Returns:
            numpy ndarray: the estimated DCM representing the attitude (from
                the inertial to body frame) of the spacecraft (at a given time)
        """
        if self.magnetometer is None or self.earth_horizon_sensor is None:
            return self.q
        else:
            d_earth_body = self.earth_horizon_sensor.estimate_earth_direction(
                self.q, self.r, t, delta_t)
            d_earth_inertial = compute_earth_direction(self.r)
            B_body = self.magnetometer.estimate_magnetic_field(
                self.q, self.r, t, delta_t)
            B_inertial = compute_local_magnetic_field(self.r)
            T_estimated = triad(d_earth_body, d_earth_inertial, B_body,
                                B_inertial)
            return T_estimated

    def approximate_gravity_gradient_torque(self):
        """Computes the gravity gradient torque acting on a spacecraft

        Returns:
            numpy ndarray: the gravity gradient torque acting on the spacecraft
                coordinatized in the body frame
        """
        return gravity_gradient_torque(self.r, self.J)

    def approximate_magnetic_field_torque(self):
        """Computes the magnetic field torque acting on a spacecraft
        
        Returns:
            numpy ndarray: the magnetic field torque acting on the spacecraft
                coordinatized in the body frame
        """
        return magnetic_field_torque(self.r, self.dipole,
                                     quaternion_to_dcm(self.q))
