# -*- coding: utf-8 -*-
"""Sensors module for attitude determination and control system.

This module models the behavior of a suite of sensors and contains classes
that abstract gyroscopes, Earth horizon sensors, and magnetometers in order
to produce attitude and angular velocity estimates.
"""
import numpy as np
from scipy.stats import norm
from math_utils import normalize, quaternion_to_dcm, skew_symmetric, cross
from sensor_models import compute_local_magnetic_field, compute_earth_direction


class Gyros(object):
    """A class to store gyroscope parameters and methods
    
    Args:
        bias_stability (float): the bias stability of the gyro in deg/hr
        angular_random_walk (float): the angular random walk of the gyro
            in deg/sqrt(hr)
    
    Attributes:
        bias (numpy ndarray): the biases of the gyros throughout the duration
            of a simulation (3x1)
        noise_vals (numpy ndarray): a cache of pre-generated noise values
                to aid in the addition of noise
    """

    def __init__(self, bias_stability, angular_random_walk):
        """Constructs a set of gyroscopes to estimate angular velocity
        
        Args:
            bias_stability (float): the bias stability of the gyro in deg/hr
            angular_random_walk (float): the angular random walk of the gyro
                in deg/sqrt(hr)
        """
        # convert from deg/hr to rad/s
        bias_stability = bias_stability * np.pi / 180 / 3600

        # set bias to some random value for the simulation duration
        distribution = norm(loc=0.0, scale=bias_stability)
        self.bias = distribution.rvs(size=3)

        # convert angular random walk from deg/sqrt(hr) to rad/sqrt(s)
        # generate a distribution centered at 0 with a sigma equal to ARW
        # NOTE: you will have to post-divide by sqrt(delta_t) to get the noise
        ARW = angular_random_walk * np.pi / 180 / 60
        noise_func = norm(loc=0.0, scale=ARW)
        self.noise_vals = np.array([
            noise_func.rvs(size=100000),
            noise_func.rvs(size=100000),
            noise_func.rvs(size=100000)
        ])

    def estimate_angular_velocity(self, w_actual, t, delta_t):
        """Provides an estimated angular velocity (adding noise & bias to the actual)
        
        Args:
            w_actual (numpy ndarray): the actual angular velocity in rad/s
            t (float): the current simulation time in seconds
            delta_t (float): the time between user-defined integrator steps
                (not the internal/adaptive integrator steps) in seconds
        
        Returns:
            numpy ndarray: the estimated angular velocity in rad/s
        """
        return self.bias + w_actual + self.get_noise(t, delta_t)

    def get_noise(self, t, delta_t):
        """Gets the Gaussian noise for a set of x, y, and z gyro measurements

        NOTE: This method uses a cache of random values generated in this
              class's constructor. This is done to (1) reduce the overhead
              of many individual `rvs` calls and (2) ensure that all adaptive
              integrator steps in between user-defined steps use the same
              noise value (so that the dynamics are not constantly changing).
              Without this, the integrator fails to move forward in time.
              Therefore, a custom hash function is used to apply the same
              noise is necessary cases.
        
        Args:
            t (float): the current simulation time in seconds
            delta_t (float): the time between user-defined integrator steps
                (not the internal/adaptive integrator steps) in seconds
        
        Returns:
            numpy ndarray: the noise vector (noise in x, y, and z)
        """
        # randomly apply some noise, but use a cache so that each
        # integrator-defined adaptive integration step between user-defined
        # integration steps uses the same noise value (otherwise the
        # propagator fails)
        # NOTE: we have to post-divide by sqrt(delta_t) to get the noise
        # based on how we defined the distribution in the constructor
        return np.array([
            self.noise_vals[0, int(t // delta_t) % len(self.noise_vals[0])],
            self.noise_vals[1, int(t // delta_t) % len(self.noise_vals[1])],
            self.noise_vals[2, int(t // delta_t) % len(self.noise_vals[2])]
        ]) / np.sqrt(delta_t)


class Magnetometer(object):
    """A class to store magnetometer parameters and methods
    
    Args:
        resolution (float): the resolution of the magnetometer in Tesla;
            a resolution of 0 T means no measurement noise (it is a perfect
            sensor)
    
    Attributes:
        noise_vals (numpy ndarray): a cache of pre-generated noise values
                to aid in the addition of noise
    """

    def __init__(self, resolution=0):
        """Constructs a magnetometer to measure the local magnetic field
        
        Args:
            resolution (float): the resolution of the magnetometer in Tesla;
                a resolution of 0 T means no measurement noise (it is a perfect
                sensor)
        """
        noise_func = norm(loc=0.0, scale=resolution)
        self.noise_vals = np.array([
            noise_func.rvs(size=100000),
            noise_func.rvs(size=100000),
            noise_func.rvs(size=100000)
        ])

    def estimate_magnetic_field(self, q_actual, r, t, delta_t):
        """Estimates the local magnetic field in body coordinates
        
        Args:
            q_actual (numpy ndarray): the actual quaternion representing the
                attitude (from the inertial to body frame) of the spacecraft
            r (numpy ndarray): the inertial position of the spacecraft (m)
            t (float): the current simulation time in seconds
            delta_t (float): the time between user-defined integrator steps
                (not the internal/adaptive integrator steps) in seconds
        
        Returns:
            numpy ndarray: the vector representing the local magnetic field
                measured in body coordinates (with measurement noise applied)
        """
        B_i = compute_local_magnetic_field(r)
        T_actual = quaternion_to_dcm(q_actual)
        B_b = np.matmul(T_actual, B_i) + self.get_noise(t, delta_t)
        return B_b

    def get_noise(self, t, delta_t):
        """Gets the Gaussian noise for a set of x, y, and z magnetic field measurements

        NOTE: This method uses a cache of random values generated in this
              class's constructor. This is done to (1) reduce the overhead
              of many individual `rvs` calls and (2) ensure that all adaptive
              integrator steps in between user-defined steps use the same
              noise value (so that the dynamics are not constantly changing).
              Without this, the integrator fails to move forward in time.
              Therefore, a custom hash function is used to apply the same
              noise is necessary cases.
        
        Args:
            t (float): the current simulation time in seconds
            delta_t (float): the time between user-defined integrator steps
                (not the internal/adaptive integrator steps) in seconds
        
        Returns:
            numpy ndarray: the noise vector (noise in x, y, and z)
        """
        # randomly apply some noise, but use a cache so that each integrator-defined adaptive integration step
        # between user-defined integration steps uses the same noise value (otherwise the propagator fails)
        return np.array([
            self.noise_vals[0, int(t // delta_t) % len(self.noise_vals[0])],
            self.noise_vals[1, int(t // delta_t) % len(self.noise_vals[1])],
            self.noise_vals[2, int(t // delta_t) % len(self.noise_vals[2])]
        ])


class EarthHorizonSensor(object):
    """A class to store Earth horizon sensor parameters and methods
    
    Args:
        resolution (float): the accuracy of the sensor in degrees; an
            accuracy of 0 means no measurement noise is applied (it is
            a perfect sensor)
    
    Attributes:
        noise_vals (numpy ndarray): a cache of pre-generated noise values
                to aid in the addition of noise
    """

    def __init__(self, accuracy):
        """Constructs an Earth horizon sensor to measure the Earth direction
        
        Args:
            resolution (float): the accuracy of the sensor in degrees; an
                accuracy of 0 means no measurement noise is applied (it is
                a perfect sensor)
        """
        # convert from deg to rad
        accuracy *= np.pi / 180

        # decompose overall accuracy (represented as a single angular error
        # between the correct vector and the estimated vector) into angular
        # errors in the x, y, and z directions (error in each is the sqrt of
        # 1/3 the accuracy squared so that the norm of the vector containing
        # all 3 angular errors is the overall angular error)
        noise_factor = np.sqrt(accuracy**2 / 3)
        noise_func = norm(loc=0.0, scale=noise_factor)
        self.noise_vals = np.array([
            noise_func.rvs(size=100000),
            noise_func.rvs(size=100000),
            noise_func.rvs(size=100000)
        ])

    def estimate_earth_direction(self, q_actual, r, t, delta_t):
        """Estimates the direction vector to the Earth
        
        Args:
            q_actual (numpy ndarray): the actual quaternion representing the
                attitude (from the inertial to body frame) of the spacecraft
            r (numpy ndarray): the inertial position of the spacecraft (m)
            t (float): the current simulation time in seconds
            delta_t (float): the time between user-defined integrator steps
                (not the internal/adaptive integrator steps) in seconds
        
        Returns:
            numpy ndarray: the direction vector to the Earth as measured by the
                sensor in body coordinates
        """
        d_earth_inertial = compute_earth_direction(r)
        T_actual = quaternion_to_dcm(q_actual)
        noise = self.get_noise(t, delta_t)
        theta = np.linalg.norm(noise)
        e = normalize(noise)
        e_x = skew_symmetric(e)
        T_err = (np.diag([1, 1, 1]) - 2 * np.sin(theta) * e_x +
                 (1 - np.cos(theta)) * np.matmul(e_x, e_x))
        d_earth_body = np.matmul(T_err, np.matmul(T_actual, d_earth_inertial))
        return d_earth_body

    def get_noise(self, t, delta_t):
        """Gets the Gaussian noise for a set of x, y, and z direction measurements

        NOTE: This method uses a cache of random values generated in this
              class's constructor. This is done to (1) reduce the overhead
              of many individual `rvs` calls and (2) ensure that all adaptive
              integrator steps in between user-defined steps use the same
              noise value (so that the dynamics are not constantly changing).
              Without this, the integrator fails to move forward in time.
              Therefore, a custom hash function is used to apply the same
              noise is necessary cases.
        
        Args:
            t (float): the current simulation time in seconds
            delta_t (float): the time between user-defined integrator steps
                (not the internal/adaptive integrator steps) in seconds
        
        Returns:
            numpy ndarray: the noise vector (noise in x, y, and z)
        """
        # randomly apply some noise, but use a cache so that each integrator-defined adaptive integration step
        # between user-defined integration steps uses the same noise value (otherwise the propagator fails)
        return np.array([
            self.noise_vals[0, int(t // delta_t) % len(self.noise_vals[0])],
            self.noise_vals[1, int(t // delta_t) % len(self.noise_vals[1])],
            self.noise_vals[2, int(t // delta_t) % len(self.noise_vals[2])]
        ])