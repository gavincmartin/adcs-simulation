import numpy as np
from scipy.stats import norm, uniform


class Gyros(object):
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
        distribution = uniform(-bias_stability, 2 * bias_stability)
        self.bias = distribution.rvs(size=3)

        # TODO: see if this is the right conversion factor
        # convert angular random walk from deg/sqrt(hr) to rad/s
        noise_factor = (angular_random_walk * np.pi / 180)**2 / 3600
        noise_func = norm(loc=0.0, scale=noise_factor)
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
        # randomly apply some noise, but use a cache so that each integrator-defined adaptive integration step
        # between user-defined integration steps uses the same noise value (otherwise the propagator fails)
        return np.array([
            self.noise_vals[0, int(t // delta_t) % len(self.noise_vals)],
            self.noise_vals[1, int(t // delta_t) % len(self.noise_vals)],
            self.noise_vals[2, int(t // delta_t) % len(self.noise_vals)]
        ])


class Magnetometer(object):
    pass