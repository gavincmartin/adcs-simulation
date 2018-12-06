import numpy as np
from math_utils import cross, normalize
from sensor_models import compute_local_magnetic_field


def gravity_gradient_torque(r, J):
    """Computes the gravity gradient torque acting on a spacecraft
    
    Args:
        r (numpy ndarray): the inertial position of the spacecraft (m)
        J (numpy ndarray): the spacecraft's inertia tensor (3x3) (kg * m^2)
    
    Returns:
        numpy ndarray: the gravity gradient torque acting on the spacecraft
            coordinatized in the body frame
    """
    r_mag = np.linalg.norm(r)
    mu_earth = 3.986004418e14
    w_orb = np.sqrt(mu_earth / r_mag**3)
    d_b = -normalize(r)
    M_gg = 3 * w_orb**2 * cross(d_b, np.matmul(J, d_b))
    return M_gg


def magnetic_field_torque(r, dipole, DCM):
    """Computes the magnetic field torque acting on a spacecraft
    
    Args:
        r (numpy ndarray): the inertial position of the spacecraft (m)
        dipole (numpy ndarray): the spacecraft's residual magnetic dipole
            vector (A * m^2)
        DCM (numpy ndarray): the DCM from inertial to body frame
    
    Returns:
        numpy ndarray: the magnetic field torque acting on the spacecraft
            coordinatized in the body frame
    """
    B_i = compute_local_magnetic_field(r)
    B_b = np.matmul(DCM, B_i)
    M_b = cross(dipole, B_b)
    return M_b
