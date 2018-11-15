import numpy as np
from scipy.integrate import ode
from kinematics import quaternion_derivative
from dynamics import angular_velocity_derivative
from math_utils import normalize


def derivatives_func(t, x, satellite, errors_func, nominal_state_func,
                     perturbations_func):
    """Computes the derivative of the spacecraft state (used as the propagator function)
    
    Args:
        t (float): the time (in seconds)
        x (numpy ndarray): the state (7x1) where the elements are:
            [0, 1, 2, 3]: the quaternion describing the spacecraft attitude
            [4, 5, 6]: the angular velocity of the spacecraft
        satellite (Spacecraft): an object that stores the spacecraft's inertia tensor, sensors, actuators, and controller
        errors_func (function): the function that should compute the attitude error and attitude rate errors; its header must be t, q_estimated, w_estimated, q_desired, w_desired
        nominal_state_func (function): [description]
        perturbations_func (function): [description]
    
    Returns:
        [type]: [description]
    """

    q = normalize(x[0:4])
    w = x[4:7]
    M_applied, _ = simulate_estimation_and_control(
        t, q, w, satellite, errors_func, nominal_state_func)

    # TODO: determine whether perturbations + dynamics/kinematics are computed
    # with the true attitude & angvel or the estimated

    # calculate the perturbing torques on the satellite
    M_perturb = perturbations_func(t, q, w)

    dx = np.empty((7, ))
    dx[0:4] = quaternion_derivative(q, w)
    dx[4:7] = angular_velocity_derivative(satellite.J, w,
                                          [M_applied, M_perturb])
    return dx


def simulate_adcs(start_time,
                  delta_t,
                  stop_time,
                  init_state,
                  satellite,
                  errors_func,
                  nominal_state_func,
                  perturbations_func,
                  verbose=False):

    solver = ode(derivatives_func)
    solver.set_integrator('dopri5', rtol=1e-10, atol=1e-12, nsteps=10000)
    solver.set_initial_value(y=init_state, t=start_time)
    solver.set_f_params(satellite, errors_func, nominal_state_func,
                        perturbations_func)

    length = int((stop_time - 0) / delta_t + 1)
    times = np.empty((length, ))
    q_actual = np.empty((length, 4))
    w_actual = np.empty((length, 3))
    q_estimated = np.empty((length, 4))
    w_estimated = np.empty((length, 3))
    q_desired = np.empty((length, 4))
    w_desired = np.empty((length, 3))
    attitude_err = np.empty((length, 3))
    attitude_rate_err = np.empty((length, 3))
    M_ctrl = np.empty((length, 3))
    M_applied = np.empty((length, 3))
    M_perturb = np.empty((length, 3))
    i = 0

    if verbose:
        print("Starting propagation at time: {}".format(start_time))

    while solver.successful() and solver.t <= stop_time:
        if verbose:
            print("Time: {}\nState: {}\n".format(solver.t, solver.y))

        # this section currently duplicates calculations for logging purposes
        t = solver.t
        q = normalize(solver.y[0:4])
        w = solver.y[4:7]
        _, log = simulate_estimation_and_control(
            t, q, w, satellite, errors_func, nominal_state_func)
        times[i] = t
        q_actual[i] = q
        w_actual[i] = w
        q_estimated[i] = log["q_estimated"]
        w_estimated[i] = log["w_estimated"]
        q_desired[i] = log["q_desired"]
        w_desired[i] = log["w_desired"]
        attitude_err[i] = log["attitude_err"]
        attitude_rate_err[i] = log["attitude_rate_err"]
        M_ctrl[i] = log["M_ctrl"]
        M_applied[i] = log["M_applied"]
        M_perturb[i] = perturbations_func(t, q, w)
        i += 1

        # continue integrating
        solver.integrate(solver.t + delta_t)

    results = {}
    results["times"] = times
    results["q_actual"] = q_actual
    results["w_actual"] = w_actual
    results["q_estimated"] = q_estimated
    results["w_estimated"] = w_estimated
    results["q_desired"] = q_desired
    results["w_desired"] = w_desired
    results["attitude_err"] = attitude_err
    results["attitude_rate_err"] = attitude_rate_err
    results["M_ctrl"] = M_ctrl
    results["M_applied"] = M_applied
    results["M_perturb"] = M_perturb
    return results


def simulate_estimation_and_control(t, q, w, satellite, errors_func,
                                    nominal_state_func):
    # get an attitude and angular velocity estimate from the sensors
    # q_estimated = satellite.sensors.estimate_attitude(q)
    # w_estimated = sensors.estimate_angvel(w)
    q_estimated = q
    w_estimated = w

    # compute the desired attitude and angular velocity
    q_desired, w_desired = nominal_state_func(t, q_estimated, w_estimated)

    # calculate the errors between your desired and actual state
    attitude_err, attitude_rate_err = errors_func(t, q_estimated, w_estimated,
                                                  q_desired, w_desired)

    # determine the control torques necessary to change state
    M_ctrl = satellite.controller.calculate_control_torques(
        attitude_err, attitude_rate_err, satellite.J)

    # use actuators to apply the control torques
    # M_applied = satellite.actuators.apply_control_torques(M_ctrl)
    M_applied = M_ctrl

    logged_results = {
        "q_estimated": q_estimated,
        "w_estimated": w_estimated,
        "q_desired": q_desired,
        "w_desired": w_desired,
        "attitude_err": attitude_err,
        "attitude_rate_err": attitude_rate_err,
        "M_ctrl": M_ctrl,
        "M_applied": M_applied
    }

    return M_applied, logged_results