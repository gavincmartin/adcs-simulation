import numpy as np
from scipy.integrate import ode, RK45, odeint
from kinematics import quaternion_derivative
from dynamics import angular_velocity_derivative
from math_utils import normalize


def derivatives_func(t, x, satellite, errors_func, nominal_state_func,
                     perturbations_func, delta_t):
    """Computes the derivative of the spacecraft state
    
    Args:
        t (float): the time (in seconds)
        x (numpy ndarray): the state (10x1) where the elements are:
            [0, 1, 2, 3]: the quaternion describing the spacecraft attitude
            [4, 5, 6]: the angular velocity of the spacecraft
            [7, 8, 9]: the angular velocities of the reaction wheels
        satellite (Spacecraft): the Spacecraft object that represents the
            satellite being modeled
        errors_func (function): the function that should compute the attitude
            error and attitude rate errors; its header must be (t, q_estimated,
            w_estimated, q_desired, w_desired)
        nominal_state_func (function): the function that should compute the
            nominal attitude and angular velocity; its header must be (t, q, w)
        perturbations_func (function): the function that should compute the
            perturbation torques (N * m); its header must be (t, q, w)
        delta_t (float): the time between user-defined integrator steps
                (not the internal/adaptive integrator steps) in seconds
    
    Returns:
        numpy ndarray: the derivative of the state (10x1) with respect to time
    """
    satellite.q = normalize(x[0:4])
    satellite.w = x[4:7]
    # only set if the satellite has actuators
    try:
        satellite.actuators.w_rxwls = x[7:10]
    except AttributeError:
        pass
    M_applied, w_dot_rxwls, _ = simulate_estimation_and_control(
        t, satellite, errors_func, nominal_state_func, delta_t)

    # calculate the perturbing torques on the satellite
    M_perturb = perturbations_func(t, satellite.q, satellite.w)

    dx = np.empty((10, ))
    dx[0:4] = quaternion_derivative(satellite.q, satellite.w)
    dx[4:7] = angular_velocity_derivative(satellite.J, satellite.w,
                                          [M_applied, M_perturb])
    dx[7:10] = w_dot_rxwls
    return dx


def simulate_adcs(satellite,
                  errors_func,
                  nominal_state_func,
                  perturbations_func,
                  start_time=0,
                  delta_t=1,
                  stop_time=6000,
                  verbose=False):
    """Simulates an attitude determination and control system
    
    Args:
        satellite (Spacecraft): the Spacecraft object that represents the
            satellite being modeled
        errors_func (function): the function that should compute the attitude
            error and attitude rate errors; its header must be (t, q_estimated,
            w_estimated, q_desired, w_desired)
        nominal_state_func (function): the function that should compute the
            nominal attitude and angular velocity; its header must be (t, q, w)
        perturbations_func (function): the function that should compute the
            perturbation torques (N * m); its header must be (t, q, w)
        verbose (bool, optional): Defaults to False. [description]
        start_time (float, optional): Defaults to 0. The start time of the
            simulation in seconds
        delta_t (float, optional): Defaults to 1. The time between user-defined
            integrator steps (not the internal/adaptive integrator steps) in
            seconds
        stop_time (float, optional): Defaults to 6000. The end time of the
            simulation in seconds
        verbose (bool, optional). Defaults to False. Whether or not to print
            integrator output to the console while running.
    
    Returns:
        dict: a dictionary of simulation results. Each value is an NxM numpy
            ndarray where N is the number of time steps taken and M is the
            size of the data being represented at that time (M=1 for time, 
            3 for angular velocity, 4 for quaternion, etc.)
            Contains:
                - times (numpy ndarray): the times of all associated data
                - q_actual (numpy ndarray): actual quaternion
                - w_actual (numpy ndarray): actual angular velocity
                - q_estimated (numpy ndarray): estimated quaternion
                - w_estimated (numpy ndarray): estimated angular velocity
                - q_desired (numpy ndarray): desired quaternion
                - q_desired (numpy ndarray): desired angular velocity
                - attitude_err (numpy ndarray): attitude error
                - attitude_rate_err (numpy ndarray): attitude rate error
                - M_ctrl (numpy ndarray): control torque
                - M_applied (numpy ndarray): applied control torque
                - w_dot_rxwls (numpy ndarray): angular acceleration of
                    reaction wheels
                - M_perturb (numpy ndarray): sum of perturbation torques

    """
    try:
        init_state = [*satellite.q, *satellite.w, *satellite.actuators.w_rxwls]
    except AttributeError:
        init_state = [*satellite.q, *satellite.w, 0, 0, 0]

    solver = ode(derivatives_func)
    solver.set_integrator(
        'lsoda',
        rtol=(1e-10, 1e-10, 1e-10, 1e-10, 1e-10, 1e-10, 1e-10, 1e-6, 1e-6,
              1e-6),
        atol=(1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-8, 1e-8,
              1e-8),
        nsteps=10000)
    solver.set_initial_value(y=init_state, t=start_time)
    solver.set_f_params(satellite, errors_func, nominal_state_func,
                        perturbations_func, delta_t)

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
    w_dot_rxwls = np.empty((length, 3))
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
        satellite.q = q
        satellite.w = w
        _, _, log = simulate_estimation_and_control(
            t, satellite, errors_func, nominal_state_func, delta_t, log=True)
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
        w_dot_rxwls[i] = log["w_dot_rxwls"]
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


def simulate_estimation_and_control(t,
                                    satellite,
                                    errors_func,
                                    nominal_state_func,
                                    delta_t,
                                    log=False):
    """Simulates attitude estimation and control for derivatives calculation
    
    Args:
        t (float): the time (in seconds)
        satellite (Spacecraft): the Spacecraft object that represents the
            satellite being modeled
        errors_func (function): the function that should compute the attitude
            error and attitude rate errors; its header must be (t, q_estimated,
            w_estimated, q_desired, w_desired)
        nominal_state_func (function): the function that should compute the
            nominal attitude and angular velocity; its header must be (t, q, w)
        perturbations_func (function): the function that should compute the
            perturbation torques (N * m); its header must be (t, q, w)
        delta_t (float): the time between user-defined integrator steps
                (not the internal/adaptive integrator steps) in seconds
    
    Returns:
        numpy ndarray: the control moment (3x1) as actually applied on
                the reaction wheels (the input control torque with some
                Gaussian noise applied) (N * m)
        numpy ndarray: the angular acceleration of the 3 reaction wheels
            applied to achieve the applied torque (rad/s^2)
        dict: a dictionary containing results logged for this simulation step;
            Contains:
                - q_estimated (numpy ndarray): estimated quaternion
                - w_estimated (numpy ndarray): estimated angular velocity
                - q_desired (numpy ndarray): desired quaternion
                - q_desired (numpy ndarray): desired angular velocity
                - attitude_err (numpy ndarray): attitude error
                - attitude_rate_err (numpy ndarray): attitude rate error
                - M_ctrl (numpy ndarray): control torque
                - M_applied (numpy ndarray): applied control torque
                - w_dot_rxwls (numpy ndarray): angular acceleration of
                    reaction wheels
    """
    # get an attitude and angular velocity estimate from the sensors
    # q_estimated = satellite.sensors.estimate_attitude(q)
    # w_estimated = sensors.estimate_angvel(w)
    q_estimated = satellite.q
    w_estimated = satellite.w

    # compute the desired attitude and angular velocity
    q_desired, w_desired = nominal_state_func(t, q_estimated, w_estimated)

    # calculate the errors between your desired and actual state
    attitude_err, attitude_rate_err = errors_func(t, q_estimated, w_estimated,
                                                  q_desired, w_desired)

    # determine the control torques necessary to change state
    # print("time: {}".format(t))
    M_ctrl = satellite.calculate_control_torques(attitude_err,
                                                 attitude_rate_err)

    # use actuators to apply the control torques
    M_applied, w_dot_rxwls = satellite.apply_control_torques(
        M_ctrl, t, delta_t)

    if log:
        logged_results = {
            "q_estimated": q_estimated,
            "w_estimated": w_estimated,
            "q_desired": q_desired,
            "w_desired": w_desired,
            "attitude_err": attitude_err,
            "attitude_rate_err": attitude_rate_err,
            "M_ctrl": M_ctrl,
            "M_applied": M_applied,
            "w_dot_rxwls": w_dot_rxwls
        }
    else:
        logged_results = None

    return M_applied, w_dot_rxwls, logged_results