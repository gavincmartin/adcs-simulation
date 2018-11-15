import numpy as np
import matplotlib.pyplot as plt
from errors import calculate_attitude_error, calculate_attitude_rate_error
from spacecraft import Spacecraft
from controller import PDController
from math_utils import t1_matrix, t2_matrix, t3_matrix, dcm_to_quaternion
from simulation import simulate_adcs


def main():
    test()


def test():
    J = np.diag([100, 50, 60])
    controller = PDController(
        k_d=np.diag([.707, .707, .707]), k_p=np.diag([.707, .707, .707]))
    sensors = None
    actuators = None
    satellite = Spacecraft(J, controller, sensors, actuators)

    # # # problem-specific # # #
    period = 6000
    dcm_init = np.matmul(t1_matrix(-np.pi / 2), t2_matrix(-np.pi))
    print(dcm_init)
    dcm_nominal_func = lambda t: np.matmul(t3_matrix(-t / period * 2 * np.pi), dcm_init)
    M_perturb_func = lambda t, q, w: 0.01 * np.array([np.cos(2*np.pi*t/100), np.sin(2*np.pi*t/100), -np.sin(2*np.pi*t/100)])
    w_nominal = np.array([0, 0, -2 * np.pi / period])

    q_0 = dcm_to_quaternion(dcm_init)
    w_0 = np.array([0, 0, -2 * np.pi / period])

    init_attitude_err = np.array([2 * np.pi / 180, 0, 0])
    init_attitude_rate_err = np.array([0, 0, 0.1])

    def err_func(t, q_estimated, w_estimated, q_desired, w_desired):
        if t == 0:
            return init_attitude_err, init_attitude_rate_err
        else:
            attitude_err = calculate_attitude_error(q_desired, q_estimated)
            attitude_rate_err = calculate_attitude_rate_error(
                w_desired, w_estimated, attitude_err)
            return attitude_err, attitude_rate_err

    def desired_state_func(t, q_estimated, w_estimated):
        dcm_desired = dcm_nominal_func(t)
        q_desired = dcm_to_quaternion(dcm_desired)
        return q_desired, w_nominal

    print(desired_state_func(0, 0, 0))

    # # # # # # # # # # # # # #

    results = simulate_adcs(
        start_time=0,
        delta_t=1,
        stop_time=6000,
        init_state=np.array([*q_0, *w_0]),
        satellite=satellite,
        errors_func=err_func,
        nominal_state_func=desired_state_func,
        perturbations_func=M_perturb_func,
        verbose=True)

    plt.figure(1)
    plt.subplot(411)
    plt.title("Evolution of Quaternion Components over Time (With PD Control)")
    plt.plot(results["times"], results["q_actual"][:, 0])
    plt.ylabel("Q0")
    plt.xlabel("Time (s)")
    plt.subplot(412)
    plt.plot(results["times"], results["q_actual"][:, 1])
    plt.ylabel("Q1")
    plt.xlabel("Time (s)")
    plt.subplot(413)
    plt.plot(results["times"], results["q_actual"][:, 2])
    plt.ylabel("Q2")
    plt.xlabel("Time (s)")
    plt.subplot(414)
    plt.plot(results["times"], results["q_actual"][:, 3])
    plt.ylabel("Q3")
    plt.xlabel("Time (s)")
    # plt.show()

    plt.figure(2)
    plt.subplot(311)
    plt.title("Evolution of Angular Velocity over Time (With PD Control)")
    plt.plot(results["times"], results["w_actual"][:, 0])
    plt.ylabel("w_x (rad/s)")
    plt.xlabel("Time (s)")
    plt.subplot(312)
    plt.plot(results["times"], results["w_actual"][:, 1])
    plt.ylabel("w_y (rad/s)")
    plt.xlabel("Time (s)")
    plt.subplot(313)
    plt.plot(results["times"], results["w_actual"][:, 2])
    plt.ylabel("w_z (rad/s)")
    plt.xlabel("Time (s)")
    # plt.show()

    plt.figure(3)
    plt.subplot(311)
    plt.title("Control Torques over Time (With PD Control)")
    plt.plot(results["times"], results["M_ctrl"][:, 0])
    plt.ylabel("M_ctrl_x (N m)")
    plt.xlabel("Time (s)")
    plt.subplot(312)
    plt.plot(results["times"], results["M_ctrl"][:, 1])
    plt.ylabel("M_ctrl_y (N m)")
    plt.xlabel("Time (s)")
    plt.subplot(313)
    plt.plot(results["times"], results["M_ctrl"][:, 2])
    plt.ylabel("M_ctrl_z (N m)")
    plt.xlabel("Time (s)")
    # plt.show()


if __name__ == "__main__":
    main()
