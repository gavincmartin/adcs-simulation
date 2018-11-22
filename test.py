import numpy as np
import matplotlib.pyplot as plt
from errors import calculate_attitude_error, calculate_attitude_rate_error
from spacecraft import Spacecraft
from actuators import Actuators
from controller import PDController
from math_utils import quaternion_multiply
from simulation import simulate_adcs


def main():
    no_control()
    control()
    control_w_noise()


def no_control():
    J = np.diag([1000, 500, 600])
    controller = PDController(k_d=np.diag([0, 0, 0]), k_p=np.diag([0, 0, 0]))
    sensors = None
    actuators = Actuators(
        rxwl_mass=14,
        rxwl_radius=0.1845,
        rxwl_max_torque=np.inf,
        noise_factor=0.01)

    # # # problem-specific # # #
    period = 6000

    q_0_nominal = np.array([0.5, 0.5, 0.5, 0.5])
    w_nominal = np.array([0, 0, -2 * np.pi / period])

    q_0 = quaternion_multiply(
        np.array(
            [0, np.sin(2 * np.pi / 180 / 2), 0,
             np.cos(2 * np.pi / 180 / 2)]), q_0_nominal)
    w_0 = w_nominal + np.array([0.01, 0, 0])

    M_perturb_func = lambda t, q, w: 0.01 * np.array([np.cos(2*np.pi*t/100), np.sin(2*np.pi*t/100), -np.sin(2*np.pi*t/100)])

    def err_func(t, q_estimated, w_estimated, q_desired, w_desired):
        attitude_err = calculate_attitude_error(q_desired, q_estimated)
        attitude_rate_err = calculate_attitude_rate_error(
            w_desired, w_estimated, attitude_err)
        return attitude_err, attitude_rate_err

    def desired_state_func(t, q_estimated, w_estimated):
        q_intermediate = np.array(
            [0, 0,
             np.sin(w_nominal[2] * t / 2),
             np.cos(w_nominal[2] * t / 2)])
        q_desired = quaternion_multiply(q_intermediate, q_0_nominal)
        return q_desired, w_nominal

    satellite = Spacecraft(J, controller, sensors, actuators, q=q_0, w=w_0)

    # # # # # # # # # # # # # #
    results = simulate_adcs(
        satellite=satellite,
        errors_func=err_func,
        nominal_state_func=desired_state_func,
        perturbations_func=M_perturb_func,
        start_time=0,
        delta_t=1,
        stop_time=6000,
        verbose=True)

    plt.figure(1)
    plt.subplot(411)
    plt.title("Evolution of Quaternion Components over Time (No Control)")
    plt.plot(results["times"], results["q_actual"][:, 0], label="actual")
    plt.plot(results["times"], results["q_desired"][:, 0], label="desired")
    plt.ylabel("Q0")
    plt.xlabel("Time (s)")
    plt.legend()
    plt.subplot(412)
    plt.plot(results["times"], results["q_actual"][:, 1], label="actual")
    plt.plot(results["times"], results["q_desired"][:, 1], label="desired")
    plt.ylabel("Q1")
    plt.xlabel("Time (s)")
    plt.legend()
    plt.subplot(413)
    plt.plot(results["times"], results["q_actual"][:, 2], label="actual")
    plt.plot(results["times"], results["q_desired"][:, 2], label="desired")
    plt.ylabel("Q2")
    plt.xlabel("Time (s)")
    plt.legend()
    plt.subplot(414)
    plt.plot(results["times"], results["q_actual"][:, 3], label="actual")
    plt.plot(results["times"], results["q_desired"][:, 3], label="desired")
    plt.ylabel("Q3")
    plt.xlabel("Time (s)")
    plt.legend()
    plt.show()

    plt.figure(2)
    plt.subplot(311)
    plt.title("Evolution of Angular Velocity over Time (No Control)")
    plt.plot(results["times"], results["w_actual"][:, 0], label="actual")
    plt.plot(results["times"], results["w_desired"][:, 0], label="desired")
    plt.ylabel("w_x (rad/s)")
    plt.xlabel("Time (s)")
    plt.legend()
    plt.subplot(312)
    plt.plot(results["times"], results["w_actual"][:, 1], label="actual")
    plt.plot(results["times"], results["w_desired"][:, 1], label="desired")
    plt.ylabel("w_y (rad/s)")
    plt.xlabel("Time (s)")
    plt.legend()
    plt.subplot(313)
    plt.plot(results["times"], results["w_actual"][:, 2], label="actual")
    plt.plot(results["times"], results["w_desired"][:, 2], label="desired")
    plt.ylabel("w_z (rad/s)")
    plt.xlabel("Time (s)")
    plt.legend()
    plt.show()


def control():
    J = np.diag([1000, 500, 600])
    controller = PDController(
        k_d=np.diag([.01, .01, .01]), k_p=np.diag([.1, .1, .1]))
    sensors = None
    actuators = Actuators(
        rxwl_mass=14,
        rxwl_radius=369 / 1000 / 2,
        rxwl_max_torque=np.inf,
        noise_factor=0.0)

    # # # problem-specific # # #
    period = 6000

    q_0_nominal = np.array([0.5, 0.5, 0.5, 0.5])
    w_nominal = np.array([0, 0, -2 * np.pi / period])

    q_0 = quaternion_multiply(
        np.array(
            [0, np.sin(2 * np.pi / 180 / 2), 0,
             np.cos(2 * np.pi / 180 / 2)]), q_0_nominal)
    w_0 = w_nominal + np.array([0.01, 0, 0])

    M_perturb_func = lambda t, q, w: 0.01 * np.array([np.cos(2*np.pi*t/100), np.sin(2*np.pi*t/100), -np.sin(2*np.pi*t/100)])

    def err_func(t, q_estimated, w_estimated, q_desired, w_desired):
        attitude_err = calculate_attitude_error(q_desired, q_estimated)
        attitude_rate_err = calculate_attitude_rate_error(
            w_desired, w_estimated, attitude_err)
        return attitude_err, attitude_rate_err

    def desired_state_func(t, q_estimated, w_estimated):
        q_intermediate = np.array(
            [0, 0,
             np.sin(w_nominal[2] * t / 2),
             np.cos(w_nominal[2] * t / 2)])
        q_desired = quaternion_multiply(q_intermediate, q_0_nominal)
        return q_desired, w_nominal

    satellite = Spacecraft(J, controller, sensors, actuators, q=q_0, w=w_0)

    # # # # # # # # # # # # # #
    results = simulate_adcs(
        start_time=0,
        delta_t=1,
        stop_time=6000,
        satellite=satellite,
        errors_func=err_func,
        nominal_state_func=desired_state_func,
        perturbations_func=M_perturb_func,
        verbose=True)

    plt.figure(1)
    plt.subplot(411)
    plt.title("Evolution of Quaternion Components over Time (With PD Control)")
    plt.plot(results["times"], results["q_actual"][:, 0], label="actual")
    plt.plot(results["times"], results["q_desired"][:, 0], label="desired")
    plt.ylabel("Q0")
    plt.xlabel("Time (s)")
    plt.legend()
    plt.subplot(412)
    plt.plot(results["times"], results["q_actual"][:, 1], label="actual")
    plt.plot(results["times"], results["q_desired"][:, 1], label="desired")
    plt.ylabel("Q1")
    plt.xlabel("Time (s)")
    plt.legend()
    plt.subplot(413)
    plt.plot(results["times"], results["q_actual"][:, 2], label="actual")
    plt.plot(results["times"], results["q_desired"][:, 2], label="desired")
    plt.ylabel("Q2")
    plt.xlabel("Time (s)")
    plt.legend()
    plt.subplot(414)
    plt.plot(results["times"], results["q_actual"][:, 3], label="actual")
    plt.plot(results["times"], results["q_desired"][:, 3], label="desired")
    plt.ylabel("Q3")
    plt.xlabel("Time (s)")
    plt.legend()
    plt.show()

    plt.figure(2)
    plt.subplot(311)
    plt.title("Evolution of Angular Velocity over Time (With PD Control)")
    plt.plot(results["times"], results["w_actual"][:, 0], label="actual")
    plt.plot(results["times"], results["w_desired"][:, 0], label="desired")
    plt.ylabel("w_x (rad/s)")
    plt.xlabel("Time (s)")
    plt.legend()
    plt.subplot(312)
    plt.plot(results["times"], results["w_actual"][:, 1], label="actual")
    plt.plot(results["times"], results["w_desired"][:, 1], label="desired")
    plt.ylabel("w_y (rad/s)")
    plt.xlabel("Time (s)")
    plt.legend()
    plt.subplot(313)
    plt.plot(results["times"], results["w_actual"][:, 2], label="actual")
    plt.plot(results["times"], results["w_desired"][:, 2], label="desired")
    plt.ylabel("w_z (rad/s)")
    plt.xlabel("Time (s)")
    plt.legend()
    plt.show()


def control_w_noise():
    J = np.diag([1000, 500, 600])
    controller = PDController(
        k_d=np.diag([.01, .01, .01]), k_p=np.diag([.1, .1, .1]))
    sensors = None
    actuators = Actuators(
        rxwl_mass=14,
        rxwl_radius=369 / 1000 / 2,
        rxwl_max_torque=np.inf,
        noise_factor=0.03)

    # # # problem-specific # # #
    period = 6000

    q_0_nominal = np.array([0.5, 0.5, 0.5, 0.5])
    w_nominal = np.array([0, 0, -2 * np.pi / period])

    q_0 = quaternion_multiply(
        np.array(
            [0, np.sin(2 * np.pi / 180 / 2), 0,
             np.cos(2 * np.pi / 180 / 2)]), q_0_nominal)
    w_0 = w_nominal + np.array([0.01, 0, 0])

    M_perturb_func = lambda t, q, w: 0.01 * np.array([np.cos(2*np.pi*t/100), np.sin(2*np.pi*t/100), -np.sin(2*np.pi*t/100)])

    def err_func(t, q_estimated, w_estimated, q_desired, w_desired):
        attitude_err = calculate_attitude_error(q_desired, q_estimated)
        attitude_rate_err = calculate_attitude_rate_error(
            w_desired, w_estimated, attitude_err)
        return attitude_err, attitude_rate_err

    def desired_state_func(t, q_estimated, w_estimated):
        q_intermediate = np.array(
            [0, 0,
             np.sin(w_nominal[2] * t / 2),
             np.cos(w_nominal[2] * t / 2)])
        q_desired = quaternion_multiply(q_intermediate, q_0_nominal)
        return q_desired, w_nominal

    satellite = Spacecraft(J, controller, sensors, actuators, q=q_0, w=w_0)

    # # # # # # # # # # # # # #
    results = simulate_adcs(
        start_time=0,
        delta_t=1,
        stop_time=6000,
        satellite=satellite,
        errors_func=err_func,
        nominal_state_func=desired_state_func,
        perturbations_func=M_perturb_func,
        verbose=True)

    plt.figure(1)
    plt.subplot(411)
    plt.title(
        "Evolution of Quaternion Components over Time (PD Control w/ Actuator Noise)"
    )
    plt.plot(results["times"], results["q_actual"][:, 0], label="actual")
    plt.plot(results["times"], results["q_desired"][:, 0], label="desired")
    plt.ylabel("Q0")
    plt.xlabel("Time (s)")
    plt.legend()
    plt.subplot(412)
    plt.plot(results["times"], results["q_actual"][:, 1], label="actual")
    plt.plot(results["times"], results["q_desired"][:, 1], label="desired")
    plt.ylabel("Q1")
    plt.xlabel("Time (s)")
    plt.legend()
    plt.subplot(413)
    plt.plot(results["times"], results["q_actual"][:, 2], label="actual")
    plt.plot(results["times"], results["q_desired"][:, 2], label="desired")
    plt.ylabel("Q2")
    plt.xlabel("Time (s)")
    plt.legend()
    plt.subplot(414)
    plt.plot(results["times"], results["q_actual"][:, 3], label="actual")
    plt.plot(results["times"], results["q_desired"][:, 3], label="desired")
    plt.ylabel("Q3")
    plt.xlabel("Time (s)")
    plt.legend()
    plt.show()

    plt.figure(2)
    plt.subplot(311)
    plt.title(
        "Evolution of Angular Velocity over Time (PD Control w/ Actuator Noise)"
    )
    plt.plot(results["times"], results["w_actual"][:, 0], label="actual")
    plt.plot(results["times"], results["w_desired"][:, 0], label="desired")
    plt.ylabel("w_x (rad/s)")
    plt.xlabel("Time (s)")
    plt.legend()
    plt.subplot(312)
    plt.plot(results["times"], results["w_actual"][:, 1], label="actual")
    plt.plot(results["times"], results["w_desired"][:, 1], label="desired")
    plt.ylabel("w_y (rad/s)")
    plt.xlabel("Time (s)")
    plt.legend()
    plt.subplot(313)
    plt.plot(results["times"], results["w_actual"][:, 2], label="actual")
    plt.plot(results["times"], results["w_desired"][:, 2], label="desired")
    plt.ylabel("w_z (rad/s)")
    plt.xlabel("Time (s)")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
