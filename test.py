import numpy as np
import matplotlib.pyplot as plt
from spacecraft import Spacecraft
from actuators import Actuators
from sensors import Gyros, Magnetometer, EarthHorizonSensor
from controller import PDController
from math_utils import quaternion_multiply
from simulation import simulate_adcs


def main():
    J = np.diag([1000, 500, 600])
    no_controller = PDController(
        k_d=np.diag([0, 0, 0]), k_p=np.diag([0, 0, 0]))
    controller = PDController(
        k_d=np.diag([.01, .01, .01]), k_p=np.diag([.1, .1, .1]))

    # Northrop Grumman LN-200S Gyros
    gyros = Gyros(bias_stability=1, angular_random_walk=0.07)
    perfect_gyros = Gyros(bias_stability=0, angular_random_walk=0)

    # NewSpace Systems Magnetometer
    magnetometer = Magnetometer(resolution=10e-9)
    perfect_magnetometer = Magnetometer(resolution=0)

    # Adcole Maryland Aerospace MAI-SES Static Earth Sensor
    earth_horizon_sensor = EarthHorizonSensor(accuracy=0.25)
    perfect_earth_horizon_sensor = EarthHorizonSensor(accuracy=0)

    # L-3 RWA-15 Reaction Wheel Assembly
    perfect_actuators = Actuators(
        rxwl_mass=14,
        rxwl_radius=0.1845,
        rxwl_max_torque=np.inf,
        noise_factor=0.0)
    actuators = Actuators(
        rxwl_mass=14,
        rxwl_radius=0.1845,
        rxwl_max_torque=np.inf,
        noise_factor=0.01)

    # # # problem-specific # # #
    mu_earth = 3.986004418e14
    orbit_radius = 6600000
    orbit_w = np.sqrt(mu_earth / orbit_radius**3)
    period = 2 * np.pi / orbit_w

    q_0_nominal = np.array([0.5, 0.5, 0.5, 0.5])
    w_nominal = np.array([0, 0, -2 * np.pi / period])

    q_0 = quaternion_multiply(
        np.array(
            [0, np.sin(2 * np.pi / 180 / 2), 0,
             np.cos(2 * np.pi / 180 / 2)]), q_0_nominal)
    w_0 = w_nominal + np.array([0.01, 0, 0])

    M_perturb_func = lambda t, q, w: 0.01 * np.array([np.cos(2*np.pi*t/100), np.sin(2*np.pi*t/100), -np.sin(2*np.pi*t/100)])

    # no_perturb_func = lambda t, q, w: np.array([0, 0, 0])

    def position_velocity_func(t):
        r = orbit_radius / np.sqrt(2) * np.array([
            -np.cos(orbit_w * t),
            np.sqrt(2) * np.sin(orbit_w * t),
            np.cos(orbit_w * t),
        ])
        v = orbit_w * orbit_radius / np.sqrt(2) * np.array([
            np.sin(orbit_w * t),
            np.sqrt(2) * np.cos(orbit_w * t),
            -np.sin(orbit_w * t),
        ])
        return r, v

    def desired_state_func(t, q_estimated, w_estimated):
        q_intermediate = np.array(
            [0, 0,
             np.sin(w_nominal[2] * t / 2),
             np.cos(w_nominal[2] * t / 2)])
        q_desired = quaternion_multiply(q_intermediate, q_0_nominal)
        return q_desired, w_nominal

    r_0, v_0 = position_velocity_func(0)

    satellite_no_control = Spacecraft(
        J=J,
        controller=no_controller,
        gyros=perfect_gyros,
        magnetometer=perfect_magnetometer,
        earth_horizon_sensor=perfect_earth_horizon_sensor,
        actuators=perfect_actuators,
        q=q_0,
        w=w_0,
        r=r_0,
        v=v_0)

    satellite_perfect = Spacecraft(
        J=J,
        controller=controller,
        gyros=perfect_gyros,
        magnetometer=perfect_magnetometer,
        earth_horizon_sensor=perfect_earth_horizon_sensor,
        actuators=perfect_actuators,
        q=q_0,
        w=w_0,
        r=r_0,
        v=v_0)

    satellite_noise = Spacecraft(
        J=J,
        controller=controller,
        gyros=gyros,
        magnetometer=magnetometer,
        earth_horizon_sensor=earth_horizon_sensor,
        actuators=actuators,
        q=q_0,
        w=w_0,
        r=r_0,
        v=v_0)

    simulate(
        satellite_no_control,
        desired_state_func,
        M_perturb_func,
        position_velocity_func,
        tag="(No Control)")

    simulate(
        satellite_perfect,
        desired_state_func,
        M_perturb_func,
        position_velocity_func,
        tag="(Perfect Estimation & Control)")

    simulate(
        satellite_noise,
        desired_state_func,
        M_perturb_func,
        position_velocity_func,
        tag="(Actual Estimation & Control)")


def simulate(satellite,
             desired_state_func,
             M_perturb_func,
             position_velocity_func,
             tag=""):
    results = simulate_adcs(
        satellite=satellite,
        nominal_state_func=desired_state_func,
        perturbations_func=M_perturb_func,
        position_velocity_func=position_velocity_func,
        start_time=0,
        delta_t=1,
        stop_time=6000,
        verbose=True)

    plt.figure(1)
    plt.subplot(411)
    plt.title("Evolution of Quaternion Components over Time" + " " + tag)
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

    plt.figure(2)
    plt.subplot(311)
    plt.title("Evolution of Angular Velocity over Time" + " " + tag)
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

    plt.figure(3)
    plt.subplot(311)
    plt.title("Evolution of Angular Velocity over Time" + " " + tag)
    plt.plot(results["times"], results["w_actual"][:, 0], label="actual")
    plt.plot(results["times"], results["w_estimated"][:, 0], label="estimated")
    plt.ylabel("w_x (rad/s)")
    plt.xlabel("Time (s)")
    plt.legend()
    plt.subplot(312)
    plt.plot(results["times"], results["w_actual"][:, 1], label="actual")
    plt.plot(results["times"], results["w_estimated"][:, 1], label="estimated")
    plt.ylabel("w_y (rad/s)")
    plt.xlabel("Time (s)")
    plt.legend()
    plt.subplot(313)
    plt.plot(results["times"], results["w_actual"][:, 2], label="actual")
    plt.plot(results["times"], results["w_estimated"][:, 2], label="estimated")
    plt.ylabel("w_z (rad/s)")
    plt.xlabel("Time (s)")
    plt.legend()

    plt.figure(4)
    plt.subplot(411)
    plt.title("Evolution of Quaternion Components over Time" + " " + tag)
    plt.plot(results["times"], results["q_actual"][:, 0], label="actual")
    plt.plot(results["times"], results["q_estimated"][:, 0], label="estimated")
    plt.ylabel("Q0")
    plt.xlabel("Time (s)")
    plt.legend()
    plt.subplot(412)
    plt.plot(results["times"], results["q_actual"][:, 1], label="actual")
    plt.plot(results["times"], results["q_estimated"][:, 1], label="estimated")
    plt.ylabel("Q1")
    plt.xlabel("Time (s)")
    plt.legend()
    plt.subplot(413)
    plt.plot(results["times"], results["q_actual"][:, 2], label="actual")
    plt.plot(results["times"], results["q_estimated"][:, 2], label="estimated")
    plt.ylabel("Q2")
    plt.xlabel("Time (s)")
    plt.legend()
    plt.subplot(414)
    plt.plot(results["times"], results["q_actual"][:, 3], label="actual")
    plt.plot(results["times"], results["q_estimated"][:, 3], label="estimated")
    plt.ylabel("Q3")
    plt.xlabel("Time (s)")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
