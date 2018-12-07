# -*- coding: utf-8 -*-
"""Project module for attitude determination and control system.

This module is simply the control script that utilizes the simulation engine
and plots the results.
"""

import numpy as np
import matplotlib.pyplot as plt
from spacecraft import Spacecraft
from actuators import Actuators
from sensors import Gyros, Magnetometer, EarthHorizonSensor
from controller import PDController
from math_utils import (quaternion_multiply, t1_matrix, t2_matrix, t3_matrix,
                        dcm_to_quaternion, quaternion_to_dcm, normalize, cross)
from simulation import simulate_adcs


def main():
    # Define 6U CubeSat mass, dimensions, drag coefficient
    sc_mass = 8
    sc_dim = [226.3e-3, 100.0e-3, 366.0e-3]
    J = 1 / 12 * sc_mass * np.diag([
        sc_dim[1]**2 + sc_dim[2]**2, sc_dim[0]**2 + sc_dim[2]**2,
        sc_dim[0]**2 + sc_dim[1]**2
    ])
    sc_dipole = np.array([0, 0.018, 0])

    # Define two `PDController` objects—one to represent no control and one
    # to represent PD control with the specified gains
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

    # Sinclair Interplanetary 60 mNm-sec RXWLs
    actuators = Actuators(
        rxwl_mass=226e-3,
        rxwl_radius=0.5 * 65e-3,
        rxwl_max_torque=20e-3,
        rxwl_max_momentum=0.18,
        noise_factor=0.03)
    perfect_actuators = Actuators(
        rxwl_mass=226e-3,
        rxwl_radius=0.5 * 65e-3,
        rxwl_max_torque=np.inf,
        rxwl_max_momentum=np.inf,
        noise_factor=0.0)

    # define some orbital parameters
    mu_earth = 3.986004418e14
    R_e = 6.3781e6
    orbit_radius = R_e + 400e3
    orbit_w = np.sqrt(mu_earth / orbit_radius**3)
    period = 2 * np.pi / orbit_w

    # define a function that returns the inertial position and velocity of the
    # spacecraft (in m & m/s) at any given time
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

    # compute the initial inertial position and velocity
    r_0, v_0 = position_velocity_func(0)

    # define the body axes in relation to where we want them to be:
    # x = Earth-pointing
    # y = pointing along the velocity vector
    # z = normal to the orbital plane
    b_x = -normalize(r_0)
    b_y = normalize(v_0)
    b_z = cross(b_x, b_y)

    # construct the nominal DCM from inertial to body (at time 0) from the body
    # axes and compute the equivalent quaternion
    dcm_0_nominal = np.stack([b_x, b_y, b_z])
    q_0_nominal = dcm_to_quaternion(dcm_0_nominal)

    # compute the nominal angular velocity required to achieve the reference
    # attitude; first in inertial coordinates then body
    w_nominal_i = 2 * np.pi / period * normalize(cross(r_0, v_0))
    w_nominal = np.matmul(dcm_0_nominal, w_nominal_i)

    # provide some initial offset in both the attitude and angular velocity
    q_0 = quaternion_multiply(
        np.array(
            [0, np.sin(2 * np.pi / 180 / 2), 0,
             np.cos(2 * np.pi / 180 / 2)]), q_0_nominal)
    w_0 = w_nominal + np.array([0.005, 0, 0])

    # define a function that will model perturbations
    def perturb_func(satellite):
        return (satellite.approximate_gravity_gradient_torque() +
                satellite.approximate_magnetic_field_torque())

    # define a function that returns the desired state at any given point in
    # time (the initial state and a subsequent rotation about the body x, y, or
    # z axis depending upon which nominal angular velocity term is nonzero)
    def desired_state_func(t):
        if w_nominal[0] != 0:
            dcm_nominal = np.matmul(t1_matrix(w_nominal[0] * t), dcm_0_nominal)
        elif w_nominal[1] != 0:
            dcm_nominal = np.matmul(t2_matrix(w_nominal[1] * t), dcm_0_nominal)
        elif w_nominal[2] != 0:
            dcm_nominal = np.matmul(t3_matrix(w_nominal[2] * t), dcm_0_nominal)
        return dcm_nominal, w_nominal

    # construct three `Spacecraft` objects composed of all relevant spacecraft
    # parameters and objects that resemble subsystems on-board
    # 1st Spacecraft: no controller
    # 2nd Spacecraft: PD controller with perfect sensors and actuators
    # 3rd Spacecraft: PD controller with imperfect sensors and actuators

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

    # Simulate the behavior of all three spacecraft over time
    simulate(
        satellite=satellite_no_control,
        nominal_state_func=desired_state_func,
        perturbations_func=perturb_func,
        position_velocity_func=position_velocity_func,
        stop_time=6000,
        tag=r"(No Control)")

    simulate(
        satellite=satellite_perfect,
        nominal_state_func=desired_state_func,
        perturbations_func=perturb_func,
        position_velocity_func=position_velocity_func,
        stop_time=6000,
        tag=r"(Perfect Estimation \& Control)")

    simulate(
        satellite=satellite_noise,
        nominal_state_func=desired_state_func,
        perturbations_func=perturb_func,
        position_velocity_func=position_velocity_func,
        stop_time=6000,
        tag=r"(Actual Estimation \& Control)")


def simulate(satellite,
             nominal_state_func,
             perturbations_func,
             position_velocity_func,
             stop_time=6000,
             tag=""):

    # carry out the actual simulation and gather the results
    results = simulate_adcs(
        satellite=satellite,
        nominal_state_func=nominal_state_func,
        perturbations_func=perturbations_func,
        position_velocity_func=position_velocity_func,
        start_time=0,
        delta_t=1,
        stop_time=stop_time,
        verbose=True)

    # plt.rc('text', usetex=True)
    # plt.rc('font', family='serif')
    plt.rc("figure", dpi=120)
    plt.rc("savefig", dpi=120)

    # plot the desired results (logged at each delta_t)

    plt.figure(1)
    plt.subplot(411)
    plt.title(r"Evolution of Quaternion Components over Time {}".format(tag))
    plt.plot(results["times"], results["q_actual"][:, 0])
    plt.ylabel(r"$Q_0$")
    plt.subplot(412)
    plt.plot(results["times"], results["q_actual"][:, 1])
    plt.ylabel(r"$Q_1$")
    plt.subplot(413)
    plt.plot(results["times"], results["q_actual"][:, 2])
    plt.ylabel(r"$Q_2$")
    plt.subplot(414)
    plt.plot(results["times"], results["q_actual"][:, 3])
    plt.ylabel(r"$Q_3$")
    plt.xlabel(r"Time (s)")
    plt.subplots_adjust(
        left=0.08, right=0.94, bottom=0.08, top=0.94, hspace=0.3)

    plt.figure(2)
    plt.subplot(311)
    plt.title(r"Evolution of Angular Velocity over Time {}".format(tag))
    plt.plot(results["times"], results["w_actual"][:, 0], label="actual")
    plt.plot(
        results["times"],
        results["w_desired"][:, 0],
        label="desired",
        linestyle="--")
    plt.ylabel(r"$\omega_x$ (rad/s)")
    plt.legend()
    plt.subplot(312)
    plt.plot(results["times"], results["w_actual"][:, 1], label="actual")
    plt.plot(
        results["times"],
        results["w_desired"][:, 1],
        label="desired",
        linestyle="--")
    plt.ylabel(r"$\omega_y$ (rad/s)")
    plt.legend()
    plt.subplot(313)
    plt.plot(results["times"], results["w_actual"][:, 2], label="actual")
    plt.plot(
        results["times"],
        results["w_desired"][:, 2],
        label="desired",
        linestyle="--")
    plt.ylabel(r"$\omega_z$ (rad/s)")
    plt.xlabel(r"Time (s)")
    plt.legend()
    plt.subplots_adjust(
        left=0.08, right=0.94, bottom=0.08, top=0.94, hspace=0.3)

    plt.figure(3)
    plt.subplot(311)
    plt.title(r"Angular Velocity of Reaction Wheels over Time {}".format(tag))
    plt.plot(results["times"], results["w_rxwls"][:, 0])
    plt.ylabel(r"$\omega_1$ (rad/s)")
    plt.subplot(312)
    plt.plot(results["times"], results["w_rxwls"][:, 1])
    plt.ylabel(r"$\omega_2$ (rad/s)")
    plt.subplot(313)
    plt.plot(results["times"], results["w_rxwls"][:, 2])
    plt.ylabel(r"$\omega_3$ (rad/s)")
    plt.xlabel(r"Time (s)")
    plt.subplots_adjust(
        left=0.08, right=0.94, bottom=0.08, top=0.94, hspace=0.3)

    plt.figure(4)
    plt.subplot(311)
    plt.title(r"Perturbation Torques over Time {}".format(tag))
    plt.plot(results["times"], results["M_perturb"][:, 0])
    plt.ylabel(r"$M_x (N \cdot m)$")
    plt.subplot(312)
    plt.plot(results["times"], results["M_perturb"][:, 1])
    plt.ylabel(r"$M_y (N \cdot m)$")
    plt.subplot(313)
    plt.plot(results["times"], results["M_perturb"][:, 2])
    plt.ylabel(r"$M_z (N \cdot m)$")
    plt.xlabel(r"Time (s)")
    plt.subplots_adjust(
        left=0.08, right=0.94, bottom=0.08, top=0.94, hspace=0.3)

    plt.figure(5)
    DCM_actual = np.empty(results["DCM_desired"].shape)
    for i, q in enumerate(results["q_actual"]):
        DCM_actual[i] = quaternion_to_dcm(q)

    k = 1
    for i in range(3):
        for j in range(3):
            plot_num = int("33{}".format(k))
            plt.subplot(plot_num)
            if k == 2:
                plt.title(
                    r"Evolution of DCM Components over Time {}".format(tag))
            plt.plot(results["times"], DCM_actual[:, i, j], label="actual")
            plt.plot(
                results["times"],
                results["DCM_desired"][:, i, j],
                label="desired",
                linestyle="--")
            element = "T_{" + str(i + 1) + str(j + 1) + "}"
            plt.ylabel("$" + element + "$")
            if k >= 7:
                plt.xlabel(r"Time (s)")
            plt.legend()
            k += 1
    plt.subplots_adjust(
        left=0.08, right=0.94, bottom=0.08, top=0.94, hspace=0.25, wspace=0.3)

    plt.figure(6)
    k = 1
    for i in range(3):
        for j in range(3):
            plot_num = int("33{}".format(k))
            plt.subplot(plot_num)
            if k == 2:
                plt.title(
                    r"Actual vs Estimated Attitude over Time {}".format(tag))
            plt.plot(results["times"], DCM_actual[:, i, j], label="actual")
            plt.plot(
                results["times"],
                results["DCM_estimated"][:, i, j],
                label="estimated",
                linestyle="--",
                color="y")
            element = "T_{" + str(i + 1) + str(j + 1) + "}"
            plt.ylabel("$" + element + "$")
            if k >= 7:
                plt.xlabel(r"Time (s)")
            plt.legend()
            k += 1
    plt.subplots_adjust(
        left=0.08, right=0.94, bottom=0.08, top=0.94, hspace=0.25, wspace=0.3)

    plt.figure(7)
    plt.subplot(311)
    plt.title(r"Actual vs Estimated Angular Velocity over Time {}".format(tag))
    plt.plot(results["times"], results["w_actual"][:, 0], label="actual")
    plt.plot(
        results["times"],
        results["w_estimated"][:, 0],
        label="estimated",
        linestyle="--",
        color="y")
    plt.ylabel(r"$\omega_x$ (rad/s)")
    plt.legend()
    plt.subplot(312)
    plt.plot(results["times"], results["w_actual"][:, 1], label="actual")
    plt.plot(
        results["times"],
        results["w_estimated"][:, 1],
        label="estimated",
        linestyle="--",
        color="y")
    plt.ylabel(r"$\omega_y$ (rad/s)")
    plt.legend()
    plt.subplot(313)
    plt.plot(results["times"], results["w_actual"][:, 2], label="actual")
    plt.plot(
        results["times"],
        results["w_estimated"][:, 2],
        label="estimated",
        linestyle="--",
        color="y")
    plt.ylabel(r"$\omega_z$ (rad/s)")
    plt.xlabel(r"Time (s)")
    plt.legend()
    plt.subplots_adjust(
        left=0.08, right=0.94, bottom=0.08, top=0.94, hspace=0.3)

    plt.figure(8)
    plt.subplot(311)
    plt.title(r"Commanded vs Applied Torques over Time {}".format(tag))
    plt.plot(results["times"], results["M_applied"][:, 0], label="applied")
    plt.plot(
        results["times"],
        results["M_ctrl"][:, 0],
        label="commanded",
        linestyle="--")
    plt.ylabel(r"$M_x (N \cdot m)$")
    plt.legend()
    plt.subplot(312)
    plt.plot(results["times"], results["M_applied"][:, 1], label="applied")
    plt.plot(
        results["times"],
        results["M_ctrl"][:, 1],
        label="commanded",
        linestyle="--")
    plt.ylabel(r"$M_y (N \cdot m)$")
    plt.legend()
    plt.subplot(313)
    plt.plot(results["times"], results["M_applied"][:, 2], label="applied")
    plt.plot(
        results["times"],
        results["M_ctrl"][:, 2],
        label="commanded",
        linestyle="--")
    plt.ylabel(r"$M_z (N \cdot m)$")
    plt.xlabel(r"Time (s)")
    plt.legend()
    plt.subplots_adjust(
        left=0.08, right=0.94, bottom=0.08, top=0.94, hspace=0.3)

    plt.show()


if __name__ == "__main__":
    main()
