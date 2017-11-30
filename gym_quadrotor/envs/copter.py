""" This file contains classes and functions that are used for the simulation of the quadrotor
    helicopter.
"""
from collections import namedtuple

import numpy as np
from .propeller import Propeller
from .geo import make_quaternion


AccelerationData = namedtuple("AccelerationResult", ("linear", "angular", "rotor_speeds"))


class CopterStatus(object):
    """
    Describes the dynamical state of a quadcopter in flight. This includes position, velocity, attitude and
    angular velocity as well as the rotor speeds.
    """
    def __init__(self, pos=None, vel=None, att=None, avel=None, rspeed=None):
        self.position = np.zeros(3) if pos is None else pos
        self.velocity = np.zeros(3) if vel is None else vel
        self._attitude = np.zeros(3) if att is None else att
        self.angular_velocity = np.zeros(3) if avel is None else avel
        self.rotor_speeds = np.array([1, 1, -1, -1.0]) * 180.0 if rspeed is None else rspeed
        self._rotation_matrix = None

    @property
    def altitude(self):
        """ The current altitude (z position) """
        return self.position[2]

    @property
    def attitude(self):
        """ Gets the current attitude """
        return self._attitude

    @attitude.setter
    def attitude(self, value):
        """ Sets the current attitude and resets the cached rotation matrix. """
        self._attitude = value
        self._rotation_matrix = None

    @property
    def rotation_matrix(self):
        """
        Gets the rotation matrix based on the current quadcopter attitude. Once calculated the value is
        cached until the attitude setter is called again.
        """
        if self._rotation_matrix is None:
            self._rotation_matrix = make_quaternion(self.attitude[0], self.attitude[1], self.attitude[2]).rotation_matrix
        return self._rotation_matrix

    def to_world_direction(self, axis):
        """
        Transforms a directional vector from local into world coordinates.
        """
        return np.dot(self.rotation_matrix, axis)

    def to_world_position(self, pos):
        """
        Transforms a position vector from local into world coordinates.
        """
        return self.to_world_direction(pos) + self.position

    def to_local_direction(self, axis):
        """
        Transforms a directional vector from world to local coordinates.
        """
        return np.dot(np.linalg.inv(self.rotation_matrix), axis)

    def to_local_position(self, pos):
        """
        Transforms a position vector from world to local coordinates.
        """
        return self.to_local_direction(pos - self.position)

    def __repr__(self):
        return "CopterStatus(%r, %r, %r, %r, %r)" % (self.position, self.velocity, self.attitude, self.angular_velocity,
                                                     self.rotor_speeds)

    def __str__(self):
        return "CopterStatus(pos=%s, vel=%s, att=%s, avel=%s, rspeed=%s)" % (self.position, self.velocity, self.attitude,
                                                                             self.angular_velocity, self.rotor_speeds)


class CopterSetup(object):
    """
    This class collects the setup of a quadcopter, which are the (immutable) physical parameters.
    These are aerodynamical coefficients (a. b. mu, lm), the motor torque, masses, interia and dimensions of the
    quadcopter.
    """
    def __init__(self):
        # rotor aerodynamcis coefficients
        self.a = 5.324e-5
        self.b = 8.721e-7

        # more parameters
        self.mu = np.array([1, 1, 1, 1]) * 1e-7
        self.lm = np.array([1, 1, 1, 1]) * 1e-7

        # motor data
        self.motor_torque = 0.040  # [NM]

        self.l = 0.31   # Arm length
        self.m = 0.723  # mass
        self.J = 7.321e-5   # Rotor inertia
        # inverse Inertia (assumed to be diagonal)
        self.iI = np.linalg.inv([[8.678e-3, 0, 0], [0, 8.678e-3, 0], [0, 0, 3.217e-2]])

        cfg = {'a': self.a, 'b': self.b, 'lm': self.lm, 'mu': self.mu, 'axis': [0.0, 0.0, -1.0]}
        P1 = Propeller(d=1,  p = self.l*np.array([1, 0, 0.0]), **cfg)
        P2 = Propeller(d=1,  p = self.l*np.array([-1, 0, 0.0]), **cfg)
        P3 = Propeller(d=-1, p = self.l*np.array([0, 1, 0.0]), **cfg)
        P4 = Propeller(d=-1, p = self.l*np.array([0, -1, 0.0]), **cfg)

        self.propellers = [P1, P2, P3, P4]


def calc_forces(status, setup):
    """
    Calculate the forces (and moments) that act on a quadcopter in a given configuration.
    :param CopterStatus status: The current status / situation of the copter.
    :param CopterSetup setup: The copter setup, its physical parameters.
    :return: A tuple consisting of a force and a moment vector in world coordinates,
    and a list of (scalar) moments acting on the different rotors
    """
    moment = np.zeros(3)
    force  = np.zeros(3)
    rot_t  = []

    # calculate contributions for each propeller
    for p, w in zip(setup.propellers, status.rotor_speeds):
        f, m, ma = p.get_dynamics(w, status)
        force  += f 
        moment += m
        rot_t  += [ma]

    force += setup.m * np.array([0.0, 0.0, -9.81])

    return force, moment, rot_t


def calc_accelerations(setup, status, control):
    """
    Calculates the acceleration action of the copter in a given status for certain control inputs.
    :param CopterStatus status: The current status / situation of the copter.
    :param CopterSetup setup: The copter setup, its physical parameters.
    :param np.ndarray control: Rotor control as an array with values in [0, 1].
    :return: A named tuple (AccelerationData) of linear and angular acceleration (world coordinates) and a list of
    accelerations for the rotors.
    """
    force, moment, ma = calc_forces(status, setup)

    rot_acc = np.array(ma)
    motor_torque = np.clip(control, 0.0, 1.0) * setup.motor_torque

    for i, w in enumerate(ma):
        torque     = setup.propellers[i].direction * motor_torque[i]
        rot_acc[i] = (w + torque) / setup.J
        # the motor creates the reverse torque on the helicopter.
        moment    -= torque * status.to_world_direction(setup.propellers[i].axis)
    
    lin_acc = force / setup.m
    ang_acc = np.dot(setup.iI, moment)

    return AccelerationData(lin_acc, ang_acc, rot_acc)


def simulate(status, params, control, dt):
    """
    Updates the copter status based on the current control.
    :param CopterStatus status: The current status / situation of the copter.
    :param CopterSetup params: The copter setup, its physical parameters.
    :param np.ndarray control: Rotor control as an array with values in [0, 1].
    :param dt: Time step in seconds.
    """
    acceleration = calc_accelerations(params, status, control)
    # position update
    status.position += status.velocity * dt + 0.5 * acceleration.linear * dt * dt
    status.velocity += acceleration.linear * dt

    # angle update. Needs to be done in local coordinates.
    aa = status.to_local_direction(acceleration.angular)
    status.attitude += status.angular_velocity * dt + 0.5 * aa * dt * dt
    status.angular_velocity += aa * dt

    # rotor speed update
    status.rotor_speeds += acceleration.rotor_speeds * dt


def calculate_equilibrium_acceleration(setup, strength):
    """
    Calculates the acceleration given a certain control strength
    (using a control vector with equal values). This function simulates
    a few timesteps but resets the copters position and velocity to their initial values
    after each step. This allows the rotor speeds to reach an equilibrium, for which
    the resulting data is calculated.
    :param CopterSetup setup: The copter setup, its physical parameters.
    :param strength: The motor control signal, a scalar that will be sent to all motors.
    :return AccelerationData: Linear and angular acceleration as well as the final rotor speeds.
    """
    control = np.ones(4) * strength
    status = CopterStatus()

    for i in range(50):
        #print(status)
        simulate(status, setup, control, 0.1)
        status.position = np.array([0.0, 0.0, 0.0])
        status.velocity = np.array([0.0, 0.0, 0.0])
        status.attitude = np.array([0.0, 0.0, 0.0])
        status.angular_velocity = np.array([0.0, 0.0, 0.0])

    result = calc_accelerations(setup, status, control)
    return AccelerationData(linear=result[0], angular=result[1], rotor_speeds=status.rotor_speeds)
