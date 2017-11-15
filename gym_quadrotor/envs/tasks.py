import math
import numpy as np
from gym_quadrotor.envs import geo


class CopterTask(object):
    """
    CopterTask is the base class for task objects that can be added to a CopterEnv environment.
    A task defines a reward function and may add additional dimensions to the state. Each task
    has an assigned weight that modulates the reward. A task can set itself up as failed, which
    will cause the current learning episode to terminate.

    For consistent random number generation each CopterTask gets its own RandomState object
    that is supplied by the current environment.
    """

    def __init__(self, weight=1.0):
        self.has_failed = False
        self.weight = weight
        self._reward = 0.0
        self._random_state = None

    def seed(self, seed):
        """
        Initialise the task local random number generator with the given `seed`.
        """
        self._random_state = np.random.RandomState(seed)

    @property
    def rng(self):
        return self._random_state

    def reward(self):
        """
        Gets the weighted reward for the last time step.
        """
        return self._reward * self.weight

    def reset(self, status):
        """
        Called whenever the environment resets.
        """
        self.has_failed = False
        self._reset(status)

    def _reset(self, status):
        """
        This gets called when the environment has been reset. It gets passed
        the status of the quad copter after the reset.
        :param status: CopterStatus of copter after reset.
        :return: Nothing.
        """
        pass

    def draw(self, viewer, status):
        """
        Override this function if additional data should be drawn to the visualizations for this task.
        :param viewer: The viewer object in which the visualization is drawn.
        :param status: Current status of the copter.
        """
        pass

    def step(self, status, control):
        """
        Called for each step of the simulation.
        :param status: Copter status.
        :param control: Current control input.
        :return: Nothing.
        """
        self._reward = self._step(status, control)

    def _step(self, status, control):
        """
        Called for each step of the simulation. Calculate reward and update the task.
        This function has to be implemented in derived classes.
        :param status: Copter status.
        :param control: Current control input.
        :return: Reward for the current step.
        """
        raise NotImplementedError()

    def get_state(self, status):
        """
        Gets the state of the task, i.e. all the data that should be appended to the
        state visible to the learning agent.
        :param status: The status of the copter.
        :return: An array with the state variables.
        """
        return np.array([])


class StayAliveTask(CopterTask):
    def __init__(self, **kwargs):
        super(StayAliveTask, self).__init__(**kwargs)

    def _step(self, status, control):
        reward = 0
        if status.altitude < 0.0 or status.altitude > 10:
            reward = -10
            self.has_failed = True
        #elif copterstatus.altitude < 0.2 or copterstatus.altitude > 9.8:
        #    reward = -0.1
        return reward


class FlySmoothlyTask(CopterTask):
    def __init__(self, **kwargs):
        super(FlySmoothlyTask, self).__init__(**kwargs)

    def _step(self, status, control):
        # reward for keeping velocities low
        velmag = np.mean(np.abs(status.angular_velocity))
        reward = max(0.0, 0.1 - velmag)

        # reward for constant control
        cchange = np.mean(np.abs(control - self._last_control))
        reward += max(0, 0.1 - 10*cchange)

        self._last_control = control

        return reward / 0.2  # normed to 1

    def _reset(self, status):
        self._last_control = np.zeros(4)


class HoldAngleTask(CopterTask):
    def __init__(self, threshold, fail_threshold, **kwargs):
        super(HoldAngleTask, self).__init__(**kwargs)
        self.threshold = threshold
        self.fail_threshold = fail_threshold

    def _step(self, status, control):
        # reward calculation
        attitude = status.attitude
        err = np.mean(np.abs(attitude - self.target))
        # positive reward for not falling over
        reward = max(0.0, 0.2 * (1 - err / self.fail_threshold))
        if err < self.threshold:
            merr = np.mean(np.abs(attitude - self.target))  # this is guaranteed to be smaller than err
            rerr = merr / self.threshold
            reward += 1.1 - rerr

        if err > self.fail_threshold:
            reward = -10
            self.has_failed = True

        # change target
        if self.rng.rand() < 0.01:
            self.target += self.rng.uniform(low=-3, high=3, size=(3,)) * math.pi / 180

        return reward

    # TODO how do we pass np_random stuff
    def _reset(self, status):
        self.target = self.rng.uniform(low=-10, high=10, size=(3,)) * math.pi / 180

    def draw(self, viewer, copterstatus):
        # draw target orientation
        start = (copterstatus.position[0], copterstatus.altitude)
        rotated = np.dot(geo.make_quaternion(self.target[0], self.target[1], self.target[2]).rotation_matrix,
                         [0, 0, 0.5])
        err = np.max(np.abs(copterstatus.attitude - self.target))
        if err < self.fail_threshold:
            color = (0.0, 0.5, 0.0)
        else:
            color = (1.0, 0.0, 0.0)
        viewer.draw_line(start, (start[0]+rotated[0], start[1]+rotated[2]), color=color)

    def get_state(self, status):
        return np.array([self.target])


class HoverTask(CopterTask):
    def __init__(self, threshold, fail_threshold, **kwargs):
        super(HoverTask, self).__init__(**kwargs)
        self.threshold = threshold
        self.fail_threshold = fail_threshold
        self.target_altitude = 1.0

    def _step(self, status, control):
        attitude = status.attitude
        # yaw is irrelevant for hovering
        err = np.mean(np.abs(attitude[0:2]))
        perr = np.abs(status.altitude - self.target_altitude)
        # positive reward for not falling over
        reward = max(0.0, 1.0 - (err / self.fail_threshold)**2)
        reward += max(0.0, 1.0 - np.mean(status.velocity ** 2)) * 0.25
        reward += max(0.0, 1.0 - perr**2) * 0.25

        if err > self.fail_threshold or perr > 1:
            reward = -10
            self.has_failed = True

        return reward

    def draw(self, viewer, copterstatus):
        # draw target orientation
        start = (copterstatus.position[0], copterstatus.altitude)
        rotated = np.dot(geo.make_quaternion(0, 0, 0).rotation_matrix,
                         [0, 0, 0.5])
        err = np.mean(np.abs(copterstatus.attitude))
        if err < self.threshold:
            color = (0.0, 0.5, 0.0)
        else:
            color = (1.0, 0.0, 0.0)
        viewer.draw_line(start, (start[0]+rotated[0], start[1]+rotated[2]), color=color)

    def _reset(self, status):
        self.target_altitude = status.altitude + self.rng.uniform(low=-0.2, high=0.2)

    def get_state(self, status):
        return np.array([status.altitude - self.target_altitude])
