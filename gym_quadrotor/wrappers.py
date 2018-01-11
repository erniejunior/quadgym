from gym.core import Wrapper, ObservationWrapper, RewardWrapper
from gym.spaces import Box
from gym_quadrotor.envs.copter import CopterStatus
import numpy as np


class ResetWrapper(Wrapper):
    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        return self._reset(**kwargs)

    def _reset(self, **kwargs):
        raise NotImplementedError


class CopterstateResetWrapper(ResetWrapper):
    def __init__(self, env, ang_veloc_range, veloc_range, xy_pos_range, min_height, max_height, att_range):
        ResetWrapper.__init__(self, env)
        self.ang_veloc_range = ang_veloc_range
        self.veloc_range = veloc_range
        self.xy_pos_range = xy_pos_range
        self.min_height = min_height
        self.max_height = max_height
        self.att_range = att_range

    def _reset(self, **kwargs):
        self.unwrapped.copterstatus = CopterStatus()

        self.unwrapped.copterstatus.angular_velocity = self.unwrapped.np_random.uniform(low=-self.ang_veloc_range,
                                                                            high=self.ang_veloc_range, size=(3,))
        self.unwrapped.copterstatus.velocity = self.unwrapped.np_random.uniform(low=-self.veloc_range, high=self.veloc_range,
                                                                    size=(3,))
        self.unwrapped.copterstatus.position = self.unwrapped.np_random.uniform(
            low=[-self.xy_pos_range, -self.xy_pos_range, self.min_height],
            high=[self.xy_pos_range, self.xy_pos_range, self.max_height])
        self.unwrapped.copterstatus.attitude = self.unwrapped.np_random.uniform(low=-self.att_range,
                                                                    high=self.att_range, size=(3,))
        self.unwrapped.center = self.unwrapped.copterstatus.position[0]
        return self.unwrapped._get_state()


class CleanResetWrapper(CopterstateResetWrapper):
    def __init__(self, env):
        CopterstateResetWrapper.__init__(self, env, 0, 0, 0, 1, 1, 0)


class FriendlyResetWrapper(CopterstateResetWrapper):
    def __init__(self, env):
        CopterstateResetWrapper.__init__(self, env, 0.1, 0.1, 2, 1, 2, np.radians(5))


class AggressiveyResetWrapper(CopterstateResetWrapper):
    def __init__(self, env):
        CopterstateResetWrapper.__init__(self, env, 1, 1, 5, 0.5, 5, np.radians(180))


class OnlyIMUAndHeightObservations(ObservationWrapper):
    def __init__(self, env):
        ObservationWrapper.__init__(self, env)
        self.unwrapped.observation_space = Box(-np.inf, np.inf, (7,))

    def _observation(self, observation):
        return observation[[2, 6, 7, 8, 9, 10, 11]]


class PosDistReward(RewardWrapper):
    def __init__(self, env, hoverpos=[0, 0, 2]):
        RewardWrapper.__init__(self, env)
        self.hoverpos = hoverpos

    def _reward(self, reward):
        return np.sqrt(np.sum((self.unwrapped.copterstatus.position - [0, 0, 2]) ** 2))


class FullstateDistReward(RewardWrapper):
    def __init__(self, env, hoverpos=[0, 0, 2]):
        RewardWrapper.__init__(self, env)
        self.hoverpos = hoverpos
        self.targetstate = np.concatenate([hoverpos, np.zeros(3), np.zeros(3), np.zeros(3)])

    def _reward(self, reward):
        s = self.unwrapped.copterstatus
        curstate = np.concatenate([s.position, s.velocity, s.attitude, s.angular_velocity])
        return np.sqrt(np.sum((curstate - self.targetstate) ** 2))


class AngularDistReward(RewardWrapper):
    def __init__(self, env):
        RewardWrapper.__init__(self, env)

    def _reward(self, reward):
        s = self.unwrapped.copterstatus
        curstate = np.concatenate([s.attitude, s.angular_velocity])
        return np.sqrt(np.sum(curstate ** 2))


class InverseExpDistReward(RewardWrapper):
    def __init__(self, env, threshold=0.1):
        RewardWrapper.__init__(self, env)
        self.threshold = threshold

    def _reward(self, reward):
        return 1 if reward < self.threshold else self.threshold / reward


class InverseDistReward(RewardWrapper):
    def __init__(self, env):
        RewardWrapper.__init__(self, env)

    def _reward(self, reward):
        return -reward
