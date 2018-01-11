import math
import gym
from gym import spaces
from gym.spaces import Box
from gym.utils import seeding

from .copter import *
from . import geo

from gym.envs.classic_control import rendering

def _draw_ground(viewer, center):
    """ helper function that draws the ground indicator.
        The parameter center indicates where the screen center is supposed to be.
    """
    viewer.draw_line((-10+center, 0.0), (10+center, 0.0))
    for i in range(-8, 10, 2):
        pos = round(center / 2) * 2
        viewer.draw_line((pos+i, 0.0), (pos+i-1, -1.0))

def _draw_copter(viewer, setup, status):
    # transformed main axis
    trafo = status.rotation_matrix
    start = (status.position[0], status.altitude)
    def draw_prop(p):
        rotated = np.dot(trafo, setup.propellers[p].position)
        end     = (start[0]+rotated[0], start[1]+rotated[2])
        viewer.draw_line(start, end)
        copter = rendering.make_circle(.1)
        copter.set_color(0,0,0)
        copter.add_attr(rendering.Transform(translation=end))
        viewer.add_onetime(copter)
   
    # draw current orientation
    rotated = np.dot(trafo, [0, 0, 0.5])
    viewer.draw_line(start, (start[0]+rotated[0], start[1]+rotated[2]))

    for i in range(4): draw_prop(i)

class CopterEnvBase(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self, strict_actions=False):
        self.viewer = None
        self.setup = CopterSetup()
        self._seed()
        self._strict_action_space = strict_actions
        self.allowed_fly_range = Box(np.array([-10, -10, 0]), np.array([10, 10, 10]))
        self.observation_space = Box(-np.inf, np.inf, (4*3 + 4))

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        if self._strict_action_space:
            assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        else:
            action = np.clip(action, self.action_space.low, self.action_space.high)
        
        self._control = self._control_from_action(action)
        for i in range(2):
            simulate(self.copterstatus, self.setup, self._control, 0.01)

        done = not self.allowed_fly_range.contains(self.copterstatus.position)
        reward = -1 if done else 0
        return self._get_state(), reward, done, {"rotor-speed": self.copterstatus.rotor_speeds}

    def _get_state(self):
        s = self.copterstatus
        return np.concatenate([s.position, s.velocity, s.attitude, s.angular_velocity, s.rotor_speeds])

    def _reset(self):
        self.copterstatus = CopterStatus()
        return self._get_state()

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        if self.viewer is None:
            self.viewer = rendering.Viewer(500, 500)
            self.center = self.copterstatus.position[0]
        
        self.center = 0.9*self.center + 0.1*self.copterstatus.position[0]
        self.viewer.set_bounds(-7 + self.center, 7 + self.center,-1, 13)

        
        # draw ground
        _draw_ground(self.viewer, self.center)
        _draw_copter(self.viewer, self.setup, self.copterstatus)

        # finally draw stuff related to the tasks
        for task in self._tasks: task.draw(self.viewer, self.copterstatus)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def _control_from_action(self, action):
        raise NotImplementedError()


class CopterEnv_v1(CopterEnvBase):
    def __init__(self):
        CopterEnvBase.__init__(self)
        self.action_space = spaces.Box(0, 1, (4,))

    def _control_from_action(self, action):
        return np.array(action) + 3.3


class CopterEnvEuler_v1(CopterEnvBase):
    def __init__(self):
        CopterEnvBase.__init__(self)
        self.action_space = spaces.Box(np.array([0.0, -1.0, -1.0, -1.0]), np.ones(4))

    def _control_from_action(self, action):
        # TODO add tests to show that these arguments are ordered correctly
        total = action[0] * 4
        roll  = action[1] / 2   # rotation about x axis
        pitch = action[2] / 2  # rotation about y axis
        yaw   = action[3] / 2
        return coupled_motor_action(total, roll, pitch, yaw) + 3.3

def coupled_motor_action(total, roll, pitch, yaw):
    a = total / 4 - pitch / 2 + yaw / 4
    b = total / 4 + pitch / 2 + yaw / 4
    c = total / 4 + roll  / 2 - yaw / 4
    d = total / 4 - roll  / 2 - yaw / 4
    return np.array([a, b, c, d])
