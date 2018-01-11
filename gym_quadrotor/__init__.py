from gym.envs.registration import register
from .wrappers import CleanResetWrapper, FriendlyResetWrapper, AggressiveyResetWrapper, OnlyIMUAndHeightObservations, \
    PosDistReward, AngularDistReward, FullstateDistReward, InverseExpDistReward, InverseDistReward

register(
    id='Quadrotor-v0',
    entry_point='gym_quadrotor.envs:CopterEnv_v0',
    max_episode_steps=1000
)

register(
    id='QuadrotorTorqueControl-v0',
    entry_point='gym_quadrotor.envs:CopterEnvEuler_v0',
    max_episode_steps=1000
)

register(
    id='Quadrotor-v1',
    entry_point='gym_quadrotor.envs:CopterEnv_v1',
    max_episode_steps=1000
)

register(
    id='QuadrotorTorqueControl-v1',
    entry_point='gym_quadrotor.envs:CopterEnvEuler_v1',
    max_episode_steps=1000
)
