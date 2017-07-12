from gym_quadrotor.envs.copter_env import coupled_motor_action
from gym_quadrotor.envs.copter import *
import numpy as np

def test(yaw):
    startatt = [0.0, 0.0, yaw]
    test_couple_direction(0, startatt)
    test_couple_direction(1, startatt)
    test_couple_direction(2, startatt)

def test_couple_direction(index, startat = None):
    setup = CopterSetup()
    copterstatus = CopterStatus()
    if startat is not None:
        copterstatus.attitude = startat
    base = np.zeros(3)
    base[index] = 1.0
    control = coupled_motor_action(1.0, *base)
    start_attitude = np.copy(copterstatus.attitude)
    for i in range(10):
        simulate(copterstatus, setup, control, 0.01)
    delta = copterstatus.attitude - start_attitude
    #assert(delta[index] > 0.0)
    nd = delta / delta[index]
    ref = np.zeros(3)
    ref[index] = 1
    err = (nd - ref) ** 2
    assert (err < 1e-12).all(), (err, index, nd, ref)

for yaw in np.linspace(0, 2*np.pi):
    test( yaw )
