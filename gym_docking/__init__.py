import numpy as np

from gym.envs.registration import register

config = {
    t_step = 0.01
    env_size = 100 #environment will be env_size x env_size meters(?)
}

register(
    id = 'docking-v0',
    entry_point = 'gym_docking.envs:docking',
    kwargs = {'env_config': config}
)
