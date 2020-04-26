import gym
from docking import Docking
import numpy as np

from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv

config = {
    "t_step": 0.01, # length of timestep
    "env_size": 100, # environment will be env_size x env_size meters(?)
    "desired_vel": 1, # desired cruise velocity in m/s

    # reward function parameters
    "r_prop_change":-0.05, # penalize large changes in propeller speed
    "r_rudder_change":-0.05, # penalize large changes in rudder angle
    "r_vel_error":-0.05, # penalize velocity error
    "r_track_error":-0.05, # penalize cross track error
    "r_heading_error":-0.05, # penalize heading error

    # scaling parameters (observation values will be between -max and max)
    "max_prop_vel":200, # rev/s (?)
    "max_prop_vel_change":100, # rev/s (?)
    "max_rudder_angle":35*(np.pi/180), # radians (?)
    "max_rudder_angle_change":np.pi/10, # radians(?)
    "max_velocity":20, #m/s
    "max_yaw":np.pi, # radians
    "max_yaw_rate":np.pi/8, # rad/s
    "max_velocity_error":2, # m/sec
    "max_track_error":25, # m
    "max_heading_error":np.pi, # radians
    "max_wave_amp":2, # m
    "max_wave_direction":np.pi #radians
}

hyperparams = {
        'n_steps': 6144,
        'nminibatches': 1024,
        'learning_rate': 5e-5,
        'lam': 0.95,
        'gamma': 0.999,
        'noptepochs': 4,
        'cliprange': 0.2,
        'ent_coef': 0.01,
        'verbose': 2
}

if __name__ == '__main__':
    env = DummyVecEnv([lambda: Docking(config)])
    model = PPO2(MlpPolicy, env, **hyperparams)

    time_steps = int(1e5)
    model.learn(total_timesteps = time_steps)
    model.save("./model.pkl")
