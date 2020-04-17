import os

import gym
import gym_docking

from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy

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
    env = gym.make('docking_v0')
    model = PPO2(MlpPolicy, env)

    time_steps = 1e5
    model.learn(total_timesteps = time_steps)
    model.save("./model.pkl")
