import gym
from docking import Docking

from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv

config = {
    "t_step": 0.01, # length of timestep
    "env_size": 100, # environment will be env_size x env_size meters(?)
    "desired_vel": 1, # desired velocity in m/s

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
