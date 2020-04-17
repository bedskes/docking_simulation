import gym
import numpy as np

from gym_docking.objects.Boat import Boat

class Docking(gym.env):
    def __init__(self, env_config):
        for key in env_config:
            setattr(self, key, env_config[key])
        # all observations and actions are normalized to aid learning
        self.nobservations = 17
        # set limits on what actions the boat can take
        # structured as [propeller velocity, rudder angle]
        self.action_space = gym.spaces.Box(low=np.array([0, -1]),
                                           high=np.array([1, 1]),
                                           dtype=np.float32)
        # structured as []
        self.observation_space = gym.spaces.Box(
            low=np.array([-1]*self.nobservations),
            high=np.array([1]*self.nobservations),
            dtype=np.float32)

        self.reset()
        self.generate()

    def generate(self):
        #place dock and vessel randomly in the map
        dock = np.random.rand(2, 1)*self.env_size
        init_pos = np.random.rand(3, 1)*self.env_size
        init_pos[2] = 0
        init_vel = np.array([0,0,0])
        init_theta = np.random.rand(3, 1)*2*np.pi
        init_theta[0:2] = 0
        init_w = 0
        init_t = 0

        #TODO define path so that we can easily determine parameters
        self.path = Path([initial_pos, dock])
        #TODO add sea state to state?
        self.vessel = Boat(init_pos, init_vel, init_theta, init_w, init_t)

        #initialize waves
        self.wave = (1, 1, 1, 0, 0)
        #initialize current
        #TODO determine current format
        self.current = 0

    def step(self, action):
        # Simulate dynamics one time-step and save action and state
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.vessel.step(self.t_step, action[0], action[1], self.wave, self.current)
        vessel_state = np.hstack([vessel.pos, vessel.v, vessel.theta, vessel.w])
        self.past_states.append(np.copy(self.vessel.state))
        self.past_actions.append(action)

        # Observe normalized control errors
        obs = self.observe(action)
        self.past_obs.append(obs)

        # Calculate reward based on observation
        done, step_reward = self.step_reward(obs, action)
        info = {}

        # Save sim time info
        self.total_t_steps += 1
        self.time.append(self.total_t_steps * self.t_step)

        return obs, step_reward, done, info

    def observe(self, action):
        #placeholder until we determine what our observations are
        return obs

    def step_reward(self, obs, action):
        #placeholder until reward function is determined
        return done, reward
