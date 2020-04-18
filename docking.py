import gym
import numpy as np

from Boat import Boat
from Path import Path
from Wave import Wave

class Docking(gym.Env):
    def __init__(self, env_config):
        for key in env_config:
            setattr(self, key, env_config[key])
        # all observations and actions are normalized to aid learning
        self.nobservations = 11
        # set limits on what actions the boat can take
        # structured as [propeller velocity, rudder angle]
        self.action_space = gym.spaces.Box(low=np.array([0, -1]),
                                           high=np.array([1, 1]),
                                           dtype=np.float32)
        # structured as [propeller velocity,
        #                rudder angle,
        #                velocity (x, y),
        #                yaw, yaw rate,
        #                velocity error,
        #                cross track error,
        #                heading error,
        #                wave amplitude,
        #                wave direction]
        self.observation_space = gym.spaces.Box(
            low=np.array([-1]*self.nobservations),
            high=np.array([1]*self.nobservations),
            dtype=np.float32)

        self.reset()
        self.generate()

    def generate(self):
        #place dock and vessel randomly in the map
        self.dock = np.random.rand(2)*self.env_size
        init_pos = np.random.rand(3)*self.env_size
        init_pos[2] = 0
        init_vel = np.array([0,0,0])
        init_theta = np.random.rand(3)*2*np.pi
        init_theta[0:2] = 0
        init_w = np.array([0, 0, 0])
        init_t = 0

        self.path = Path([init_pos[:2], self.dock])
        #TODO add sea state to state?
        self.vessel = Boat(init_pos, init_vel, init_theta, init_w, init_t)

        #initialize waves
        self.wave = Wave(1, 1, 1, 0, 0)
        #initialize current
        #TODO determine current format
        self.current = np.array([0, 0, 0])

    def step(self, action):
        # Simulate dynamics one time-step and save action and state
        action = np.clip(action, self.action_space.low, self.action_space.high)
        # TODO action comes out normalized, need to convert to actual state space
        self.vessel.step_time(self.t_step, action[0], action[1], self.wave, self.current)
        vessel_state = np.hstack([self.vessel.pos, self.vessel.v, self.vessel.theta, self.vessel.w])
        self.past_states.append(np.copy(vessel_state))
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
        vel_error = np.linalg.norm(self.vessel.v[:2]) - self.desired_vel
        track_error = self.path.distance_to(self.vessel.pos[:2])
        head_error = self.path.angle_to(self.vessel.pos[:2])

        obs = np.zeros(self.nobservations)
        obs[0] = action[0] #propeller velocity,
        obs[1] = action[1] #rudder angle,
        obs[2:4] = self.vessel.v[:2 ]#velocity (x, y),
        obs[4] = self.vessel.theta[2] #yaw
        obs[5] = self.vessel.w[2] #yaw rate,
        obs[6] = vel_error #velocity error,
        obs[7] = track_error #cross track error,
        obs[8] = head_error#heading error,
        obs[9] = self.wave.A#wave amplitude,
        obs[10] = self.wave.theta #wave direction
        return obs

    def step_reward(self, obs, action):
        #placeholder until reward function is determined
        done = False
        reward = 0
        reward += obs[6] * -0.05
        reward += obs[7] * -0.05
        reward += obs[8] * -0.05
        return done, reward

    def reset(self):
        self.vessel = None
        self.dock = None
        self.path = None
        self.reward = 0
        self.past_states = []
        self.past_actions = []
        self.past_obs = []
        self.time = []
        self.total_t_steps = 0

        self.generate()

        return self.observe([0,0])

    def render(self, mode='human', close=False):
        print("DEBUG:", self.vessel.pos, self.dock, self.time[-1])
