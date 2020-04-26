import numpy as np
import matplotlib.pyplot as plt
import gym
import pandas as pd

from train import config
from docking import Docking
from stable_baselines import PPO2
import gif

def simulate(env, model, time_steps = int(1e3)):
    action = np.array([0,0])
    print_progress = int(time_steps/20)
    for step in range(time_steps):
        if i%print_progress == 0:
            print("Simulation " + str(100*i/time_steps) + "% complete")
        obs = env.observe(action)
        action = model.predict(obs, deterministic = True)[0]
        _, _, done, _ = env.step(action)

    time = np.array(env.time, ndmin = 2).reshape([time_steps, 1])
    all_info = np.hstack([time, np.array(env.past_actions), np.array(env.past_states)])
    labels = np.array(['time',
                       'propeller speed',
                       'rudder angle',
                       'x', 'y', 'z',
                       'vx', 'vy', 'vz',
                       'roll', 'pitch', 'yaw',
                       'roll rate', 'pitch rate', 'yaw rate'])
    return env.path, pd.DataFrame(all_info, columns = labels)

@gif.frame
def get_frame(path, data, i):
    max_x = max(data['x'].append(pd.Series(path.x_vals)))
    min_x = min(data['x'].append(pd.Series(path.x_vals)))
    max_y = max(data['y'].append(pd.Series(path.y_vals)))
    min_y = min(data['y'].append(pd.Series(path.y_vals)))
    x = data['x'][:i+1]
    y = data['y'][:i+1]

    plt.figure()
    plt.xlim(min_x, max_x)
    plt.ylim(min_y, max_y)
    plt.plot(x, y, 'k-', label = 'Path Taken')
    # potentially make this nicer by modifying path lol
    plt.plot(path.x_vals, path.y_vals, 'b--', label = "Path Commanded")
    plt.plot(path.points[1][0], path.points[1][1], '.', c = 'lime', ms = 10, label = "Dock")
    plt.plot(path.points[0][0], path.points[0][1], 'r.', ms = 10, label = "Start Point")
    plt.title("Reinforcement Learning Docking Simulation")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.legend()

def gif_data(path, data):
    frames = []
    num_frames = data.shape[0]
    print_progress = int(num_frames/20)
    for i in range(num_frames):
        if i%print_progress == 0:
            print("GIF " + str(100*i/num_frames) + "% complete")
        frame = get_frame(path, data, i)
        frames.append(frame)

    gif.save(frames, "path.gif", duration=1)

def plot_data(path, data):
    x = data['x']
    y = data['y']
    plt.plot(x, y, 'k-', label = 'Path Taken')
    # potentially make this nicer by modifying path lol
    plt.plot([path.points[0][0], path.points[1][0]], [path.points[0][1], path.points[1][1]], 'b--', label = "Path Commanded")
    plt.plot(path.points[1][0], path.points[1][1], '.', c = 'lime', ms = 10, label = "Dock")
    plt.plot(path.points[0][0], path.points[0][1], 'r.', ms = 10, label = "Start Point")
    plt.title("Reinforcement Learning Docking Simulation")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    env = Docking(config)
    model = PPO2.load("./model.pkl")

    time_steps = int(1e5)
    path, data = simulate(env, model, time_steps)

    plot_data(path, data)
