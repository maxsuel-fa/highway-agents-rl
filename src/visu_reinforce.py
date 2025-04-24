import gymnasium as gym 
import numpy as np 
from util import *
import highway_env
import torch
from policygradagent import REINFORCE

config_dict = {
    "observation": {
        "type": "OccupancyGrid",
        "features": ['presence', 'on_road'],
        "grid_size": [[-18, 18], [-18, 18]],
        "grid_step": [3, 3],
        "as_image": False,
        "align_to_vehicle_axes": True
    },
    "action": {
        "type": "ContinuousAction",
        "longitudinal": False,
        "lateral": True
    },
    "simulation_frequency": 15,
    "policy_frequency": 5,
    "duration": 60,
    "collision_reward": -1,
    "lane_centering_cost": 4,
    "action_reward": -0.3,
    "controlled_vehicles": 1,
    "other_vehicles": 3,
    "screen_width": 600,
    "screen_height": 600,
    "centering_position": [0.5, 0.5],
    "scaling": 7,
    "show_trajectories": False,
    "render_agent": True,
    "offscreen_rendering": False
}

env = gym.make("racetrack-v0", render_mode="rgb_array", config=config_dict)
gamma = 0.8
batch_size = 1

learning_rate = 1e-3

agent = REINFORCE(
    env, 
    gamma, 
    batch_size,
    learning_rate,
    is_train=False
)

weights = './weights/race-env-new-lr'
checkpoint = agent.load(weights, 'best_model')

rewards = np.zeros((200))
print(checkpoint['durations'].shape, checkpoint['rewards'].shape)
begin = 0
for i in range(200):
    rewards[i] = checkpoint['rewards'][begin:begin + checkpoint['durations'][i]].sum()
    begin += checkpoint['durations'][i]

plot_durations(rewards, 'Episode Reward', 100)
plot_durations(checkpoint['durations'], 'Episode Duration', 100)

eval_result = agent.eval(100, False)

eval_reward = np.asarray(eval_result['rewards'])
eval_duration = np.asarray(eval_result['durations'])

print(f'Reward: {eval_reward.mean()} +- {eval_reward.std()}')
print(f'Duration: {eval_duration.mean()} +- {eval_duration.std()}')

