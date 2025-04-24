import gymnasium as gym 
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
    "collision_reward": -2.0,
    "on_road_reward": 1.0,
    "action_reward": -0.3,
    "simulation_frequency": 15,
    "policy_frequency": 5,
    "duration": 60,
    "lane_centering_cost": 4,
    "controlled_vehicles": 1,
    "other_vehicles": 3,
    "screen_width": 600,
    "screen_height": 600,
    "centering_position": [0.5, 0.5],
    "scaling": 7,
    "show_trajectories": True,
    "render_agent": True,
    "offscreen_rendering": False
}

env = gym.make("racetrack-v0", render_mode="rgb_array")
env.unwrapped.configure(config_dict)
env.reset()

gamma = 0.8
batch_size = 5

learning_rate = 5e-4

agent = REINFORCE(
    env, 
    gamma, 
    batch_size,
    learning_rate
)

train_args = {
    'n_episodes': 600,
    'eval_every': 10,
    'eval_n_simulations': 5,
    'eval_display': True,
    'save_every': 100,
    'save_best': True,
    'save_dir': './weights/race-env-new-lr-max'
}

agent.train(train_args)
