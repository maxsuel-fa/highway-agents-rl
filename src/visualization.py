import gymnasium as gym
import numpy as np
from util import *
import highway_env
from gymnasium.wrappers import RecordVideo
import torch
from dqnagent import DQN


env = make_env('./config/base_config.pkl')
env = RecordVideo(env, video_folder="./videos",
              episode_trigger=lambda e: True)

gamma = 0.8
batch_size = 32
buffer_capacity = 15000
update_target_every = 1

epsilon_start = 0.9
decrease_epsilon_factor = 1000
epsilon_min = 0.05
learning_rate = 5e-4

agent = DQN(
    env, 
    gamma, 
    batch_size,
    buffer_capacity,
    update_target_every,
    epsilon_start,
    decrease_epsilon_factor,
    epsilon_min,
    learning_rate,
    is_train=False
)

weights = './weights/base-env-new-lr-4'
checkpoint = agent.load(weights, 'last')
plot_durations(checkpoint['rewards'], 'Episode Reward', 100)
plot_durations(checkpoint['durations'], 'Episode Duration', 100)

eval_result = agent.eval(1, True)

eval_reward = np.asarray(eval_result['rewards'])
eval_duration = np.asarray(eval_result['durations'])

print(f'Reward: {eval_reward.mean()} +- {eval_reward.std()}')
print(f'Duration: {eval_duration.mean()} +- {eval_duration.std()}')
