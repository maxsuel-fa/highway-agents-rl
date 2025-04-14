import gymnasium as gym 
from util import *
import highway_env
import torch
from dqnagent import DQN


env = make_env('./config/simple_config.pkl')

gamma = 0.8
batch_size = 32
buffer_capacity = 15000
update_target_every = 100

epsilon_start = 0.1
decrease_epsilon_factor = 1000
epsilon_min = 0.05
learning_rate = 1e-3

agent = DQN(
    env, 
    gamma, 
    batch_size,
    buffer_capacity,
    update_target_every,
    epsilon_start,
    decrease_epsilon_factor,
    epsilon_min,
    learning_rate
)

train_args = {
    'n_episodes': 600,
    'eval_every': 100,
    'eval_n_simulations': 5,
    'eval_display': True,
    'save_every': 100,
    'save_best': True,
    'save_dir': './weights/simple-env'
}

agent.train(train_args)
