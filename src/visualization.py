import gymnasium as gym 
from util import *
import highway_env
import torch
from dqnagent import DQN


env = make_env('./config/base_config.pkl')

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

agent.load('./weights', 600)
print(agent.eval(10, True))
