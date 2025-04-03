from deepqnet import DQN
from util import *
import gymnasium as gym
import matplotlib.pyplot as plt


if __name__ == '__main__':
    env = make_env('./base_config.pkl')

    action_space = env.action_space
    observation_space = env.observation_space

    gamma = 0.99
    batch_size = 12
    buffer_capacity = 10000
    update_target_every = 32

    epsilon_start = 0.1
    decrease_epsilon_factor = 1000
    epsilon_min = 0.05

    learning_rate = 1e-2

    arguments = (
        action_space,
        observation_space,
        gamma,
        batch_size,
        buffer_capacity,
        update_target_every,
        epsilon_start,
        decrease_epsilon_factor,
        epsilon_min,
        learning_rate,
    )

    agent = DQN(*arguments)
    agent.q_net.to('cuda')
    agent.target_net.to('cuda')
    output = train(env, agent, 10, 'cuda')
    print(output['losses'])
    plt.plot(output['losses'])
    plt.show()

    plot_durations(output['durations'])

