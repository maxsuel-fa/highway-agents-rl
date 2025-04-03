from collections import namedtuple
import gymnasium as gym
import highway_env
import numpy as np
import random
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim


Transition = namedtuple(
    'Transition',
    ('state', 'action', 'reward', 'terminated', 'next_state')
)


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, state, action, reward, terminated, next_state):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(
                state, action, reward, terminated, next_state
        )
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.choices(self.memory, k=batch_size)

    def __len__(self):
        return len(self.memory)


class BaseNetwork(nn.Module):
    """
    TODO
    """
    def __init__(
            self, 
            n_observations: int, 
            n_actions: int, 
            hidden_dim: int = 128
    ) -> None:
        """
        TODO
        """
        super(BaseNetwork, self).__init__()
        self.layer1 = nn.Linear(n_observations, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, n_actions)

    def forward(self, x):
        """
        TODO
        """
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


class ConvNetwork(nn.Module):
    def __init__(self, n_actions):
        super().__init__()
        # in: [B, 7, 8, 8]
        self.conv1 = nn.Conv2d(7, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        # output shape after conv3 still [B, 64, 8, 8] if we use padding=1

        self.fc1 = nn.Linear(256 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, n_actions)

    def forward(self, x):
        # x: [B, 7, 8, 8]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = x.view(x.size(0), -1)  # flatten: [B, 64*8*8]
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class DuelingConvDQN(nn.Module):
    def __init__(self, n_actions):
        super().__init__()
        # Convolutional feature extractor
        self.conv1 = nn.Conv2d(7, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        # Fully-connected layers for the shared feature map
        self.fc_hidden = 256  # size of your hidden dimension
        self.fc = nn.Linear(64 * 8 * 8, self.fc_hidden)

        # Dueling streams
        self.value_stream = nn.Linear(self.fc_hidden, 1)
        self.advantage_stream = nn.Linear(self.fc_hidden, n_actions)

    def forward(self, x):
        # Convolutional feature extraction
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Flatten
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))

        # Dueling architecture
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)

        # Q-values: Q = Value + (Advantage - mean(Advantage, dim=1, keepdim=True))
        advantage_mean = advantage.mean(dim=1, keepdim=True)
        q_values = value + (advantage - advantage_mean)
        return q_values

class DQN:
    def __init__(
            self,
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
    ) -> None:
        self.action_space = action_space
        self.observation_space = observation_space
        self.gamma = gamma

        self.batch_size = batch_size
        self.buffer_capacity = buffer_capacity
        self.update_target_every = update_target_every

        self.epsilon_start = epsilon_start
        self.decrease_epsilon_factor = decrease_epsilon_factor
        self.epsilon_min = epsilon_min

        self.learning_rate = learning_rate

        self.reset()

    def get_action(self, state, device: str = 'cuda'):
        """
        TODO
        """
        sample = random.random()

        if sample > self.epsilon:
            action = self.q_net(state).max(1).indices.view(1, 1)
        else:
            action = torch.tensor(
                [[self.action_space.sample()]], device=device, dtype=torch.long
            )

        return action

    def update(self, *data, device):
        """
        Updates the buffer and the network(s)
        """
        if len(self.buffer) < self.batch_size:
            return np.inf
        
        transitions = self.buffer.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(
            tuple(
                map(lambda s: s is not None, batch.next_state)
            ), 
            device=device, dtype=torch.bool
        )

        non_final_next_states = torch.cat(
             [torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0) for s in batch.next_state if s is not None]
        )

        state_batch = torch.cat(
            [torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0) for s in batch.state]
        )
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(
            [torch.tensor(r, dtype=torch.float32, device=device).unsqueeze(0) for r in batch.reward]
        )

        state_action_values = self.q_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.batch_size, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values

        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        loss = self.loss_function(
                state_action_values, expected_state_action_values.unsqueeze(1)
        )

        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_value_(self.q_net.parameters(), 100)
        self.optimizer.step()

        
        if not (self.n_eps + 1) % self.update_target_every:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return loss.item()

    def get_q(self, state):
        """
        Compute Q function for a states
        """
        state_tensor = torch.tensor(state).unsqueeze(0)
        with torch.no_grad():
            output = self.q_net.forward(state_tensor) # shape (1,  n_actions)
        return output.numpy()[0]  # shape  (n_actions)

    def decrease_epsilon(self):
        self.epsilon = self.epsilon_min + (self.epsilon_start - self.epsilon_min) * (
                        np.exp(-1. * self.n_eps / self.decrease_epsilon_factor ) )

    def reset(self):
        obs_size = self.observation_space.shape[0]
        n_actions = self.action_space.n

        self.buffer = ReplayBuffer(self.buffer_capacity)
        self.q_net =  DuelingConvDQN(n_actions)
        self.target_net = DuelingConvDQN(n_actions)

        self.loss_function = nn.MSELoss()
        self.optimizer = optim.Adam(params=self.q_net.parameters(), lr=self.learning_rate)

        self.epsilon = self.epsilon_start
        self.n_steps = 0
        self.n_eps = 0



if __name__ == '__main__':
    env = gym.make("highway-v0", render_mode="rgb_array")

    action_space = env.action_space
    observation_space = env.observation_space

    gamma = 0.99
    batch_size = 1
    buffer_capacity = 10_000
    update_target_every = 32

    epsilon_start = 0.1
    decrease_epsilon_factor = 1000
    epsilon_min = 0.05

    learning_rate = 2

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
    print(agent.q_net)

