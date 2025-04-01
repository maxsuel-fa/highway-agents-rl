from collections import namedtuple
import gymnasium as gym
import highway_env
import torch.functional as F
import torch.nn as nn
import torch.optim as optim


Transition = namedtuple(
    'Transition',
    ('state', 'action', 'reward', 'terminated', 'next_step')
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
            action = self.get_q(state)
        else:
            action = torch.tensor(
                [[self.action_space.sample()]], device=device, dtype=torch.long
            )

        self.n_steps += 1
        self.decrease_epsilon()

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
             [s for s in batch.next_state if s is not None]
        )

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

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
        self.optimizer.step()

        if not self.n_steps % self.update_target_every:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return loss

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
        hidden_size = 128

        obs_size = self.observation_space.shape[0]
        n_actions = self.action_space.n

        self.buffer = ReplayBuffer(self.buffer_capacity)
        self.q_net =  BaseNetwork(obs_size, hidden_size, n_actions)
        self.target_net = BaseNetwork(obs_size, hidden_size, n_actions)

        self.loss_function = nn.SmoothL1Loss()
        self.optimizer = optim.Adam(params=self.q_net.parameters(), lr=self.learning_rate)

        self.epsilon = self.epsilon_start
        self.n_steps = 0
        self.n_eps = 0



if __name__ == '__main__':
    env = gym.make("highway-v0", render_mode="rgb_array")

    action_space = env.action_space
    observation_space = env.observation_space

    gamma = 0.99
    batch_size = 12
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
