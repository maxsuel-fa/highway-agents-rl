from collections import deque, namedtuple
import random

Transition = namedtuple(
    'Transition',
    ('state', 'action', 'reward', 'next_state', 'done')
)


class ReplayBuffer:
    def __init__(self, capacity, min_replay_size: int = 1000):
        self.memory = deque(maxlen=capacity)
        self.min_replay_size = min_replay_size

    
    def initialize(self, env):
        state, _ = env.reset()
        for _ in range(self.min_replay_size):
            action = env.action_space.sample()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            self.push(state, action, reward, next_state, done)

            state = next_state

            if done:
                state, _ = env.reset() 


    def push(self, state, action, reward, next_state, done):
        """
        TODO
        """
        self.memory.append(
            Transition(state, action, reward, next_state, done)
        )

    def sample(self, batch_size):
        """
        TODO
        """
        return random.choices(self.memory, k=batch_size)

    def __len__(self):
        """
        TODO
        """
        return len(self.memory)
