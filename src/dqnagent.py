from memory import ReplayBuffer
from networks import ConvNetwork

from copy import deepcopy
import itertools
import numpy as np
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from typing import Dict

from tqdm import trange

class DQN:
    def __init__(
            self,
            env,
            gamma,
            batch_size,
            buffer_capacity,
            update_target_every,
            epsilon_start,
            decrease_epsilon_factor,
            epsilon_min,
            learning_rate,
    ) -> None:
        self.env = env
        self.gamma = gamma

        self.batch_size = batch_size
        self.buffer_capacity = buffer_capacity
        self.update_target_every = update_target_every

        self.epsilon_start = epsilon_start
        self.decrease_epsilon_factor = decrease_epsilon_factor
        self.epsilon_min = epsilon_min

        self.learning_rate = learning_rate

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )

        if torch.cuda.is_available():
            print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        else:
            print("CUDA not available, using CPU.")

        self.reset()

    def get_action(self, state, is_eval=False):
        """
        TODO
        """
        state_t = torch.as_tensor(state, device=self.device)
        state_t = state_t.unsqueeze(0)

        sample = random.random()
        if sample > self.epsilon or is_eval:
            with torch.no_grad():
                action = self.q_net(state_t).max(1).indices.view(1, 1)
                action = action.item()
        else:
            action = self.env.action_space.sample()

        return action

    def update(self, state, action, reward, next_state, done):
        """
        Updates the buffer and the network(s)
        """
        self.buffer.push(state, action, reward, next_state, done) 

        transition_batch = self.buffer.sample(self.batch_size)

        state_batch = torch.as_tensor( 
            np.asarray([transition.state for transition in transition_batch]),
            dtype=torch.float32, device=self.device
        ) 
        
        action_batch = torch.as_tensor( 
            np.asarray([[transition.action] for transition in transition_batch]),
            dtype=torch.int64, device=self.device
        )
        
        reward_batch = torch.as_tensor( 
            np.asarray([[transition.reward] for transition in transition_batch]),
            dtype=torch.float32, device=self.device
        )
        
        next_state_batch = torch.as_tensor( 
            np.asarray([transition.next_state for transition in transition_batch]),
            dtype=torch.float32, device=self.device
        )

        done_batch = torch.as_tensor(
            np.asarray([[transition.done] for transition in transition_batch]),
            dtype=torch.float32, device=self.device
        )

        target_q_values = self.target_net(next_state_batch) 
        max_target_q_values = target_q_values.max(dim=1, keepdim=True).values

        target = reward_batch + self.gamma * max_target_q_values * (1 - done_batch)

        q_values = self.q_net(state_batch)
        action_q_values = torch.gather(input=q_values, dim=1, index=action_batch)

        loss = self.loss_function(action_q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.n_steps += 1
        self.decrease_epsilon()

        if not self.n_steps % self.update_target_every:
            self.target_net.load_state_dict(self.q_net.state_dict())
        
        return loss.item()
        
    
    def train(self, train_arguments: Dict):
        writer = SummaryWriter(log_dir='./runs')
        self.train_results['durations'] = []
        self.train_results['rewards'] = []
        self.train_results['losses'] = []
        
        for episode in trange(train_arguments['n_episodes'], desc="Training Episodes"):
            state, _ = self.env.reset()
            episode_reward = 0.0

            for t in itertools.count():
                action = self.get_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)

                episode_reward += reward
                done = terminated or truncated

                episode_loss = self.update(state, action, reward, next_state, done)

                state = next_state

                if done:
                    self.train_results['durations'].append(t + 1)
                    self.train_results['rewards'].append(episode_reward)
                    self.train_results['losses'].append(episode_loss)
                    break

            self.n_eps += 1
            writer.add_scalar('train/episode_reward', episode_reward, self.n_eps)
            if train_arguments.get('eval_every') and self.n_eps % train_arguments['eval_every'] == 0:
                self.q_net.eval()
                eval_rewards = self.eval(train_arguments['eval_n_simulations'], train_arguments['eval_display'])
                avg_eval = np.mean(eval_rewards)
                writer.add_scalar('eval/avg_reward', avg_eval, self.n_eps)
                print(f"Episode {self.n_eps}: Average evaluation reward: {avg_eval:.2f}")
                self.q_net.train()

                # Checkpoint save (runs every save_every episodes) if eval improves
                if train_arguments.get('save_every') and self.n_eps % train_arguments['save_every'] == 0:
                    if avg_eval > self.best_eval:
                        self.best_eval = avg_eval
                        self.save(train_arguments['save_dir'], self.n_eps)
                        print(f"Checkpoint saved at episode {self.n_eps} with best eval reward: {avg_eval:.2f}")
                    else:
                        print(f"No improvement over best eval reward {self.best_eval:.2f}.")
            #deprecated
            """ if (train_arguments['save_every']
                and not self.n_eps % train_arguments['save_every']):
                self.save(train_arguments['save_dir'], self.n_eps) """



    def eval(self, n_simulations, display=False):
        """
        TODO
        """
        eval_env = deepcopy(self.env)
        eval_rewards = []

        for simulation in range(n_simulations):
            state, _ = eval_env.reset()
            sim_reward = 0.0

            for t in itertools.count():
                action = self.get_action(state, is_eval=True)
                state, reward, terminated, truncated, _ = eval_env.step(action)
                sim_reward += reward

                if display:
                    eval_env.render()

                done = terminated or truncated
                if done:
                    eval_rewards.append(sim_reward)
                    break

        return eval_rewards


    def get_q(self, state):
        """
        Compute Q function for a states
        """
        state_tensor = torch.tensor(state).unsqueeze(0)
        with torch.no_grad():
            output = self.q_net.forward(state_tensor) # shape (1,  n_actions)
        return output.numpy()[0]  # shape  (n_actions)


    def decrease_epsilon(self):
        self.epsilon = np.interp(
            self.n_steps, [0, self.decrease_epsilon_factor], 
            [self.epsilon_start, self.epsilon_min]
        )


    def reset(self):
        obs_size = self.env.observation_space.shape[0]
        n_actions = self.env.action_space.n

        self.buffer = ReplayBuffer(self.buffer_capacity)
        self.buffer.initialize(self.env)

        self.q_net =  ConvNetwork(obs_size, n_actions).to(self.device)
        self.target_net = ConvNetwork(obs_size, n_actions).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.loss_function = nn.SmoothL1Loss().to(self.device)
        self.optimizer = optim.Adam(params=self.q_net.parameters(), lr=self.learning_rate)

        self.epsilon = self.epsilon_start
        self.n_steps = 0
        self.n_eps = 0

        self.train_results = {}
        self.best_eval = -float('inf')

    
    def save(
        self,
        save_dir,
        epoch = 'latest'
    ):
        """
        TODO
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        save_path = 'weights_epoch' + str(epoch) + '.pt'
        save_path = os.path.join(save_dir, save_path)

        torch.save(
            {
                'q_net_state_dict': self.q_net.state_dict(),
                'target_state_dict': self.target_net.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'replay_buff_mem': self.buffer.memory
            },
            save_path
        )

    
    def load(
        self,
        checkpoint_dir,
        epoch = 'latest',
        is_train = False
    ):
        """
        TODO
        """
        cp_path = os.path.join(checkpoint_dir, 'weights_epoch' + str(epoch) + '.pt')
        checkpoint = torch.load(cp_path)

        self.q_net.load_state_dict(checkpoint['q_net_state_dict'])
        
        if is_train:
            self.target_net.load_state_dict(checkpoint['target_state_dict'])
            self.buffer.memory = checkpoint['replay_buff_mem']



