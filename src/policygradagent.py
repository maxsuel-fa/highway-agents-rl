from networks import ConvPolicyNetwork

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


class REINFORCE:
    def __init__(
            self,
            env,
            gamma,
            batch_size,
            learning_rate,
            is_train=True
    ) -> None:
        """
        TODO
        """
        self.env = env
        self.gamma = gamma

        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )

        if torch.cuda.is_available():
            print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        else:
            print("CUDA not available, using CPU.")

        self.reset(is_train)

    
    def full_episode(self):
        """
        TODO
        """
        state, _ = self.env.reset()
        states, actions, rewards, =  [], [], []
        done = False

        while not done:
            #state = np.concatenate((state['observation'], state['desired_goal']))
            with torch.no_grad():
                action = self.pi(state).sample()
                action = action.squeeze(0).cpu().numpy()
            
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)

            state = next_state
            done = terminated or truncated
        print(rewards)
        return {
            'states': np.asarray(states),
            'actions': np.asarray(actions),
            'rewards': np.asarray(rewards)
        }

    
    def pi(self, state):
        """
        TODO
        """
        state_t = torch.as_tensor(state, device=self.device)
        state_t = state_t.unsqueeze(0)
        
        mu, log_sigma = self.policy_net(state_t).values()

        sigma = torch.exp(log_sigma)
        pi = torch.distributions.Normal(mu, sigma)

        return pi
    
    def compute_returns(self, rewards):
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns, dtype=torch.float32)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)
    
        return returns


    def update(self):
        """
        TODO
        """
        policy_loss = []
        all_rewards = np.asarray([])
        all_durations = np.asarray([])

        for _ in range(self.batch_size):
            curr_episode = self.full_episode()

            T = curr_episode['states'].shape[0]
            rewards = curr_episode['rewards']
            returns = self.compute_returns(rewards)
         
            for t in range(T):
                G = returns[t]

                state = curr_episode['states'][t]
                action = torch.as_tensor(curr_episode['actions'][t], device=self.device)
                log_prob = self.pi(state).log_prob(action).sum()
                policy_loss.append(-G * log_prob)

            all_rewards = np.concatenate((all_rewards, rewards))
            all_durations = np.append(all_durations, T)

        policy_loss = torch.stack(policy_loss).sum()
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

        
        return {
            'rewards': all_rewards,
            'durations': all_durations,
            'loss': -policy_loss.item()
        }


    def reset(self, is_train=True):
        obs_size = self.env.observation_space.shape[0]
        action_size = self.env.action_space.shape[0]

        self.policy_net =  ConvPolicyNetwork(obs_size, action_size).to(self.device)
        if is_train:
            self.optimizer = optim.Adam(params=self.policy_net.parameters(), lr=self.learning_rate)

        self.n_eps = 0

        self.train_results = {}
        self.best_eval = -float('inf')

    
    def save(
        self,
        save_dir,
        epoch = 'last'
    ):
        """
        TODO
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        save_path = 'weights_epoch_' + str(epoch) + '.pt'
        save_path = os.path.join(save_dir, save_path)

        rewards_t = torch.as_tensor(
            np.asarray(self.train_results['rewards']), dtype=torch.float32
        )
        durations_t = torch.as_tensor(
            np.asarray(self.train_results['durations']), dtype=torch.int64
        )
        losses_t = torch.as_tensor(
            np.asarray(self.train_results['losses']), dtype=torch.float64
        )

        torch.save(
            {
                'policy_net_state_dict': self.policy_net.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'rewards': rewards_t,
                'durations': durations_t,
                'losses': losses_t
            },
            save_path
        )

    
    def load(
        self,
        checkpoint_dir,
        epoch = 'last',
        is_train = False
    ):
        """
        TODO
        """
        cp_path = os.path.join(checkpoint_dir, 'weights_epoch_' + str(epoch) + '.pt')
        checkpoint = torch.load(cp_path, weights_only=False)
        
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])

        return checkpoint



    def train(self, train_arguments: Dict):
        writer = SummaryWriter(log_dir='./runs')
        self.train_results['durations'] = np.asarray([])
        self.train_results['rewards'] = np.asarray([])
        self.train_results['losses'] = []
        
        for episode in trange(train_arguments['n_episodes'], desc="Training Episodes"):
            curr_episode = self.update()

            self.train_results['durations'] = np.concatenate(
                (self.train_results['durations'], curr_episode['durations'])
            )
            self.train_results['rewards'] = np.concatenate(
                (self.train_results['rewards'], curr_episode['rewards'])
            )
            self.train_results['losses'].append(curr_episode['loss'])
            
            self.n_eps += 1
            #writer.add_scalar('train/episode_reward', episode_reward, self.n_eps)

            if train_arguments['eval_every'] and not self.n_eps % train_arguments['eval_every']:
                self.policy_net.eval()
                eval_res = self.eval(train_arguments['eval_n_simulations'], train_arguments['eval_display'])
                eval_rewards = eval_res['rewards']
                avg_eval = np.mean(eval_rewards)
                writer.add_scalar('eval/avg_reward', avg_eval, self.n_eps)
                print(f'Episode {self.n_eps}: Average evaluation reward: {avg_eval:.2f}')

                if train_arguments['save_best'] and avg_eval > self.best_eval:
                    self.best_eval = avg_eval
                    self.save(train_arguments['save_dir'], 'best_model')
                    print(f'Checkpoint saved at episode {self.n_eps} with best eval reward: {avg_eval:.2f}')
                self.policy_net.train()

                if train_arguments['save_every'] and not self.n_eps % train_arguments['save_every']:
                    self.save(train_arguments['save_dir'], self.n_eps)

        self.save(train_arguments['save_dir'], 'last')


    def eval(self, n_simulations, display=False):
        """
        TODO
        """
        eval_env = deepcopy(self.env)
        eval_rewards = []
        eval_durations = []

        for simulation in range(n_simulations):
            state, _ = eval_env.reset()
            sim_reward = 0.0

            for t in itertools.count():
                #state = np.concatenate((state['observation'], state['desired_goal']))
                action = self.pi(state).sample()
                action = action.squeeze(0).cpu().numpy()
                state, reward, terminated, truncated, _ = eval_env.step(action)
                sim_reward += reward

                if display:
                    eval_env.render()

                done = terminated or truncated
                if done:
                    eval_rewards.append(sim_reward)
                    eval_durations.append(t + 1)
                    break

        return {
            'rewards': eval_rewards,
            'durations': eval_durations
        }

