from networks import PolicyNetwork

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
            with torch.no_grad():
                action = self.pi(state['observation']).sample()
                action = action.squeeze(0).cpu().numpy()
            
            next_state, reward, terminated, truncated, _ = self.env.step(action)

            states.append(state['observation'])
            actions.append(action)
            rewards.append(reward)

            state = next_state
            done = terminated or truncated

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
        
        distribution_params = self.policy_net(state_t)

        mu = distribution_params['mu']
        log_sigma = distribution_params['log_sigma']
        sigma = torch.exp(log_sigma.squeeze(0))
        
        pi = torch.distributions.MultivariateNormal(mu, torch.diag(sigma))

        return pi


    def update(self):
        """
        TODO
        """
        curr_episode = self.full_episode()
        curr_episode['losses'] = []

        T = curr_episode['states'].shape[0]
        gamma_seq = np.asarray([self.gamma ** t for t in range(T)])
        rewards = curr_episode['rewards']

        for t in range(T):
            return_g = np.multiply(gamma_seq, rewards).sum(axis=0)
            
            gamma_seq = gamma_seq[t + 1:] * (1. / (self.gamma ** t))
            rewards = rewards[t + 1:]

            state = curr_episode['states'][t]
            action = torch.as_tensor(curr_episode['actions'][t], device=self.device)
            log_prob = self.pi(state).log_prob(action)
            negative_loss = -(self.gamma ** t) * return_g * log_prob

            self.optimizer.zero_grad()
            negative_loss.backward()
            self.optimizer.step()

            curr_episode['losses'].append(-negative_loss.item())

        curr_episode['duration'] = T

        return curr_episode


    def reset(self, is_train=True):
        obs_size = self.env.observation_space['observation'].shape[0]
        action_size = self.env.action_space.shape[0]

        self.policy_net =  PolicyNetwork(obs_size, action_size).to(self.device)
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

            self.train_results['durations'] = np.append(
                self.train_results['durations'], curr_episode['duration']
            )
            self.train_results['rewards'] = np.concatenate(
                (self.train_results['rewards'], curr_episode['rewards'])
            )
            self.train_results['losses'].extend(curr_episode['losses'])
            
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
                action = self.pi(state['observation']).sample()
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
