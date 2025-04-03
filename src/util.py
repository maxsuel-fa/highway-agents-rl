import gymnasium as gym
from itertools import count
import matplotlib
import matplotlib.pyplot as plt
# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
plt.ion()

import pickle
import torch
from tqdm import tqdm

def train_agent(env, agent, N_episodes, device):
    durations = []
    state, _ = env.reset()
    losses = []

    for ep in range(N_episodes):
        done = False
        state, _ = env.reset()

        for t in tqdm(count()): 
            state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            action = agent.get_action(state)

            next_state, reward, terminated, truncated, _ = env.step(action.item())
            loss_val = agent.update(state, action, reward, terminated, next_state, device='cuda')

            agent.buffer.push(state, action, reward, terminated, next_state)


            state = next_state
            losses.append(loss_val)

            done = terminated or truncated
            if done:
              durations.append(t + 1)
              break


    return {
        'losses': losses,
        'durations': durations
    }

from itertools import count
import torch
import torch.nn.functional as F
from tqdm import tqdm

def train(env, agent, N_episodes, device, save_every=50, checkpoint_path="checkpoint.pth"):
    durations = []
    losses = []

    for ep in range(N_episodes):
        # Reset the environment at the start of each episode
        state, _ = env.reset()
        done = False
        # Create an indefinite tqdm bar just for this episode's steps
        # We use "with tqdm(...)" so we can manually update it
        with tqdm(desc=f"Episode {ep+1}/{N_episodes}", total=None, leave=False) as pbar:
            for t in count():
                # Convert to a tensor
                state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                # Choose an action
                action = agent.get_action(state_t)

                # Perform the action
                next_state, reward, terminated, truncated, _ = env.step(action.item())

                if terminated:
                    next_state = None
                    
                # Update the agent (DQN, etc.)
                loss_val = agent.update(state_t, action, reward, terminated, next_state, device=device)
                losses.append(loss_val)
                
                agent.buffer.push(state, action, reward, terminated, next_state)
                
                # Move to next state
                state = next_state

                # Update tqdm so we see each step
                pbar.update(1)

                # Check termination
                done = terminated or truncated
                if done:
                    durations.append(t + 1)
                    break
        agent.n_eps += 1
        agent.decrease_epsilon()
        
        if (ep + 1) % save_every == 0:
            save_checkpoint(agent, checkpoint_path)
            
    return {
        'losses': losses,
        'durations': durations
    }

def make_env(config_path):
    env = gym.make("highway-v0", render_mode="rgb_array")

    with open(config_path, "rb") as fp:
        config = pickle.load(fp)

    env.unwrapped.configure(config)
    env.reset()

    return env


def plot_durations(episode_durations):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Result')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())


def save_checkpoint(agent, filename, extra_info=None):
    """
    agent: Your DQN agent object with q_net, target_net, optimizer, buffer, etc.
    filename: Path (string) to save the checkpoint.
    extra_info: A dict of any other data you want to store (e.g. current episode, epsilon).
    """
    checkpoint = {
        'q_net': agent.q_net.state_dict(),
        'target_net': agent.target_net.state_dict(),
        'optimizer': agent.optimizer.state_dict(),
        # If your replay buffer isn't too large, you can store it directly.
        # For large buffers, consider saving it separately or in a database.
        'replay_buffer': agent.buffer.memory,
    }
    
    if extra_info is not None:
        checkpoint.update(extra_info)
    
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved to {filename}")


def load_checkpoint(agent, filename, device='cuda'):
    """
    agent: Your DQN agent object with q_net, target_net, optimizer, buffer, etc.
    filename: Path to the saved checkpoint file.
    device: 'cuda' or 'cpu'
    """
    checkpoint = torch.load(filename, map_location=device)
    
    agent.q_net.load_state_dict(checkpoint['q_net'])
    agent.target_net.load_state_dict(checkpoint['target_net'])
    agent.optimizer.load_state_dict(checkpoint['optimizer'])
    
    # If your replay buffer is not too large, restore it:
    agent.buffer.memory = checkpoint['replay_buffer']
    
    # Any extra info you stored can be retrieved:
    episode = checkpoint.get('episode', 0)
    agent.epsilon = checkpoint.get('epsilon', agent.epsilon)
    
    print(f"Checkpoint loaded from {filename}. Resuming at episode {episode} with epsilon={agent.epsilon}")
    return episode


if __name__ == '__main__':
    plot_durations(list(range(1000)))
    plt.ioff()
    plt.show()

