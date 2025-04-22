import torch.nn.functional as F
import torch.nn as nn


class BaseNetwork(nn.Module):
    """
    TODO
    """
    def __init__(
            self, 
            n_observations: int, 
            n_actions: int, 
            hidden_dim: int = 128,
            last_activation = None
    ) -> None:
        """
        TODO
        """
        super(BaseNetwork, self).__init__()

        self.layer1 = nn.Linear(n_observations, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, n_actions)

        if last_activation:
            self.layer3 = nn.Sequential(
                *[self.layer3, last_activation]
            )

    def forward(self, x):
        """
        TODO
        """
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)




class ConvNetwork(nn.Module):
    def __init__(self, observation_size, n_actions):
        super().__init__()
        # in: [B, 7, 8, 8]
        self.conv1 = nn.Conv2d(observation_size, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, n_actions)

    def forward(self, x):
        # x: [B, 7, 8, 8]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # flatten: [B, 64*8*8]
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class PolicyNetwork(nn.Module):
    """
    TODO
    """
    def __init__(
            self, 
            obs_size: int, 
            action_size: int, 
            hidden_dim: int = 128,
    ) -> None:
        """
        TODO
        """
        super(PolicyNetwork, self).__init__()

        self.layer1 = nn.Linear(obs_size, hidden_dim).double()
        self.layer2 = nn.Linear(hidden_dim, hidden_dim).double()
        self.mu = nn.Linear(hidden_dim, action_size).double()
        self.log_sigma = nn.Linear(hidden_dim, action_size).double()

    def forward(self, x):
        """
        TODO
        """
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))

        mu = self.mu(x)
        log_sigma = self.log_sigma(x)

        return {
            'mu': mu,
            'log_sigma': log_sigma
        }
