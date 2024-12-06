import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.distributions import Normal
import numpy as np

class ContinuousPolicy(nn.Module):
    DEFAULT_HIDDEN_SIZES = [256, 256]
    
    def __init__(self, in_features:int, hidden_sizes: list = DEFAULT_HIDDEN_SIZES):
        super().__init__()
        self.log_probabilities = list()
        self.sugar_probabilities = list()
        self.states = list()
        self.rewards = list()
        self.linear1 = nn.Linear(in_features, hidden_sizes[0])
        self.linear2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.mean = nn.Linear(hidden_sizes[1], 1)
        self.log_std = nn.Linear(hidden_sizes[1], 1)
        
    def forward(self, x: torch.Tensor) -> [torch.Tensor, torch.Tensor]:
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        mean = self.mean(x)
        sd = torch.clamp(self.log_std(x), min=-20, max=2).exp()
        return mean.squeeze(), sd.squeeze()
    
    def act(self, state: torch.Tensor) -> torch.Tensor:
        mean, sd = self.forward(state)
        normal = Normal(mean, sd)
        action = normal.sample()
        log_prob = normal.log_prob(action)
        self.log_probabilities.append(log_prob)
        action = torch.clamp(action, min=0, max=0.066)
        return action.squeeze().detach().numpy()
    
    def reset(self):
        self.log_probabilities = list()
        self.rewards = list()






