# -*- coding: utf-8 -*-
import math
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

# Factorised NoisyLinear layer with bias
class NoisyLinear(nn.Module):
  def __init__(self, in_features, out_features, std_init=0.5):
    super(NoisyLinear, self).__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.std_init = std_init
    self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
    self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
    self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
    self.bias_mu = nn.Parameter(torch.empty(out_features))
    self.bias_sigma = nn.Parameter(torch.empty(out_features))
    self.register_buffer('bias_epsilon', torch.empty(out_features))
    self.reset_parameters()
    self.reset_noise()

  def reset_parameters(self):
    mu_range = 1 / math.sqrt(self.in_features)
    self.weight_mu.data.uniform_(-mu_range, mu_range)
    self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
    self.bias_mu.data.uniform_(-mu_range, mu_range)
    self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

  def _scale_noise(self, size):
    x = torch.randn(size, device=self.weight_mu.device)
    return x.sign().mul_(x.abs().sqrt_())

  def reset_noise(self):
    epsilon_in = self._scale_noise(self.in_features)
    epsilon_out = self._scale_noise(self.out_features)
    self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
    self.bias_epsilon.copy_(epsilon_out)

  def forward(self, input):
    if self.training:
      return F.linear(input, self.weight_mu + self.weight_sigma * self.weight_epsilon, self.bias_mu + self.bias_sigma * self.bias_epsilon)
    else:
      return F.linear(input, self.weight_mu, self.bias_mu)


class Noisy_DQN(nn.Module):
  def __init__(self, input_shape, action_space, noisy_std, device):
    super(Noisy_DQN, self).__init__()

    self.device = device
    self.convs = nn.Sequential(nn.Conv2d(input_shape[0], 32, 8, stride=4, padding=0), nn.ReLU(),
                                 nn.Conv2d(32, 64, 4, stride=2, padding=0), nn.ReLU(),
                                 nn.Conv2d(64, 64, 3, stride=1, padding=0), nn.ReLU())

    self.conv_output_size = 3136  
    self.fc1 = NoisyLinear(self.conv_output_size, 512, std_init= noisy_std)
    self.fc2 = NoisyLinear(512, action_space , std_init= noisy_std)
    
    
  def get_qvals(self, state):
    if type(state) is tuple:
        state = np.array([np.ravel(s) for s in state])

    state_t = torch.tensor(state, dtype=torch.float, device=self.device)

    return self(state_t)
  
  def forward(self, x, log=False):
    x = self.convs(x)
    
    x = F.relu(x)
    x = self.fc1(x.view(-1, self.conv_output_size))
    x = F.relu(x)
    x = self.fc2(x)
    return x


  def reset_noise(self):
    for name, module in self.named_children():
      if 'fc' in name:
        module.reset_noise()