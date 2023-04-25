import numpy as np
import random 
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque, namedtuple

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class DQN(nn.Module):
  def __init__(self, n_observations, n_actions, hidden_nn_size=128): 
    super(DQN, self).__init__()
    self.layer1 = nn.Linear(n_observations, hidden_nn_size)
    self.layer2 = nn.Linear(hidden_nn_size,hidden_nn_size)
    self.layer3 = nn.Linear(hidden_nn_size, n_actions)

  def forward(self, x):
    x = F.relu(self.layer1(x))
    x = F.relu(self.layer2(x))
    return self.layer3(x)

class Memory(object):
  def __init__(self, capacity):
    self.memory = deque([], maxlen=capacity)

  def push(self, *args): 
    self.memory.append(Transition(*args))

  def sample(self, batch_size):
    return random.sample(self.memory, batch_size)

  def __len__(self):
    return len(self.memory)

class Action(): 
    def __init__(self, action_space_len, nn, eps_start = 0.9, eps_end = 0.05, eps_decay=5000):
        self.n = action_space_len
        self.steps_done = 0
        self.nn = nn
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
    
    def select(self, state):
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
          math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                a = self.nn(state).max(1)[1].view(1, 1)
                print('[NN action]', a )
                return a
        else:
            a = torch.tensor( [[random.randint(0, self.n-1)]] )
            print('[Random Action]', a)
            return a
        
    