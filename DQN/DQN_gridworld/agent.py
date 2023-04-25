import numpy as np
import random 
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque, namedtuple

EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 20_000

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class DQN(nn.Module):
  def __init__(self, n_observations, n_actions): 
    super(DQN, self).__init__()
    self.layer1 = nn.Linear(n_observations, 128)
    self.layer2 = nn.Linear(128,128)
    self.layer3 = nn.Linear(128, n_actions)

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
    def __init__(self, action_space_len, nn):
        self.n = action_space_len
        self.steps_done = 0
        self.nn = nn
    
    def sample_action_space(self):
      return random.randint(0, self.n-1)

    def select_action(self, state):
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                a = self.nn(state).max(1)[1].view(1, 1)
                print('[NN action]', a )
                return a
        else:
            a = torch.tensor( [[self.sample_action_space()]] )
            print('[Random Action]', a)
            return a
        
    