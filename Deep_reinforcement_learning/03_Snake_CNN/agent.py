import random 
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque, namedtuple

# Define a namedtuple to hold the values for state, action, next_state, and reward
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

class DQN(nn.Module):
    """
    Deep Q-Network (DQN) class, which defines the architecture of the neural network.
    It is a convolutional neural network with two hidden linear layers, followed by a final linear output layer.
    """
    def __init__(self, input_shape, n_actions): 
        super(DQN, self).__init__()
        # print("[AGENT: Input Shape]", input_shape)
        # Define Convolutional layers
        self.conv1 = self.conv = nn.Sequential(
        nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=1),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=4, stride=1),
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=3, stride=1),
        nn.ReLU()
        )
        conv_out_size = self._get_conv_output(input_shape)
        # Define fully connected layers
        self.fc = nn.Sequential(
        nn.Linear(conv_out_size, 512),
        nn.ReLU(),
        nn.Linear(512, n_actions)
        )

    def _get_conv_output(self, shape):
        # print("[AGENT:SHAPE]", shape)
        o = self.conv(torch.zeros(1, *shape))
        return torch.numel(o)

    def forward(self, x):
      conv_out = self.conv1(x)
    #   print("AGENT: Conv out shape:", conv_out.shape)  # print the output shape
      conv_out = conv_out.view(x.size()[0], -1)  # you flatten everything except the batch dimension
    #   print("AGENT: Conv out flattened shape:", conv_out.shape)  # print the flattened shape
      return self.fc(conv_out)

    
class Memory(object):
    """
    Memory class for storing past experiences.
    """
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args): 
        """
        Store the transition in memory.
        """
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        """
        Randomly sample a batch of transitions from memory.
        """
        return random.sample(self.memory, batch_size)

    def __len__(self):
        """
        Return the current size of the memory.
        """
        return len(self.memory)

class Action(): 
    """
    Class for action selection policy. It implements an epsilon-greedy strategy.
    """
    def __init__(self,  nn, action_space_len, \
                 eps_start = 0.9, eps_end = 0.005, eps_decay=25000):
        self.n = action_space_len-1
        self.steps_done = 0
        self.nn = nn
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
    
    @torch.no_grad()
    def select(self, state):
        """
        Select action according to epsilon-greedy policy.
        """
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
          math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1
        
        dir = torch.tensor(([[0,0,0]])) # TODO, this is stack size depended
        a = None
        if sample > eps_threshold:
            with torch.no_grad():
                # Exploitation: this gets the action according to the policy
                # print("[AGENT:STATE]", state.shape)
                a = self.nn(state).argmax(dim=1).numpy()
                # print("[NN AGENT:ACTION]", a)
        else:
            # Exploration: this gets a random action
                a=random.randint(0, self.n)
        
        dir[0][a] = 1
        return dir