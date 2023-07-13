from torch import nn
import torch.nn.functional as F

# define policy network
class Policy(nn.Module):
    def __init__(self, state_space_size, action_space_size, hidden_size):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(state_space_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_space_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=1)
        return x