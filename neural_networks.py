import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

width, height = 3, 3
state_size = width * height
nh1, nh2 = 64, 64
np1, np2 = 64, 64

class ValueNetwork(nn.Module):
    
    def __init__(self):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, nh1)
        self.fc2 = nn.Linear(nh1, nh2)
        self.fc3 = nn.Linear(nh2, 1)
    
    def forward(self, x):
        y = F.relu(self.fc1(x))
        y = F.relu(self.fc2(y))
        y = F.tanh(self.fc3(y))
        return y
    
class PolicyNetwork(nn.Module):
    
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, np1)
        self.fc2 = nn.Linear(np1, np2)
        self.fc3 = nn.Linear(np2, state_size)
    
    def forward(self, x):
        y = F.relu(self.fc1(x))
        y = F.relu(self.fc2(y))
        y = F.softmax(self.fc3(y))
        return y