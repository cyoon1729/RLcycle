import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.autograd 

class DQN(nn.module):
    def __init__(self):
        pass
    
    def forward(self, state):
        pass

    def boltzmann_action(self, state):
        pass

class CnnDQN(nn.Module):
  
    def __init__(self, input_dim, action_space_dim, num_actions=1):  
        super(CnnDQN2, self).__init__()
        
        self.input_dim = input_dim
        self.action_space_dim = action_space_dim
        self.num_actions = num_actions
        
        self.features = nn.Sequential(
            nn.Conv2d(input_dim[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        self.fc = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, self.action_space_dim)
        )
        
    def forward(self, state):
        x = self.features(state)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
    def boltzmann_action(self, state):
        state = autograd.Variable(torch.from_numpy(state).float().unsqueeze(0))
        Qvals = self.forward(state)