import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.autograd 

class DQN(nn.Module):
    def __init__(self, num_in, num_out):
        super(DQN, self).__init__()
        self.num_in = num_in
        self.num_out = num_out

        self.linear1 = nn.Linear(self.num_in, 128)
        self.linear2 = nn.Linear(128, 128)
        self.linear3 = nn.Linear(128, self.num_out)

    def forward(self, state_tensor):
        qvals = F.relu(self.linear1(state_tensor))
        qvals = F.relu(self.linear2(qvals))
        qvals = self.linear3(qvals)
        
        return qvals

class CnnDQN(nn.Module):
  
    def __init__(self, input_dim, output_dim):  
        super(CnnDQN2, self).__init__()
        
        self.input_dim = input_dim
        self.action_space_dim = output_dim
        
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
        qvals = self.features(state)
        qvals = x.view(qvals.size(0), -1)
        qvals = self.fc(qvals)
        return x