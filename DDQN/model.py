import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.autograd as autograd


class DDQN(nn.Module):

    def __init__(self, num_in, num_out):
        super(DDQN, self).__init__()
        self.num_in = num_in
        self.num_out = num_out

        self.features = nn.Sequential(
            nn.Linear(self.num_in, 128),
            nn.ReLU()
        )
        self.value_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_out)
        )

    def forward(self, state_tensor):
        x = self.features(state_tensor)
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        #print("value: " + str(value))
        #print("adv: " + sstate_tensor
        qvals = value + (advantage - advantage.mean())
    
        return qvals

class CnnDDQN(nn.Module):

    def __init__(self, input_dim, output_dim):  
        super(CnnDDQN, self).__init__()
        
        self.num_in = input_dim
        self.num_out = output_dim
        
        self.features = nn.Sequential(
            nn.Conv2d(self.num_in[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        self.fc_value = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, self.num_out)
        )

        self.fc_advantage = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, self.num_out)
        )
    
    def feature_size(self):
        return self.features(autograd.Variable(torch.zeros(1, *self.input_dim))).view(1, -1).size(1)
        
    def forward(self, state_tensor):
        x = self.features(state_tensor)
        x = x.view(x.size(0), -1)
        value = self.fc_value(x)
        advantage = self.fc_advantage(x)
        qvals = value + (advantage - advantage.mean())

        return qvals