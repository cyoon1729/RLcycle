import torch
import torch.nn as nn
import torch.nn.functional as F

from common.utils import FactorizedNoisyLinear


class ConvNoisyDQN(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(ConvNoisyDQN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.conv = nn.Sequential(
            nn.Conv2d(self.input_dim[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.noisy_fc = nn.Sequential(
            FactorizedNoisyLinear(self.feature_size(), 512),
            nn.ReLU(),
            FactorizedNoisyLinear(512, self.output_dim)
        )

    def forward(self, state):
        features = self.conv(state)
        features = features.view(features.size(0), -1)
        qvals = self.noisy_fc(features)
        
        return qvals
    
    def feature_size(self):
        return self.conv(autograd.Variable(torch.zeros(1, *self.input_dim))).view(1, -1).size(1)


class NoisyDQN(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(NoisyDQN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.noisy_fc = nn.Sequential(
            nn.Linear(self.input_dim[0], 128),
            nn.ReLU(),
            FactorizedNoisyLinear(128, 128),
            nn.ReLU(),
            FactorizedNoisyLinear(128, self.output_dim)
        )
    
    def forward(self, state):
        qvals = self.noisy_fc(state)
        
        return qvals
