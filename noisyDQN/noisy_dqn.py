import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
import numpy as np

from common.utils import FactorizedNoisyLinear
from common.replay_buffers import BasicBuffer


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


class NoisyDQNAgent:

    def __init__(self, env, use_conv=True, learning_rate=1e-4, gamma=0.99, buffer_maxlen=100000):
        self.env = env
        self.use_conv = use_conv
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.replay_buffer = BasicBuffer(buffer_maxlen)

        if self.use_conv:
            self.model = ConvNoisyDQN(env.observation_space.shape, env.action_space.n)
        else:
            self.model = NoisyDQN(self.env.observation_space.shape, self.env.action_space.n) 
        
        self.MSE_loss = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def get_action(self, state):
        state = autograd.Variable(torch.FloatTensor(state).unsqueeze(0))
        qvals = self.model.forward(state)
        action = np.argmax(qvals.detach().numpy())

        return action

    def compute_loss(self, batch):
        states, actions, rewards, next_states, dones = batch
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        curr_Q = self.model.forward(states).gather(1, actions.unsqueeze(1))
        curr_Q = curr_Q.squeeze(1)
        next_Q = self.model.forward(next_states)
        max_next_Q = torch.max(next_Q, 1)[0]
        expected_Q = rewards.squeeze(1) + self.gamma * max_next_Q
        loss = self.MSE_loss(curr_Q, expected_Q)

        return loss

    def update(self, batch_size):
        batch = self.replay_buffer.sample(batch_size)
        loss = self.compute_loss(batch)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
