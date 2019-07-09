import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim
import numpy as np

from common.replay_buffers import BasicBuffer
from common.utils import KL_divergence_two_dist, dist_projection
from categoricalDQN.models import DistributionalDQN

class C51Agent:

    def __init__(self, env, use_conv=True, learning_rate=3e-4, gamma=0.99, buffer_size=10000):
        self.env = env
        self.gamma = gamma
        self.replay_buffer = BasicBuffer(buffer_size)
        self.model = DistributionalDQN(self.env.observation_space.shape, self.env.action_space.n, use_conv)
        self.optimizer = torch.optim.Adam(self.model.parameters())

    def get_action(self, state):
        state = autograd.Variable(torch.from_numpy(state).float().unsqueeze(0))
        dist, qvals = self.model.forward(state)
        action = np.argmax(qvals.detach().numpy())

        return action

    def compute_error(self, batch_size):
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        curr_dist, _  = self.model.forward(states)
        curr_action_dist = curr_dist[range(batch_size), actions]

        next_dist, next_qvals = self.model.forward(next_states)
        next_actions = torch.max(next_qvals, 1)[1]
        next_dist = self.model.softmax(next_dist)
        optimal_dist = next_dist[range(batch_size), next_actions]

        projection = dist_projection(optimal_dist, rewards, dones, self.gamma, self.model.n_atoms, self.model.Vmin, self.model.Vmax, self.model.support)

        loss = -KL_divergence_two_dist(optimal_dist, projection)

        return loss

    def update(self, batch_size):

        loss = self.compute_error(batch_size)
        loss = loss.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

