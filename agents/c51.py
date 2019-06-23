import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import numpy as np

from common.replay_buffers import BasicBuffer
from common.models import DistributionalDQN


class C51Agent:

    def __init__(self, env, use_conv=True, learning_rate=3e-4, gamma=0.99, buffer_size=10000):
        self.env = env
        self.gamma = gamma
        self.replay_buffer = BasicBuffer
        self.model = DistributionalDQN(self.env.observation_space.shape, self.env.action_space.n)
        self.optimizer = torch.optim.Adam(self.model.parameters())

    def get_action(self, state):
        pass

    def compute_error(self, batch_size):
        pass

    def update(self, batch_size):
        pass

