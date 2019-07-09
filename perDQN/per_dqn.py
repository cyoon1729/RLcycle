import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np

from common.replay_buffers import PrioritizedBuffer
from per_dqn.models import ConvDQN, DQN

class PERAgent:

    def __init__(self, env, use_conv=True, learning_rate=3e-4, gamma=0.99, buffer_size=10000):
        self.env = env
        self.gamma = gamma
        self.replay_buffer = PrioritizedBuffer(buffer_size)
        self.model = VanillaDQN(self.env.observation_space.shape, env.action_space.n, use_conv)
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.MSE_loss = nn.MSELoss()

    def get_action(self, state):
        state = autograd.Variable(torch.from_numpy(state).float().unsqueeze(0))
        qvals = self.model.forward(state)
        action = np.argmax(qvals.detach().numpy())

        return action

    def _sample(self, batch_size):
        return self.replay_buffer.sample(batch_size)

    def _compute_TDerror(self, batch_size):
        transitions, idxs, IS_weights = self._sample(batch_size)
        states, actions, rewards, next_states, dones = transitions

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
        IS_weights = torch.FloatTensor(IS_weights)

        curr_Q = self.model.forward(states).gather(1, actions.unsqueeze(1))
        curr_Q = curr_Q.squeeze(1)
        next_Q = self.model.forward(next_states)
        max_next_Q = torch.max(next_Q, 1)[0]
        expected_Q = rewards.squeeze(1) + self.gamma * max_next_Q

        td_errors = torch.pow(expected_Q - curr_Q, 2) * IS_weights

        return td_errors, idxs

    def update(self, batch_size):
        td_errors, idxs = self._compute_TDerror(batch_size)

        # update model
        td_errors_mean = td_errors.mean()
        self.optimizer.zero_grad()
        td_errors_mean.backward()
        self.optimizer.step()

        # update priorities
        for idx, td_error in zip(idxs, td_errors.detach().numpy()):
            self.replay_buffer.update_priority(idx, td_error)
