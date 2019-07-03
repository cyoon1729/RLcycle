import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from common.utils import NoisyLinear


def noisy_network(input_dim, output_dim):
    noisy_net = nn.Sequential(
        NoisyLinear(input_dim, 512),
        nn.ReLU(),
        NoisyLinear(512, output_dim)
    )

    return noisy_net

def conv_net(self, input_dim):
    conv = nn.Sequential(
        nn.Conv2d(input_dim[0], 32, kernel_size=8, stride=4),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=4, stride=2),
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=3, stride=1),
        nn.ReLU()
    )
    return conv

def feature_size(self, conv_net, input_dim):
    return conv_net(autograd.Variable(torch.zeros(1, *input_dim))).view(1, -1).size(1)


class VanillaDQN(nn.Module):
    
    def __init__(self, input_dim, output_dim, use_conv=True):
        super(NoisyDQN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_conv = use_conv
        self.fc_input_dim = self.input_dim[0]
        
        if self.use_conv:
            self.conv_net = self.get_conv_net(self.input_dim)
            self.fc_input_dim = self.feature_size()

        self.fc = nn.Sequential(
            nn.Linear(self.fc_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, self.output_dim)
        )

    def forward(self, state):
        if self.use_conv:
            feats = self.conv_net(state)
            feats = feats.view(feats.size(0), -1)
        else:
            feats = state
        qvals = self.noisy_fc(feats)
        return qvals

    def get_conv_net(self, input_dim):
        conv = nn.Sequential(
            nn.Conv2d(input_dim[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        return conv
    
    def feature_size(self):
        return self.conv_net(autograd.Variable(torch.zeros(1, *self.input_dim))).view(1, -1).size(1)


class DuelingDQN(nn.Module):

    def __init__(self, input_dim, output_dim, use_conv=True, noisy=True):
        super(DuelingDQN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_conv = use_conv

        self.features = self.conv_layer(self.input_dim) if self.use_conv else nn.Sequential(
                nn.Linear(self.input_dim[0], 128),
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
            nn.Linear(128, self.output_dim)
        )

    def forward(self, state):
        feats = self.conv_features(state) if self.use_conv else self.features(state)
        values = self.value_stream(feats)
        advantages = self.advantage_stream(feats)
        qvals = values + (advantages - advantages.mean())
        return qvals

    def conv_features(self, state):
        feats = self.features(state)
        return feats.view(feats.size(0), -1)

    def conv_layer(self, input_dim):
        conv = nn.Sequential(
            nn.Conv2d(input_dim[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        return conv

    def feature_size(self, input_dim):
        return self.features(autograd.Variable(torch.zeros(1, *input_dim))).view(1, -1).size(1)


class DistributionalDQN(nn.Module):

    def __init__(self, input_dim, output_dim, use_conv=True, n_atoms=51, Vmin=-10., Vmax=10.):
        super(DistributionalDQN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_atoms = n_atoms
        self.use_conv = use_conv

        self.Vmin = Vmin
        self.Vmax = Vmax
        self.delta_z = (Vmax - Vmin) / (self.n_atoms - 1)
        self.support = torch.arange(self.Vmin, self.Vmax + self.delta_z, self.delta_z)

        self.features = self.conv_layer(self.input_dim) if self.use_conv else None
        self.fc = nn.Sequential(
            nn.Linear(self.feature_size() if self.use_conv else self.input_dim[0], 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, self.output_dim * self.n_atoms)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, state):
        batch_size = state.size()[0]
        feats = conv_features(state) if self.use_conv else state
        dist = self.fc(state).view(batch_size, -1, self.n_atoms)
        probs = self.softmax(dist)
        Qvals = torch.sum(probs * self.support, dim=2)

        return dist, Qvals

    def get_q_vals(self, state):
        dist = self.forward(state)
        probs = self.softmax(dist)
        weights = probs * self.support
        qvals = weights.sum(dim=2)
        return dist, qvals

    def conv_features(self, state):
        feats = self.features(state)
        return feats.view(feats.size(0), -1)

    def conv_layer(self, input_dim):
        conv = nn.Sequential(
            nn.Conv2d(input_dim[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
         )


class RecurrentDQN(nn.Module):

    def __init__(self, input_dim, gru_size, output_dim, use_conv=True):
        super(RecurrentDQN, self).__init__()
        self.input_dim = input_dim
        self.gru_size = gru_size
        self.output_dim = output_dim
        self.use_conv = use_conv

        self.features = self.conv_layer() if self.use_conv else None
        self.linear1 = nn.Linear(self.feature_size() if self.use_conv else self.input_dim[0], self.gru_size)
        self.gru = nn.GRUCell(self.gru_size, self.gru_size)
        self.linear2 = nn.Linear(self.gru_size, self.output_dim)

    def forward(self, state_input, hidden_state):
        feats = self.conv_features(state_input) if self.use_conv else state_input
        x = F.relu(self.linear1(feats))
        h_in = hidden_state.reshape(-1, self.gru_size)
        h = self.gru(x, h_in)
        q = self.linear2(h)
        return q, h

    def init_hidden(self):
        return self.linear1.weight.new(1, self.gru_size).zero_()

    def conv_features(self, state):
        feats = self.features(state)
        return feats.view(feats.size(0), -1)

    def conv_layer(self, input_dim):
        conv = nn.Sequential(
            nn.Conv2d(input_dim[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
         )
        return conv

    def feature_size(self, input_dim):
        return self.features(autograd.Variable(torch.zeros(1, *input_dim))).view(1, -1).size(1)


class NoisyDQN(nn.Module):
    
    def __init__(self, input_dim, output_dim, use_conv=True, noisy=True):
        super(NoisyDQN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_conv = use_conv
        self.noisy = noisy
        self.fc_input_dim = self.input_dim[0]
        
        if self.use_conv:
            self.conv_net = self.get_conv_net(self.input_dim)
            self.fc_input_dim = self.feature_size()

        if self.noisy:
            self.noisy_fc = nn.Sequential(
                NoisyLinear(self.fc_input_dim, 512),
                nn.ReLU(),
                NoisyLinear(512, self.output_dim)
            )

        if not self.noisy:
            self.fc = nn.Sequential(
                nn.Linear(self.fc_input_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 256),
                nn.ReLU(),
                nn.Linear(256, self.output_dim)
            )

    def forward(self, state):
        feats = self.conv_net(state)
        feats = feats.view(feats.size(0), -1)
        qvals = self.noisy_fc(feats)
        # print(qvals)
        return qvals

    def get_conv_net(self, input_dim):
        conv = nn.Sequential(
            nn.Conv2d(input_dim[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        return conv
    
    def feature_size(self):
        return self.conv_net(autograd.Variable(torch.zeros(1, *self.input_dim))).view(1, -1).size(1)
