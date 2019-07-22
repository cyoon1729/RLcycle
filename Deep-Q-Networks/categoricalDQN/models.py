import torch
import torch.nn as nn
import torch.autograd as autograd


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