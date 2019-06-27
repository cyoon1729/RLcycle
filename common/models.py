import torch
import torch.nn as nn
import torch.nn.functional as F

class VanillaDQN(nn.Module):

    def __init__(self, input_dim, output_dim, use_cnn=True):
        super(VanillaDQN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_cnn = use_cnn

        self.features = self.conv_layer(self.input_dim) if self.use_cnn else None

        self.fc = nn.Sequential(
            nn.Linear(self.feature_size() if use_conv else self.input_dim[0], 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, self.output_dim)
        )

    def forward(self, state):
        feats = self.conv_features(state) if self.use_cnn else state
        qvals = self.fc(state)
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


class DuelingDQN(nn.Module):

    def __init__(self, input_dim, output_dim, use_conv=True):
        super(DuelingDQN, self).__init__:
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.features = self.conv_layer(self.input_dim) if use_conv else
            nn.Sequential(
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
            nn.Linear(128, self.num_out)
        )

    def forward(self, state):
        feats = self.conv_features(state) if use_conv else self.features(state)
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

    def __init__(self, input_dim, output_dim, use_conv=True, n_atoms=51):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_atoms = n_atoms
        self.use_conv = use_conv

        self.features = self.conv_layer(self.input_dim) if use_conv else None
        self.fc = nn.Sequential(
            nn.Linear(self.feature_size() if use_conv else self.input_dim[0], 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, self.output_dim * self.n_atoms)
        )

    def forward(self, state):
        feats = self.conv_features(state) if self.use_conv else state
        q_dist = self.fc(state)
        return q_dist

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


class RecurrentDQN(nn.Module):

    def __init__(self, input_dim, gru_size, output_dim, use_conv=True):
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
            nn.ReLU(),
         )
        return conv

     def feature_size(self, input_dim):
        return self.features(autograd.Variable(torch.zeros(1, *input_dim))).view(1, -1).size(1)



