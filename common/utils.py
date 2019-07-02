import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

# KL divergence of two univariate Gaussian distributions
def KL_divergence_mean_std(p_mean, p_std, q_mean, q_std):
    kld = torch.log(q_std/p_std) + (torch.pow(p_std) + torch.pow(p_mean - q_mean, 2))/(2 * torch.pow(q_std)) - 0.5
    return kld

# compute KL divergence of two distributions
def KL_divergence_two_dist(dist_p, dist_q):
    kld = torch.sum(dist_p * (torch.log(dist_p) - torch.log(dist_q)))
    return kld

# project value distribution onto atoms as in Categorical Algorithm
def dist_projection(optimal_dist, rewards, dones, gamma, n_atoms, Vmin, Vmax, support):
    batch_size = rewards.size(0)
    m = torch.zeros(batch_size, n_atoms)
    delta_z = (Vmax - Vmin) / (n_atoms - 1)

    for sample_idx in range(batch_size):
        reward = rewards[sample_idx]
        done = dones[sample_idx]

        for atom in range(n_atoms):
            # compute projection of Tz_j
            Tz_j = reward + (1 - done) * gamma * support[atom]
            Tz_j = torch.clamp(Tz_j, Vmin, Vmax)
            b_j = (Tz_j - Vmin) / delta_z
            l = torch.floor(b_j).long().item()
            u = torch.ceil(b_j).long().item()

            # distribute probability of Tz_j
            m[sample_idx][l] = m[sample_idx][l] + optimal_dist[sample_idx][atom] * (u - b_j)
            m[sample_idx][u] = m[sample_idx][u] + optimal_dist[sample_idx][atom] * (b_j - l)

    #print(m)
    return m


# noisy layer with independent Gaussian noise
class NoisyLinear(nn.Linear):

    def __init__(self, num_in, num_out, sigma_init=0.017):
        super(NoisyLinear, self).__init__(num_in, num_out, bias=True)
        self.sigma_weight = nn.Parameter(torch.full((num_in, num_out), sigma_init))
        self.sigma_bias = nn.Parameter(torch.full((num_out,), sigma_init))

        self.register_buffer("epsilon_weight", torch.zeros(num_out, num_in))
        self.register_buffer("epsilon_bias", torch.zeros(num_out))

        self.reset_parameters(num_in)

    def forward(self, x):
        # generate gaussian noise
        self.epsilon_weight.normal_()
        self.epsilon_bias.normal_()

        y = F.linear(x, self.weight + self.sigma_weight * self.epsilon_weight, self.sigma_bias * self.epsilon_bias)

        return y

    def reset_parameters(self, num_in):
        std = math.sqrt(3 / num_in)
        self.weight.data.uniform_(-std, std)
        self.bias.data.uniform_(-std, std)


# noisy layer for factorized Gaussion noise
class NoisyFactorizedLinear(nn.Linear):

    def __init__(self, num_in, num_out, sigma_init=0.017):
        super(NoisyFactorizedLinear, self).__init__(num_in, num_out, bias=True)
        self.sigma_weight = nn.Parameter(torch.full((num_in, num_out), sigma_init))
        self.sigma_bias = nn.Paramter(torch.full(num_out,), sigma_init)
        self.register_buffer("epsilon_i", torch.zeros(1, num_in))
        self.register_buffer("epsilon_j", torch.zeros(num_out, 1))

        self.reset_parameters(num_in)

    def forward(self, x):
        # generate guassian noise
        self.epsilon_i.normal_()
        self.epsilon_j.normal_()

        # factorize gaussian noise
        self.epsilon_i = torch.sign(self.epsilon_i) * torch.sqrt(torch.abs(self.epsilon_i))
        self.epsilon_j = torch.sign(self.epsilon_j) * torch.sqrt(torch.abs(self.epsilon_j))

        epsilon_weight = self.epsilon_i @ self.epsilon_j
        epsilon_bias = self.epsilon_j


        y = F.linear(x, self.weight + self.sigma_weight * self.epsilon_weight, self.sigma_bias * self.epsilon_bias)
        return y

    def reset_parameters(self, num_in):
        std = math.sqrt(1 / num_in)
        self.weight.data.uniform_(-std, std)
        self.weight.data.uniform_(-std, std)

